"""JobLoop — фоновая обработка видео для одной сессии.

Аналог AgentLoop из OKR Agent: фоновые стадии пайплайна выполняются в
daemon-потоке, прогресс и результаты передаются наружу через callback
on_event(text, kind), который web/server.py подключает к WebSocket-очереди.
"""
import threading
import traceback
from pathlib import Path

from . import config
from .models import ClipCandidate, TranscriptSegment, RenderResult
from .pipeline.probe import probe_video, ProbeError, ProbeResult
from .pipeline.audio_extract import extract_audio, AudioExtractError
from .pipeline.transcribe import transcribe_long_audio, TranscribeError
from .pipeline.audio_energy import detect_energy_spans, EnergySpan
from .pipeline.hook_analysis import analyze_hooks, HookAnalysisError
from .pipeline.candidates import build_candidates, make_manual_candidate
from .pipeline.face_track import compute_crop_plan
from .pipeline.render import render_clip, extract_clip_segment, RenderError


class JobLoop:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.on_event = lambda text, kind: None  # переопределяется сервером

        self.source_path: str | Path | None = None  # локальный путь или presigned URL (S3)
        self.storage_key: str | None = None  # ключ объекта в S3, для очистки после сессии
        self.upload_id: str | None = None  # id multipart-загрузки в S3, если файл крупнее порога
        self.probe: ProbeResult | None = None
        self.transcript: list[TranscriptSegment] = []
        self.energy_spans: list[EnergySpan] = []
        self.candidates: list[ClipCandidate] = []
        self.render_results: dict[str, RenderResult] = {}
        self.status: str = "idle"  # idle | uploading | analyzing | ready_for_review | rendering | done | error | cancelled
        self.error: str | None = None

        self._lock = threading.Lock()
        self._cancel_render = threading.Event()

    def _emit(self, text: str, kind: str = "progress"):
        self.on_event(text, kind)

    def _progress_cb(self, label: str):
        """callback(fraction 0..1) -> шлёт "label NN%" не чаще, чем раз на изменившийся процент,
        отдельным WS-сообщением kind="stage_progress" (фронтенд обновляет одну строку, не плодит ленту)."""
        state = {"last_pct": -1}

        def cb(fraction: float):
            pct = max(0, min(100, int(fraction * 100)))
            if pct != state["last_pct"]:
                state["last_pct"] = pct
                self._emit(f"{label} {pct}%", "stage_progress")

        return cb

    def _spawn(self, fn, *args, **kwargs):
        def _run():
            try:
                fn(*args, **kwargs)
            except Exception as e:
                self.status = "error"
                self.error = str(e)
                self._emit(f"❌ Ошибка: {e}", "error")
                traceback.print_exc()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def start_analysis(self, source_path: str | Path):
        """Запускает стадии 2–7 (probe → транскрипция → анализ → кандидаты) в фоне.

        source_path — локальный путь (старый flow) либо presigned GET URL на R2
        (новый flow для больших файлов — ffmpeg/ffprobe/cv2 сикают по HTTP Range).
        """
        self.source_path = source_path
        self.status = "analyzing"
        self._spawn(self._run_analysis_pipeline)

    def _run_analysis_pipeline(self):
        self._emit("🔍 Проверяю видео…", "progress")
        try:
            self.probe = probe_video(self.source_path)
        except ProbeError as e:
            self.status = "error"
            self.error = str(e)
            self._emit(f"❌ {e}", "error")
            return

        if not self.probe.has_audio:
            self.status = "error"
            self.error = "В видео нет аудиодорожки — анализ невозможен"
            self._emit(f"❌ {self.error}", "error")
            return

        mins = self.probe.duration / 60
        self._emit(
            f"✅ Видео: {self.probe.width}x{self.probe.height}, {mins:.1f} мин",
            "progress",
        )

        work_dir = config.WORK_DIR / self.session_id
        wav_path = work_dir / "audio.wav"

        self._emit("🎧 Извлекаю аудио…", "progress")
        try:
            extract_audio(
                self.source_path, wav_path,
                total_duration_sec=self.probe.duration,
                on_progress=self._progress_cb("🎧 Извлекаю аудио…"),
            )
        except AudioExtractError as e:
            self.status = "error"
            self.error = str(e)
            self._emit(f"❌ {e}", "error")
            return

        self._emit("📝 Распознаю речь…", "progress")
        try:
            self.transcript = transcribe_long_audio(
                wav_path, work_dir,
                on_progress=self._progress_cb("📝 Распознаю речь…"),
            )
        except TranscribeError as e:
            self.status = "error"
            self.error = str(e)
            self._emit(f"❌ {e}", "error")
            return
        self._emit(f"✅ Распознано {len(self.transcript)} фрагментов речи", "progress")

        self._emit("😂 Ищу пики смеха и эмоций по аудио…", "progress")
        self.energy_spans = detect_energy_spans(
            wav_path, on_progress=self._progress_cb("😂 Ищу пики смеха и эмоций…"),
        )
        self._emit(f"✅ Найдено {len(self.energy_spans)} эмоциональных всплесков", "progress")

        self._emit("🧠 Анализирую транскрипт на хуки, шутки и тезисы…", "progress")
        try:
            hook_spans = analyze_hooks(
                self.transcript, on_progress=self._progress_cb("🧠 Анализирую транскрипт…"),
            )
        except HookAnalysisError as e:
            self.status = "error"
            self.error = str(e)
            self._emit(f"❌ {e}", "error")
            return

        self.candidates = build_candidates(self.transcript, self.energy_spans, hook_spans)

        self.status = "ready_for_review"
        self._emit(f"✅ Анализ завершён, {len(self.candidates)} кандидатов на клипы", "ready")

    def add_manual_candidate(self, start: float, end: float):
        candidate = make_manual_candidate(start, end, self.transcript)
        self.candidates.append(candidate)
        self.candidates.sort(key=lambda c: c.score, reverse=True)
        return candidate

    def start_render(self):
        approved = [c for c in self.candidates if c.approved]
        if not approved:
            self._emit("❌ Нет одобренных клипов для рендера", "error")
            return
        self.status = "rendering"
        self._cancel_render.clear()
        self._spawn(self._run_render_pipeline, approved)

    def cancel_render(self):
        """Просит остановиться после текущего клипа — уже запущенный ffmpeg для
        текущего клипа не прерывается (чтобы не оставлять битый недописанный файл)."""
        self._cancel_render.set()
        self._emit("⏹ Отмена рендера запрошена — остановлюсь после текущего клипа…", "progress")

    def _run_render_pipeline(self, approved: list[ClipCandidate]):
        work_dir = config.WORK_DIR / self.session_id
        out_dir = config.OUTPUT_DIR / self.session_id
        total = len(approved)
        cancelled = False

        for i, candidate in enumerate(approved, start=1):
            if self._cancel_render.is_set():
                cancelled = True
                break

            done_pct = round((i - 1) / total * 100)
            self._emit(f"🎬 Рендерю клип {i}/{total} ({done_pct}%)…", "progress")

            segment_path = work_dir / f"{candidate.id}_segment.mp4"
            try:
                # Сначала вырезаем маленький локальный кусок вокруг клипа — дальше
                # поиск лица и финальный рендер работают с ним, а не качают/сикают
                # по сети весь многогигабайтный источник (была причина OOM на рендере).
                seg_start = extract_clip_segment(self.source_path, candidate.start, candidate.end, segment_path)

                crop = compute_crop_plan(
                    segment_path, candidate.start - seg_start, candidate.end - seg_start,
                    self.probe.width, self.probe.height,
                    on_progress=self._progress_cb(f"🧭 Клип {i}/{total}, ищу лицо:"),
                )
                result = render_clip(
                    segment_path, candidate, self.transcript, crop,
                    work_dir, out_dir / f"{candidate.id}.mp4",
                    on_progress=self._progress_cb(f"🎬 Клип {i}/{total}:"),
                    cut_start=candidate.start - seg_start,
                )
            except RenderError as e:
                result = RenderResult(clip_id=candidate.id, output_path="", duration=0.0, error=str(e))
            finally:
                segment_path.unlink(missing_ok=True)

            self.render_results[candidate.id] = result
            if result.error:
                self._emit(f"❌ Клип {i}/{total} не отрендерился: {result.error[:200]}", "error")
            else:
                self._emit(f"✅ Клип {i}/{total} готов", "progress")

        ok_count = sum(1 for r in self.render_results.values() if not r.error)
        if cancelled:
            self.status = "cancelled"
            self._emit(f"⏹ Рендер отменён: {ok_count}/{total} клипов готовы", "render_done")
        else:
            self.status = "done"
            self._emit(f"✅ Рендер завершён: {ok_count}/{total} клипов готовы", "render_done")
