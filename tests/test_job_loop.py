import time
from pathlib import Path

import pytest

from reels_agent.job_loop import JobLoop
from reels_agent.pipeline.probe import ProbeError, ProbeResult
from reels_agent.pipeline.audio_extract import AudioExtractError
from reels_agent.pipeline.transcribe import TranscribeError
from reels_agent.pipeline.hook_analysis import HookAnalysisError
from reels_agent.pipeline.audio_energy import EnergySpan
from reels_agent.pipeline.hook_analysis import HookSpan
from reels_agent.models import ClipCandidate, RenderResult
from reels_agent.pipeline.face_track import CropPlan


def make_job(events):
    job = JobLoop("test-session")
    job.on_event = lambda text, kind: events.append((kind, text))
    job.source_path = Path("fake_source.mp4")
    return job


class TestRunAnalysisPipelineHappyPath:
    def test_full_pipeline_sets_ready_status(self, monkeypatch, fake_transcript):
        monkeypatch.setattr("reels_agent.job_loop.probe_video",
                            lambda p: ProbeResult(duration=30.0, width=1280, height=720, fps=24, has_audio=True))
        monkeypatch.setattr("reels_agent.job_loop.extract_audio", lambda src, dst, **kw: dst)
        monkeypatch.setattr("reels_agent.job_loop.transcribe_long_audio", lambda *a, **kw: fake_transcript)
        monkeypatch.setattr("reels_agent.job_loop.detect_energy_spans", lambda wav, **kw: [EnergySpan(5.0, 6.0, 2.0)])
        monkeypatch.setattr("reels_agent.job_loop.refine_laughter_spans", lambda spans, transcript, **kw: [None])
        monkeypatch.setattr("reels_agent.job_loop.analyze_hooks",
                            lambda transcript, **kw: [HookSpan(0.0, 4.0, "тест", "hook")])

        events = []
        job = make_job(events)
        job._run_analysis_pipeline()

        assert job.status == "ready_for_review"
        assert job.error is None
        assert job.transcript == fake_transcript
        assert len(job.candidates) >= 1
        assert events[-1][0] == "ready"

    def test_emits_progress_events_in_order(self, monkeypatch, fake_transcript):
        monkeypatch.setattr("reels_agent.job_loop.probe_video",
                            lambda p: ProbeResult(duration=30.0, width=1280, height=720, fps=24, has_audio=True))
        monkeypatch.setattr("reels_agent.job_loop.extract_audio", lambda src, dst, **kw: dst)
        monkeypatch.setattr("reels_agent.job_loop.transcribe_long_audio", lambda *a, **kw: fake_transcript)
        monkeypatch.setattr("reels_agent.job_loop.detect_energy_spans", lambda wav, **kw: [])
        monkeypatch.setattr("reels_agent.job_loop.analyze_hooks", lambda transcript, **kw: [])

        events = []
        job = make_job(events)
        job._run_analysis_pipeline()

        kinds = [k for k, _ in events]
        assert kinds[-1] == "ready"
        assert kinds[:-1] == ["progress"] * (len(kinds) - 1)
        assert len(kinds) >= 5  # видео/аудио/транскрипция/эвристика/хуки — несколько стадий прогресса

    def test_stage_progress_callbacks_forwarded_as_separate_events(self, monkeypatch, fake_transcript):
        """on_progress из пайплайн-функций должен доходить до фронтенда как kind=stage_progress,
        не как обычная строка ленты прогресса (иначе на лонг-стадиях лента будет засыпана строками)."""
        monkeypatch.setattr("reels_agent.job_loop.probe_video",
                            lambda p: ProbeResult(duration=30.0, width=1280, height=720, fps=24, has_audio=True))

        def fake_extract_audio(src, dst, **kw):
            kw["on_progress"](0.5)
            return dst

        monkeypatch.setattr("reels_agent.job_loop.extract_audio", fake_extract_audio)
        monkeypatch.setattr("reels_agent.job_loop.transcribe_long_audio", lambda *a, **kw: fake_transcript)
        monkeypatch.setattr("reels_agent.job_loop.detect_energy_spans", lambda wav, **kw: [])
        monkeypatch.setattr("reels_agent.job_loop.analyze_hooks", lambda transcript, **kw: [])

        events = []
        job = make_job(events)
        job._run_analysis_pipeline()

        stage_events = [text for kind, text in events if kind == "stage_progress"]
        assert any("50%" in text for text in stage_events)


class TestRunAnalysisPipelineErrors:
    def test_probe_error_sets_error_status(self, monkeypatch):
        monkeypatch.setattr("reels_agent.job_loop.probe_video",
                            lambda p: (_ for _ in ()).throw(ProbeError("плохой файл")))
        events = []
        job = make_job(events)
        job._run_analysis_pipeline()

        assert job.status == "error"
        assert "плохой файл" in job.error
        assert events[-1][0] == "error"

    def test_no_audio_rejected(self, monkeypatch):
        monkeypatch.setattr("reels_agent.job_loop.probe_video",
                            lambda p: ProbeResult(duration=10.0, width=640, height=360, fps=24, has_audio=False))
        events = []
        job = make_job(events)
        job._run_analysis_pipeline()

        assert job.status == "error"
        assert "аудиодорожки" in job.error
        assert job.transcript == []  # дальше пайплайн не пошёл

    def test_audio_extract_error_stops_pipeline(self, monkeypatch):
        monkeypatch.setattr("reels_agent.job_loop.probe_video",
                            lambda p: ProbeResult(duration=10.0, width=640, height=360, fps=24, has_audio=True))
        monkeypatch.setattr("reels_agent.job_loop.extract_audio",
                            lambda src, dst, **kw: (_ for _ in ()).throw(AudioExtractError("ffmpeg упал")))
        events = []
        job = make_job(events)
        job._run_analysis_pipeline()

        assert job.status == "error"
        assert "ffmpeg упал" in job.error

    def test_transcribe_error_stops_pipeline(self, monkeypatch):
        monkeypatch.setattr("reels_agent.job_loop.probe_video",
                            lambda p: ProbeResult(duration=10.0, width=640, height=360, fps=24, has_audio=True))
        monkeypatch.setattr("reels_agent.job_loop.extract_audio", lambda src, dst, **kw: dst)
        monkeypatch.setattr("reels_agent.job_loop.transcribe_long_audio",
                            lambda *a, **kw: (_ for _ in ()).throw(TranscribeError("groq недоступен")))
        events = []
        job = make_job(events)
        job._run_analysis_pipeline()

        assert job.status == "error"
        assert "groq недоступен" in job.error
        assert job.energy_spans == []  # до эвристики не дошли

    def test_hook_analysis_error_stops_pipeline(self, monkeypatch, fake_transcript):
        monkeypatch.setattr("reels_agent.job_loop.probe_video",
                            lambda p: ProbeResult(duration=10.0, width=640, height=360, fps=24, has_audio=True))
        monkeypatch.setattr("reels_agent.job_loop.extract_audio", lambda src, dst, **kw: dst)
        monkeypatch.setattr("reels_agent.job_loop.transcribe_long_audio", lambda *a, **kw: fake_transcript)
        monkeypatch.setattr("reels_agent.job_loop.detect_energy_spans", lambda wav, **kw: [])
        monkeypatch.setattr("reels_agent.job_loop.analyze_hooks",
                            lambda transcript, **kw: (_ for _ in ()).throw(HookAnalysisError("llm сломался")))
        events = []
        job = make_job(events)
        job._run_analysis_pipeline()

        assert job.status == "error"
        assert "llm сломался" in job.error
        assert job.candidates == []


class TestAddManualCandidate:
    def test_appends_and_sorts(self, fake_transcript):
        job = JobLoop("s")
        job.transcript = fake_transcript
        job.candidates = [ClipCandidate(id="x", start=0, end=20, reason="r", score=1.0, source="llm")]
        new_candidate = job.add_manual_candidate(4.0, 8.0)

        assert new_candidate.source == "manual"
        assert job.candidates[0] is new_candidate  # score 999 должен оказаться первым


class TestStartRenderGuard:
    def test_no_approved_candidates_emits_error_without_rendering(self):
        events = []
        job = make_job(events)
        job.candidates = [ClipCandidate(id="x", start=0, end=20, reason="r", score=1.0, source="llm", approved=False)]
        job.start_render()

        assert events[-1][0] == "error"
        assert job.status != "rendering"


class TestRunRenderPipeline:
    def test_happy_path_all_succeed(self, monkeypatch, fake_transcript):
        monkeypatch.setattr("reels_agent.job_loop.compute_crop_plan",
                            lambda *a, **kw: CropPlan(x=0, y=0, width=405, height=720))
        monkeypatch.setattr("reels_agent.job_loop.extract_clip_segment", lambda src, start, end, out, **kw: 0.0)
        monkeypatch.setattr(
            "reels_agent.job_loop.render_clip",
            lambda src, candidate, transcript, crop, work_dir, out_path, **kw: RenderResult(
                clip_id=candidate.id, output_path=str(out_path), duration=candidate.end - candidate.start,
            ),
        )

        events = []
        job = make_job(events)
        job.transcript = fake_transcript
        job.probe = ProbeResult(duration=30.0, width=1280, height=720, fps=24, has_audio=True)
        approved = [
            ClipCandidate(id="c1", start=0, end=15, reason="r", score=1.0, source="llm", approved=True),
            ClipCandidate(id="c2", start=15, end=30, reason="r", score=1.0, source="llm", approved=True),
        ]
        job._run_render_pipeline(approved)

        assert job.status == "done"
        assert len(job.render_results) == 2
        assert all(r.error is None for r in job.render_results.values())
        assert events[-1][0] == "render_done"
        assert "2/2" in events[-1][1]

    def test_partial_failure_still_finishes_as_done(self, monkeypatch, fake_transcript):
        monkeypatch.setattr("reels_agent.job_loop.compute_crop_plan",
                            lambda *a, **kw: CropPlan(x=0, y=0, width=405, height=720))
        monkeypatch.setattr("reels_agent.job_loop.extract_clip_segment", lambda src, start, end, out, **kw: 0.0)

        def fake_render_clip(src, candidate, transcript, crop, work_dir, out_path, **kw):
            if candidate.id == "bad":
                return RenderResult(clip_id=candidate.id, output_path="", duration=0.0, error="ffmpeg сломался")
            return RenderResult(clip_id=candidate.id, output_path=str(out_path), duration=10.0)

        monkeypatch.setattr("reels_agent.job_loop.render_clip", fake_render_clip)

        events = []
        job = make_job(events)
        job.transcript = fake_transcript
        job.probe = ProbeResult(duration=30.0, width=1280, height=720, fps=24, has_audio=True)
        approved = [
            ClipCandidate(id="bad", start=0, end=15, reason="r", score=1.0, source="llm", approved=True),
            ClipCandidate(id="good", start=15, end=30, reason="r", score=1.0, source="llm", approved=True),
        ]
        job._run_render_pipeline(approved)

        assert job.status == "done"
        assert job.render_results["bad"].error == "ffmpeg сломался"
        assert job.render_results["good"].error is None
        error_events = [e for e in events if e[0] == "error"]
        assert len(error_events) == 1
        assert "1/2" in events[-1][1]  # итоговое сообщение: 1 из 2 готов

    def test_cancel_before_loop_stops_immediately(self, monkeypatch, fake_transcript):
        monkeypatch.setattr("reels_agent.job_loop.compute_crop_plan",
                            lambda *a, **kw: CropPlan(x=0, y=0, width=405, height=720))
        monkeypatch.setattr("reels_agent.job_loop.extract_clip_segment", lambda src, start, end, out, **kw: 0.0)
        calls = []
        monkeypatch.setattr(
            "reels_agent.job_loop.render_clip",
            lambda src, candidate, transcript, crop, work_dir, out_path, **kw: calls.append(candidate.id),
        )

        events = []
        job = make_job(events)
        job.transcript = fake_transcript
        job.probe = ProbeResult(duration=30.0, width=1280, height=720, fps=24, has_audio=True)
        job.cancel_render()  # отмена до старта цикла рендера
        approved = [
            ClipCandidate(id="c1", start=0, end=15, reason="r", score=1.0, source="llm", approved=True),
        ]
        job._run_render_pipeline(approved)

        assert calls == []  # ни один клип не должен начать рендериться
        assert job.status == "cancelled"
        assert events[-1][0] == "render_done"
        assert "0/1" in events[-1][1]

    def test_cancel_between_clips_keeps_already_rendered(self, monkeypatch, fake_transcript):
        monkeypatch.setattr("reels_agent.job_loop.compute_crop_plan",
                            lambda *a, **kw: CropPlan(x=0, y=0, width=405, height=720))
        monkeypatch.setattr("reels_agent.job_loop.extract_clip_segment", lambda src, start, end, out, **kw: 0.0)

        events = []
        job = make_job(events)

        def fake_render_clip(src, candidate, transcript, crop, work_dir, out_path, **kw):
            job.cancel_render()  # отмена приходит "во время" первого клипа
            return RenderResult(clip_id=candidate.id, output_path=str(out_path), duration=10.0)

        monkeypatch.setattr("reels_agent.job_loop.render_clip", fake_render_clip)

        job.transcript = fake_transcript
        job.probe = ProbeResult(duration=30.0, width=1280, height=720, fps=24, has_audio=True)
        approved = [
            ClipCandidate(id="c1", start=0, end=15, reason="r", score=1.0, source="llm", approved=True),
            ClipCandidate(id="c2", start=15, end=30, reason="r", score=1.0, source="llm", approved=True),
        ]
        job._run_render_pipeline(approved)

        assert job.status == "cancelled"
        assert len(job.render_results) == 1  # второй клип не запускался
        assert "1/2" in events[-1][1]


class TestSpawnIsAsync:
    def test_start_analysis_does_not_block_caller(self, monkeypatch):
        def slow_probe(p):
            time.sleep(0.3)
            return ProbeResult(duration=1.0, width=1, height=1, fps=1, has_audio=False)

        monkeypatch.setattr("reels_agent.job_loop.probe_video", slow_probe)
        events = []
        job = make_job(events)

        t0 = time.time()
        job.start_analysis(Path("fake.mp4"))
        elapsed = time.time() - t0

        assert elapsed < 0.1  # вернулось мгновенно, реальная работа в фоне
        assert job.status == "analyzing"  # статус выставлен синхронно до спавна потока
