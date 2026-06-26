"""Финальный рендер клипа: crop под 9:16 + субтитры через ffmpeg `ass=` фильтр.

-ss перед -i — быстрый seek по ключевым кадрам, современный ffmpeg при этом
дотягивает до точного кадра сам (accurate seek по умолчанию), поэтому не
нужно перекодировать всё видео от начала ради точной нарезки длинного файла.
"""
import subprocess
from pathlib import Path
from typing import Callable

from .. import config
from ..models import ClipCandidate, TranscriptSegment, RenderResult
from .face_track import CropPlan
from .subtitles import build_ass


class RenderError(Exception):
    pass


def extract_clip_segment(
    source_path: str | Path,
    start: float,
    end: float,
    out_path: Path,
    pad_sec: float = 10.0,
) -> float:
    """Стрим-копия (без перекодирования) широкого диапазона вокруг клипа в локальный
    файл. Дальше поиск лица и финальный рендер этого клипа работают с этим маленьким
    локальным файлом вместо presigned URL на весь многогигабайтный источник — это
    убирает сетевые seek'и и непредсказуемую буферизацию у ffmpeg при чтении
    удалённого потока, которые были источником OOM на рендере.

    pad_sec — запас с каждой стороны, т.к. `-c copy` режет только по ключевым кадрам
    (snap к предыдущему keyframe), а не точно по запрошенному start. `-avoid_negative_ts
    make_zero` гарантирует, что первый кадр в out_path лежит ровно на локальном времени 0
    — поэтому возвращаем seg_start (абсолютное время начала сегмента в таймлайне источника),
    чтобы вызывающий код мог пересчитать candidate.start/end в локальные координаты
    (local = absolute - seg_start) для render_clip(cut_start=...).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seg_start = max(0.0, start - pad_sec)
    seg_duration = (end - start) + 2 * pad_sec
    cmd = [
        config.FFMPEG_BIN, "-y",
        "-ss", str(seg_start), "-i", str(source_path), "-t", str(seg_duration),
        "-c", "copy", "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RenderError(f"Не удалось вырезать сегмент клипа: {proc.stderr.strip()[-300:]}")
    return seg_start


def _escape_filter_path(path: Path) -> str:
    # У ass-фильтра два слоя парсинга: внешний filtergraph-парсер (делит по ':' между
    # фильтрами/опциями) и внутренний парсер самого ass (делит filename:opt=val).
    # Внешний слой снимает один уровень '\' при разэкранировании, поэтому букву диска
    # (C:) нужно экранировать ДВОЙНым бэкслэшем — иначе после первого слоя дойдёт
    # обычное ':' и внутренний парсер ass примет остаток пути за имя опции
    # (например "original_size") и упадёт с ошибкой парсинга.
    s = str(path).replace("\\", "/")
    s = s.replace(":", "\\\\:")
    return s


def render_clip(
    source_path: str | Path,
    candidate: ClipCandidate,
    transcript: list[TranscriptSegment],
    crop: CropPlan,
    work_dir: Path,
    output_path: Path,
    on_progress: Callable[[float], None] | None = None,
    cut_start: float | None = None,
) -> RenderResult:
    """cut_start — позиция seek'а в source_path, если он отличается от
    candidate.start (например, source_path — локальный сегмент, вырезанный
    extract_clip_segment, со своей таймлинией). Субтитры всё равно строятся по
    абсолютным candidate.start/end — они привязаны к таймкодам транскрипта,
    а не к конкретному видеофайлу."""
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cut_start is None:
        cut_start = candidate.start

    ass_content = build_ass(transcript, candidate.start, candidate.end, candidate.subtitle_style)
    ass_path = work_dir / f"{candidate.id}.ass"
    ass_path.write_text(ass_content, encoding="utf-8")

    duration = candidate.end - candidate.start
    vf = (
        f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y},"
        f"scale={config.OUTPUT_WIDTH}:{config.OUTPUT_HEIGHT}:flags=lanczos,"
        f"ass={_escape_filter_path(ass_path)}"
    )

    cmd = [
        config.FFMPEG_BIN, "-y",
        "-ss", str(cut_start), "-i", str(source_path), "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats",
        str(output_path),
    ]
    # stderr=STDOUT (не отдельный PIPE) — иначе при бойком выводе ffmpeg в stderr
    # (libass логирует сканирование системных шрифтов для ass=) пайп переполняется,
    # а мы в этот момент блокируемся на чтении stdout: классический deadlock
    # subprocess. Один общий поток читаем построчно без риска зависания.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_lines: list[str] = []
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                value = line.strip().split("=", 1)[1] if line.startswith("out_time_ms=") else None
                # ffmpeg иногда шлёт "out_time_ms=N/A" в первых строках прогресса,
                # до того как появятся реальные данные — пропускаем такие строки.
                if on_progress and duration > 0 and value is not None and value != "N/A":
                    out_time_sec = int(value) / 1_000_000
                    on_progress(min(1.0, out_time_sec / duration))
                else:
                    output_lines.append(line)
        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        return RenderResult(clip_id=candidate.id, output_path="", duration=0.0,
                             error="ffmpeg не успел отрендерить клип за 10 минут")
    if proc.returncode != 0:
        return RenderResult(clip_id=candidate.id, output_path="", duration=0.0,
                             error="".join(output_lines).strip()[-500:])

    return RenderResult(clip_id=candidate.id, output_path=str(output_path), duration=duration)
