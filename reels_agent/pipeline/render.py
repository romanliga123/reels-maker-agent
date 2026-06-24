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
) -> RenderResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        "-ss", str(candidate.start), "-i", str(source_path), "-t", str(duration),
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
                if on_progress and duration > 0 and line.startswith("out_time_ms="):
                    out_time_sec = int(line.strip().split("=", 1)[1]) / 1_000_000
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
