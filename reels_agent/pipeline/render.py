"""Финальный рендер клипа: crop под 9:16 + субтитры через ffmpeg `ass=` фильтр.

-ss перед -i — быстрый seek по ключевым кадрам, современный ffmpeg при этом
дотягивает до точного кадра сам (accurate seek по умолчанию), поэтому не
нужно перекодировать всё видео от начала ради точной нарезки длинного файла.
"""
import subprocess
from pathlib import Path

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
        str(output_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        return RenderResult(clip_id=candidate.id, output_path="", duration=0.0,
                             error=out.stderr.strip()[-500:])

    return RenderResult(clip_id=candidate.id, output_path=str(output_path), duration=duration)
