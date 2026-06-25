"""Извлечение моно 16kHz WAV из видео — нужен и для транскрипции, и для аудио-эвристики."""
import subprocess
from pathlib import Path
from typing import Callable

from .. import config


class AudioExtractError(Exception):
    pass


def extract_audio(
    video_path: str | Path,
    out_wav_path: Path,
    total_duration_sec: float | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """video_path может быть локальным путём или presigned URL.

    on_progress(fraction_0_to_1) вызывается по ходу работы ffmpeg, если передан
    total_duration_sec (берётся из probe — длительность видео в секундах)."""
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.FFMPEG_BIN, "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-progress", "pipe:1", "-nostats",
        "-f", "wav", str(out_wav_path),
    ]
    # stderr=STDOUT (не отдельный PIPE) — иначе при достаточно "болтливом" stderr пайп
    # переполняется, а мы блокируемся на чтении stdout: classic subprocess deadlock
    # (поймано на render_clip с ass-фильтром, тут — для единообразия и на будущее).
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_lines: list[str] = []
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                value = line.strip().split("=", 1)[1] if line.startswith("out_time_ms=") else None
                # ffmpeg иногда шлёт "out_time_ms=N/A" в первых строках прогресса.
                if on_progress and total_duration_sec and value is not None and value != "N/A":
                    # ffmpeg называет это "ms", но значение на самом деле в микросекундах.
                    out_time_sec = int(value) / 1_000_000
                    on_progress(min(1.0, out_time_sec / total_duration_sec))
                else:
                    output_lines.append(line)
        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AudioExtractError("ffmpeg не успел извлечь аудио за 10 минут")
    if proc.returncode != 0:
        raise AudioExtractError(f"ffmpeg не смог извлечь аудио: {''.join(output_lines).strip()[:300]}")
    return out_wav_path
