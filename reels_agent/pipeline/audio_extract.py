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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                if on_progress and total_duration_sec and line.startswith("out_time_ms="):
                    # ffmpeg называет это "ms", но значение на самом деле в микросекундах.
                    out_time_sec = int(line.strip().split("=", 1)[1]) / 1_000_000
                    on_progress(min(1.0, out_time_sec / total_duration_sec))
        stderr = proc.stderr.read() if proc.stderr else ""
        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AudioExtractError("ffmpeg не успел извлечь аудио за 10 минут")
    if proc.returncode != 0:
        raise AudioExtractError(f"ffmpeg не смог извлечь аудио: {stderr.strip()[:300]}")
    return out_wav_path
