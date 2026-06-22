"""Извлечение моно 16kHz WAV из видео — нужен и для транскрипции, и для аудио-эвристики."""
import subprocess
from pathlib import Path

from .. import config


class AudioExtractError(Exception):
    pass


def extract_audio(video_path: str | Path, out_wav_path: Path) -> Path:
    """video_path может быть локальным путём или presigned URL."""
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.FFMPEG_BIN, "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", str(out_wav_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise AudioExtractError(f"ffmpeg не смог извлечь аудио: {out.stderr.strip()[:300]}")
    return out_wav_path
