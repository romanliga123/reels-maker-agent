"""ffprobe-метаданные исходного видео + проверка длительности."""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .. import config


@dataclass
class ProbeResult:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


class ProbeError(Exception):
    pass


def probe_video(path: Path) -> ProbeResult:
    cmd = [
        config.FFPROBE_BIN, "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise ProbeError(f"ffprobe не найден ({config.FFPROBE_BIN})")
    except subprocess.TimeoutExpired:
        raise ProbeError("ffprobe не ответил за 60с — файл повреждён или слишком велик")

    if out.returncode != 0:
        raise ProbeError(f"ffprobe не смог прочитать файл: {out.stderr.strip()[:300]}")

    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        raise ProbeError("ffprobe вернул некорректный JSON")

    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video_stream is None:
        raise ProbeError("В файле не найдена видеодорожка")

    duration = float(data.get("format", {}).get("duration") or video_stream.get("duration") or 0)
    if duration <= 0:
        raise ProbeError("Не удалось определить длительность видео")

    if duration > config.MAX_VIDEO_DURATION_SEC:
        hours = config.MAX_VIDEO_DURATION_SEC / 3600
        raise ProbeError(f"Видео слишком длинное ({duration/3600:.1f}ч). Максимум: {hours:.1f}ч")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)

    fps_raw = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return ProbeResult(
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=audio_stream is not None,
    )
