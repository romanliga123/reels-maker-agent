"""Reels Maker Agent — конфигурация: пути, бинарники, константы."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
STORAGE_DIR = ROOT / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
WORK_DIR = STORAGE_DIR / "work"
OUTPUT_DIR = STORAGE_DIR / "output"

for d in (UPLOADS_DIR, WORK_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ffmpeg/ffprobe: берём из PATH, либо из явного override через переменные окружения
FFMPEG_BIN = os.getenv("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_BIN = os.getenv("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"

MAX_VIDEO_DURATION_SEC = 3.5 * 3600  # отказ, если видео длиннее
WHISPER_CHUNK_LIMIT_BYTES = 24 * 1024 * 1024  # запас от лимита Groq в 25MB
WHISPER_CHUNK_TARGET_SEC = 600  # ~10 минут на чанк транскрипции

CLIP_MIN_SEC = 15
CLIP_MAX_SEC = 90

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

SESSION_TTL_SEC = 4 * 3600  # сессии и их файлы старше этого удаляются

# Прямая загрузка в Cloudflare R2 (S3-совместимое) — обходит лимиты Render-прокси
# на размер/длительность запроса. ffmpeg/ffprobe/cv2 читают видео прямо по
# presigned GET URL через HTTP Range-запросы, без скачивания файла на диск целиком.
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "")
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else ""
R2_ENABLED = bool(R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME)
