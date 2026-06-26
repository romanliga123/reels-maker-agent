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

# Прямая загрузка в S3-совместимое хранилище (по умолчанию Yandex Object Storage) —
# обходит лимиты Render-прокси на размер/длительность запроса. ffmpeg/ffprobe
# читают видео прямо по presigned GET URL через HTTP Range-запросы, без скачивания
# файла на диск целиком.
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
S3_REGION = os.getenv("S3_REGION", "ru-central1")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
S3_ENABLED = bool(S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY and S3_BUCKET_NAME)

# Один PUT-запрос в S3-совместимое хранилище ограничен 5 ГБ протоколом S3 —
# файлы крупнее этого порога заливаются через multipart upload (части по
# S3_MULTIPART_PART_SIZE_BYTES, presigned URL на каждую часть).
S3_MULTIPART_THRESHOLD_BYTES = 4 * 1024 ** 3  # 4 ГиБ — с запасом от лимита в 5 ГиБ
S3_MULTIPART_PART_SIZE_BYTES = 100 * 1024 ** 2  # 100 МиБ на часть
