"""Прямая загрузка в Cloudflare R2 (S3-совместимое хранилище) через presigned URL.

Браузер льёт файл напрямую в R2, минуя Render-прокси (у free-тира есть лимиты
на размер/длительность запроса — большой файл через сам Render не проходит).
Дальше ffmpeg/ffprobe/cv2 читают видео прямо по presigned GET URL через HTTP
Range-запросы — сервер никогда не скачивает файл на диск целиком, только то,
что нужно конкретной стадии пайплайна (проверено вручную: ffprobe/ffmpeg -ss
делают точечные range-запросы, а не тянут файл с начала до конца).
"""
import boto3
from botocore.config import Config as BotoConfig

from . import config


class StorageError(Exception):
    pass


def _client():
    if not config.R2_ENABLED:
        raise StorageError("R2 не настроен (нет R2_ACCOUNT_ID/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_BUCKET_NAME)")
    return boto3.client(
        "s3",
        endpoint_url=config.R2_ENDPOINT_URL,
        aws_access_key_id=config.R2_ACCESS_KEY_ID,
        aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )


def presigned_put_url(key: str, expires_in: int = 3600) -> str:
    """URL, в который браузер сам PUT'ит файл напрямую в R2."""
    try:
        return _client().generate_presigned_url(
            "put_object",
            Params={"Bucket": config.R2_BUCKET_NAME, "Key": key},
            ExpiresIn=expires_in,
        )
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Не удалось создать presigned PUT URL: {e}")


def presigned_get_url(key: str, expires_in: int) -> str:
    """URL, по которому ffmpeg/ffprobe/cv2 читают видео через Range-запросы."""
    try:
        return _client().generate_presigned_url(
            "get_object",
            Params={"Bucket": config.R2_BUCKET_NAME, "Key": key},
            ExpiresIn=expires_in,
        )
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Не удалось создать presigned GET URL: {e}")


def delete_object(key: str):
    """Удаляет объект из R2 — вызывается из задачи очистки старых сессий."""
    _client().delete_object(Bucket=config.R2_BUCKET_NAME, Key=key)
