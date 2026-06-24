"""Прямая загрузка в S3-совместимое хранилище (по умолчанию Yandex Object Storage)
через presigned URL.

Браузер льёт файл напрямую в хранилище, минуя Render-прокси (у free-тира есть
лимиты на размер/длительность запроса — большой файл через сам Render не
проходит). Дальше ffmpeg/ffprobe/cv2 читают видео прямо по presigned GET URL
через HTTP Range-запросы — сервер никогда не скачивает файл на диск целиком,
только то, что нужно конкретной стадии пайплайна (проверено вручную:
ffprobe/ffmpeg -ss делают точечные range-запросы, а не тянут файл с начала
до конца).
"""
import boto3
from botocore.config import Config as BotoConfig

from . import config


class StorageError(Exception):
    pass


def _client():
    if not config.S3_ENABLED:
        raise StorageError("Хранилище не настроено (нет S3_ACCESS_KEY_ID/S3_SECRET_ACCESS_KEY/S3_BUCKET_NAME)")
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT_URL,
        aws_access_key_id=config.S3_ACCESS_KEY_ID,
        aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
        config=BotoConfig(signature_version="s3v4"),
        region_name=config.S3_REGION,
    )


def presigned_put_url(key: str, expires_in: int = 3600, content_type: str | None = None) -> str:
    """URL, в который браузер сам PUT'ит файл напрямую в хранилище.

    content_type попадает в подписанные параметры — без него S3 сохраняет объект
    как application/octet-stream, и браузер при прямом открытии видео скачивает
    файл вместо проигрывания. Если передан, браузер должен прислать ТОЧНО такой
    же заголовок Content-Type на сам PUT, иначе подпись не совпадёт."""
    try:
        params = {"Bucket": config.S3_BUCKET_NAME, "Key": key}
        if content_type:
            params["ContentType"] = content_type
        return _client().generate_presigned_url("put_object", Params=params, ExpiresIn=expires_in)
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Не удалось создать presigned PUT URL: {e}")


def presigned_get_url(key: str, expires_in: int) -> str:
    """URL, по которому ffmpeg/ffprobe/cv2 читают видео через Range-запросы."""
    try:
        return _client().generate_presigned_url(
            "get_object",
            Params={"Bucket": config.S3_BUCKET_NAME, "Key": key},
            ExpiresIn=expires_in,
        )
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Не удалось создать presigned GET URL: {e}")


def delete_object(key: str):
    """Удаляет объект из хранилища — вызывается из задачи очистки старых сессий."""
    _client().delete_object(Bucket=config.S3_BUCKET_NAME, Key=key)


def create_multipart_upload(key: str, content_type: str | None = None) -> str:
    """Открывает multipart-загрузку для файлов крупнее лимита одного PUT (5 ГБ у S3). Возвращает UploadId.

    В отличие от presigned_put_url, здесь content_type не требует совпадающего
    заголовка на каждой части — он фиксируется один раз для всего объекта при
    открытии загрузки, отдельные UploadPart ничего про него не знают."""
    try:
        kwargs = {"Bucket": config.S3_BUCKET_NAME, "Key": key}
        if content_type:
            kwargs["ContentType"] = content_type
        resp = _client().create_multipart_upload(**kwargs)
        return resp["UploadId"]
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Не удалось открыть multipart-загрузку: {e}")


def presigned_upload_part_url(key: str, upload_id: str, part_number: int, expires_in: int) -> str:
    """URL, в который браузер PUT'ит одну часть файла в рамках multipart-загрузки."""
    try:
        return _client().generate_presigned_url(
            "upload_part",
            Params={"Bucket": config.S3_BUCKET_NAME, "Key": key, "UploadId": upload_id, "PartNumber": part_number},
            ExpiresIn=expires_in,
        )
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Не удалось создать presigned URL для части: {e}")


def complete_multipart_upload(key: str, upload_id: str, parts: list[dict]) -> None:
    """Завершает multipart-загрузку, склеивая части в один объект. parts: [{"part_number": int, "etag": str}, ...]."""
    try:
        _client().complete_multipart_upload(
            Bucket=config.S3_BUCKET_NAME,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={
                "Parts": [{"ETag": p["etag"], "PartNumber": p["part_number"]} for p in parts]
            },
        )
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Не удалось завершить multipart-загрузку: {e}")
