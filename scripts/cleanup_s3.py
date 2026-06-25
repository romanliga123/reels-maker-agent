"""Разовая чистка бакета: удаляет все незавершённые multipart-загрузки и (опционально)
все обычные объекты. Консоль Yandex Cloud не показывает незавершённые multipart-загрузки
как объекты, но они занимают место в квоте бакета — отсюда "бакет полон" при пустом
списке объектов в UI.

В проекте нет автозагрузки .env (только реальные переменные окружения, как на
Render) — перед запуском задай те же S3_* значения, что стоят в Render Dashboard,
в текущей PowerShell-сессии:

    $env:S3_ACCESS_KEY_ID = "..."
    $env:S3_SECRET_ACCESS_KEY = "..."
    $env:S3_BUCKET_NAME = "reels-maker-agent"
    python scripts/cleanup_s3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reels_agent import config

import boto3
from botocore.config import Config as BotoConfig


def _client():
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT_URL,
        aws_access_key_id=config.S3_ACCESS_KEY_ID,
        aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
        config=BotoConfig(signature_version="s3v4"),
        region_name=config.S3_REGION,
    )


def abort_incomplete_uploads(client, bucket: str) -> int:
    count = 0
    paginator = client.get_paginator("list_multipart_uploads")
    for page in paginator.paginate(Bucket=bucket):
        for upload in page.get("Uploads", []):
            key = upload["Key"]
            upload_id = upload["UploadId"]
            client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            print(f"  Удалена незавершённая загрузка: {key} ({upload_id})")
            count += 1
    return count


def delete_all_objects(client, bucket: str) -> int:
    count = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objects = page.get("Contents", [])
        if not objects:
            continue
        keys = [{"Key": o["Key"]} for o in objects]
        client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
        for o in objects:
            print(f"  Удалён объект: {o['Key']}")
        count += len(objects)
    return count


def main():
    if not config.S3_ENABLED:
        print("S3 не настроен (нет S3_ACCESS_KEY_ID/S3_SECRET_ACCESS_KEY/S3_BUCKET_NAME) — проверь .env")
        return

    client = _client()
    bucket = config.S3_BUCKET_NAME
    print(f"Бакет: {bucket}")

    print("Ищу и удаляю незавершённые multipart-загрузки…")
    aborted = abort_incomplete_uploads(client, bucket)
    print(f"Удалено незавершённых загрузок: {aborted}")

    print("Ищу и удаляю обычные объекты…")
    deleted = delete_all_objects(client, bucket)
    print(f"Удалено объектов: {deleted}")

    print("Готово.")


if __name__ == "__main__":
    main()
