import pytest

from reels_agent import storage, config


def _enable_s3(monkeypatch):
    monkeypatch.setattr(config, "S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setattr(config, "S3_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setattr(config, "S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setattr(config, "S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
    monkeypatch.setattr(config, "S3_REGION", "ru-central1")
    monkeypatch.setattr(config, "S3_ENABLED", True)


class TestNotConfigured:
    def test_presigned_put_url_raises_when_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "S3_ENABLED", False)
        with pytest.raises(storage.StorageError):
            storage.presigned_put_url("some/key.mp4")

    def test_presigned_get_url_raises_when_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "S3_ENABLED", False)
        with pytest.raises(storage.StorageError):
            storage.presigned_get_url("some/key.mp4", expires_in=3600)


class TestConfigured:
    def test_presigned_put_url_returns_signed_url(self, monkeypatch):
        _enable_s3(monkeypatch)
        url = storage.presigned_put_url("session123/source.mp4")
        assert url.startswith("https://storage.yandexcloud.net/test-bucket/session123/source.mp4")
        assert "X-Amz-Signature" in url

    def test_presigned_get_url_returns_signed_url(self, monkeypatch):
        _enable_s3(monkeypatch)
        url = storage.presigned_get_url("session123/source.mp4", expires_in=7200)
        assert "X-Amz-Expires=7200" in url
        assert "X-Amz-Signature" in url

    def test_delete_object_calls_s3_delete(self, monkeypatch):
        _enable_s3(monkeypatch)
        calls = []

        class FakeClient:
            def delete_object(self, Bucket, Key):
                calls.append((Bucket, Key))

        monkeypatch.setattr(storage, "_client", lambda: FakeClient())
        storage.delete_object("session123/source.mp4")
        assert calls == [("test-bucket", "session123/source.mp4")]


class TestMultipart:
    def test_create_multipart_upload_returns_upload_id(self, monkeypatch):
        _enable_s3(monkeypatch)

        class FakeClient:
            def create_multipart_upload(self, Bucket, Key):
                return {"UploadId": "upload-abc"}

        monkeypatch.setattr(storage, "_client", lambda: FakeClient())
        assert storage.create_multipart_upload("session123/source.mp4") == "upload-abc"

    def test_presigned_upload_part_url_returns_signed_url(self, monkeypatch):
        _enable_s3(monkeypatch)
        url = storage.presigned_upload_part_url("session123/source.mp4", "upload-abc", 1, expires_in=3600)
        assert url.startswith("https://storage.yandexcloud.net/test-bucket/session123/source.mp4")
        assert "X-Amz-Signature" in url

    def test_complete_multipart_upload_calls_s3(self, monkeypatch):
        _enable_s3(monkeypatch)
        calls = []

        class FakeClient:
            def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
                calls.append((Bucket, Key, UploadId, MultipartUpload))

        monkeypatch.setattr(storage, "_client", lambda: FakeClient())
        storage.complete_multipart_upload(
            "session123/source.mp4", "upload-abc",
            [{"part_number": 1, "etag": "e1"}, {"part_number": 2, "etag": "e2"}],
        )
        assert calls == [(
            "test-bucket", "session123/source.mp4", "upload-abc",
            {"Parts": [{"ETag": "e1", "PartNumber": 1}, {"ETag": "e2", "PartNumber": 2}]},
        )]
