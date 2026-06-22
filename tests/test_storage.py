import pytest

from reels_agent import storage, config


def _enable_r2(monkeypatch):
    monkeypatch.setattr(config, "R2_ACCOUNT_ID", "test-account")
    monkeypatch.setattr(config, "R2_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setattr(config, "R2_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setattr(config, "R2_BUCKET_NAME", "test-bucket")
    monkeypatch.setattr(config, "R2_ENDPOINT_URL", "https://test-account.r2.cloudflarestorage.com")
    monkeypatch.setattr(config, "R2_ENABLED", True)


class TestNotConfigured:
    def test_presigned_put_url_raises_when_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "R2_ENABLED", False)
        with pytest.raises(storage.StorageError):
            storage.presigned_put_url("some/key.mp4")

    def test_presigned_get_url_raises_when_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "R2_ENABLED", False)
        with pytest.raises(storage.StorageError):
            storage.presigned_get_url("some/key.mp4", expires_in=3600)


class TestConfigured:
    def test_presigned_put_url_returns_signed_url(self, monkeypatch):
        _enable_r2(monkeypatch)
        url = storage.presigned_put_url("session123/source.mp4")
        assert url.startswith("https://test-account.r2.cloudflarestorage.com/test-bucket/session123/source.mp4")
        assert "X-Amz-Signature" in url

    def test_presigned_get_url_returns_signed_url(self, monkeypatch):
        _enable_r2(monkeypatch)
        url = storage.presigned_get_url("session123/source.mp4", expires_in=7200)
        assert "X-Amz-Expires=7200" in url
        assert "X-Amz-Signature" in url

    def test_delete_object_calls_s3_delete(self, monkeypatch):
        _enable_r2(monkeypatch)
        calls = []

        class FakeClient:
            def delete_object(self, Bucket, Key):
                calls.append((Bucket, Key))

        monkeypatch.setattr(storage, "_client", lambda: FakeClient())
        storage.delete_object("session123/source.mp4")
        assert calls == [("test-bucket", "session123/source.mp4")]
