import io
import uuid
import zipfile

import pytest
from fastapi.testclient import TestClient

from web.server import app, _get_or_create_session
from reels_agent.job_loop import JobLoop
from reels_agent.models import ClipCandidate, RenderResult
from reels_agent import config, storage


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def no_real_pipeline_work(monkeypatch):
    """Контрактные тесты проверяют только HTTP-слой — настоящий ffmpeg/Groq не должен запускаться."""
    monkeypatch.setattr(JobLoop, "start_analysis", lambda self, path: None)
    monkeypatch.setattr(JobLoop, "start_render", lambda self: None)


def new_session_id() -> str:
    return str(uuid.uuid4())


class TestSession:
    def test_create_session_returns_id(self, client):
        resp = client.post("/api/session")
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    def test_index_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestUpload:
    def test_rejects_unsupported_extension(self, client, tmp_storage):
        sid = new_session_id()
        resp = client.post(f"/api/{sid}/upload", files={"file": ("doc.txt", b"hello", "text/plain")})
        assert resp.status_code == 400

    def test_accepts_valid_extension_and_persists_file(self, client, tmp_storage):
        sid = new_session_id()
        resp = client.post(f"/api/{sid}/upload", files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        saved = tmp_storage["uploads"] / sid / "source.mp4"
        assert saved.exists()
        assert saved.read_bytes() == b"fake video bytes"


class TestCandidatesEndpoints:
    def test_candidates_empty_initially(self, client):
        sid = new_session_id()
        resp = client.get(f"/api/{sid}/candidates")
        assert resp.status_code == 200
        body = resp.json()
        assert body["candidates"] == []
        assert body["status"] == "idle"

    def test_manual_add_missing_fields_returns_400(self, client):
        sid = new_session_id()
        resp = client.post(f"/api/{sid}/candidates/manual", json={"start": 5})
        assert resp.status_code == 400

    def test_manual_add_end_before_start_returns_400(self, client):
        sid = new_session_id()
        resp = client.post(f"/api/{sid}/candidates/manual", json={"start": 10, "end": 5})
        assert resp.status_code == 400

    def test_manual_add_success(self, client):
        sid = new_session_id()
        resp = client.post(f"/api/{sid}/candidates/manual", json={"start": 5, "end": 30})
        assert resp.status_code == 200
        candidate = resp.json()["candidate"]
        assert candidate["source"] == "manual"
        assert candidate["score"] == 999.0

        listed = client.get(f"/api/{sid}/candidates").json()["candidates"]
        assert len(listed) == 1

    def test_patch_unknown_candidate_404(self, client):
        sid = new_session_id()
        resp = client.patch(f"/api/{sid}/candidates/does-not-exist", json={"approved": True})
        assert resp.status_code == 404

    def test_patch_updates_fields(self, client):
        sid = new_session_id()
        add_resp = client.post(f"/api/{sid}/candidates/manual", json={"start": 5, "end": 30})
        cid = add_resp.json()["candidate"]["id"]

        resp = client.patch(f"/api/{sid}/candidates/{cid}",
                             json={"approved": True, "start": 6.0, "end": 31.0, "subtitle_style": "static"})
        assert resp.status_code == 200
        updated = resp.json()["candidate"]
        assert updated["approved"] is True
        assert updated["start"] == 6.0
        assert updated["end"] == 31.0
        assert updated["subtitle_style"] == "static"


class TestRenderEndpoints:
    def test_render_start_returns_ok(self, client):
        sid = new_session_id()
        resp = client.post(f"/api/{sid}/render")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_render_status_shape(self, client):
        sid = new_session_id()
        resp = client.get(f"/api/{sid}/render/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "results" in body

    def test_download_missing_clip_404(self, client):
        sid = new_session_id()
        resp = client.get(f"/api/{sid}/download/no-such-clip")
        assert resp.status_code == 404

    def test_download_existing_clip_200(self, client, tmp_path):
        sid = new_session_id()
        fake_output = tmp_path / "rendered.mp4"
        fake_output.write_bytes(b"fake mp4 bytes")

        sess = _get_or_create_session(sid)
        job: JobLoop = sess["job"]
        job.render_results["clip1"] = RenderResult(clip_id="clip1", output_path=str(fake_output), duration=10.0)

        resp = client.get(f"/api/{sid}/download/clip1")
        assert resp.status_code == 200
        assert resp.content == b"fake mp4 bytes"

    def test_download_all_empty_404(self, client):
        sid = new_session_id()
        resp = client.get(f"/api/{sid}/download_all")
        assert resp.status_code == 404

    def test_download_all_returns_zip(self, client, tmp_path):
        sid = new_session_id()
        f1 = tmp_path / "c1.mp4"
        f1.write_bytes(b"clip one")
        f2 = tmp_path / "c2.mp4"
        f2.write_bytes(b"clip two")

        sess = _get_or_create_session(sid)
        job: JobLoop = sess["job"]
        job.render_results["c1"] = RenderResult(clip_id="c1", output_path=str(f1), duration=5.0)
        job.render_results["c2"] = RenderResult(clip_id="c2", output_path=str(f2), duration=5.0)

        resp = client.get(f"/api/{sid}/download_all")
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = sorted(zf.namelist())
        assert names == ["c1.mp4", "c2.mp4"]
        assert zf.read("c1.mp4") == b"clip one"


class TestConfigEndpoint:
    def test_reports_s3_disabled_by_default(self, client, monkeypatch):
        monkeypatch.setattr(config, "S3_ENABLED", False)
        resp = client.get("/api/config")
        assert resp.status_code == 200
        assert resp.json() == {"s3_enabled": False}

    def test_reports_s3_enabled_when_configured(self, client, monkeypatch):
        monkeypatch.setattr(config, "S3_ENABLED", True)
        resp = client.get("/api/config")
        assert resp.json() == {"s3_enabled": True}


class TestPresignedUploadFlow:
    def test_upload_url_503_when_s3_disabled(self, client, monkeypatch):
        monkeypatch.setattr(config, "S3_ENABLED", False)
        sid = new_session_id()
        resp = client.post(f"/api/{sid}/upload-url", json={"filename": "clip.mp4"})
        assert resp.status_code == 503

    def test_upload_url_rejects_bad_extension(self, client, monkeypatch):
        monkeypatch.setattr(config, "S3_ENABLED", True)
        sid = new_session_id()
        resp = client.post(f"/api/{sid}/upload-url", json={"filename": "doc.txt"})
        assert resp.status_code == 400

    def test_upload_url_returns_put_url_and_stores_key(self, client, monkeypatch):
        monkeypatch.setattr(config, "S3_ENABLED", True)
        monkeypatch.setattr(storage, "presigned_put_url", lambda key, **kw: f"https://fake.r2/{key}?sig=abc")

        sid = new_session_id()
        resp = client.post(f"/api/{sid}/upload-url", json={"filename": "clip.mp4"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["key"] == f"{sid}/source.mp4"
        assert body["upload_url"] == f"https://fake.r2/{sid}/source.mp4?sig=abc"

        job = _get_or_create_session(sid)["job"]
        assert job.storage_key == f"{sid}/source.mp4"

    def test_upload_url_storage_error_returns_502(self, client, monkeypatch):
        monkeypatch.setattr(config, "S3_ENABLED", True)

        def boom(key, **kw):
            raise storage.StorageError("боком")

        monkeypatch.setattr(storage, "presigned_put_url", boom)
        sid = new_session_id()
        resp = client.post(f"/api/{sid}/upload-url", json={"filename": "clip.mp4"})
        assert resp.status_code == 502

    def test_upload_complete_without_prior_upload_url_400(self, client):
        sid = new_session_id()
        resp = client.post(f"/api/{sid}/upload-complete")
        assert resp.status_code == 400

    def test_upload_complete_triggers_analysis(self, client, monkeypatch):
        monkeypatch.setattr(config, "S3_ENABLED", True)
        monkeypatch.setattr(storage, "presigned_put_url", lambda key, **kw: f"https://fake.r2/{key}")
        monkeypatch.setattr(storage, "presigned_get_url", lambda key, expires_in: f"https://fake.r2/{key}?get=1")

        started_with = []
        monkeypatch.setattr(JobLoop, "start_analysis", lambda self, path: started_with.append(path))

        sid = new_session_id()
        client.post(f"/api/{sid}/upload-url", json={"filename": "clip.mp4"})
        resp = client.post(f"/api/{sid}/upload-complete")

        assert resp.status_code == 200
        assert started_with == [f"https://fake.r2/{sid}/source.mp4?get=1"]


class TestWebsocket:
    def test_connect_receives_greeting(self, client):
        sid = new_session_id()
        with client.websocket_connect(f"/ws/{sid}") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert sid[:8] in msg["text"]
