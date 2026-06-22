import uuid

import pytest
from fastapi.testclient import TestClient

from web.server import app
from tests.conftest import has_groq_key

pytestmark = pytest.mark.network
requires_groq = pytest.mark.skipif(not has_groq_key(), reason="требуется реальный GROQ_API_KEY")


@requires_groq
def test_upload_through_real_pipeline_reaches_ready_or_clean_error(synth_video, tmp_storage):
    """Полный живой прогон: upload -> JobLoop запускает реальные ffmpeg/Groq стадии ->
    WS должен получить либо 'ready' (даже с 0 кандидатами, т.к. synth_video без речи),
    либо аккуратную 'error' — но не зависнуть и не упасть с необработанным исключением."""
    client = TestClient(app)
    session_id = str(uuid.uuid4())

    with client.websocket_connect(f"/ws/{session_id}") as ws:
        greeting = ws.receive_json()
        assert greeting["type"] == "connected"

        with open(synth_video, "rb") as f:
            resp = client.post(f"/api/{session_id}/upload",
                                files={"file": ("synth.mp4", f, "video/mp4")})
        assert resp.status_code == 200

        final_kind = None
        for _ in range(50):  # до ~50 сообщений прогресса, реальный пайплайн небыстрый
            msg = ws.receive_json(mode="text")
            if msg["type"] in ("ready", "error"):
                final_kind = msg["type"]
                break

        assert final_kind in ("ready", "error")

    status_resp = client.get(f"/api/{session_id}/candidates")
    assert status_resp.status_code == 200
