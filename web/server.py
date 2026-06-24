"""
Reels Maker Agent — Web Server (FastAPI)

Запуск:
    python start_web.py
"""
import sys
import uuid
import json
import math
import time
import asyncio
import dataclasses
import threading
import queue as _queue
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, RedirectResponse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from reels_agent.job_loop import JobLoop
from reels_agent import config, storage

app = FastAPI(title="Reels Maker Agent")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# session_id -> {"job": JobLoop, "created": float}
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _get_or_create_session(session_id: str) -> dict:
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = {
                "job": JobLoop(session_id),
                "created": time.time(),
            }
        return _sessions[session_id]


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html, headers={"Cache-Control": "no-store"})


@app.post("/api/session")
async def create_session():
    session_id = str(uuid.uuid4())
    _get_or_create_session(session_id)
    return {"session_id": session_id}


@app.get("/api/config")
async def get_config():
    """Фронтенд решает, каким способом грузить файл: напрямую в S3-хранилище
    (большие файлы, нет лимита Render-прокси) или старым способом через сервер
    (если хранилище не настроено)."""
    return {"s3_enabled": config.S3_ENABLED}


@app.post("/api/{session_id}/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    sess = _get_or_create_session(session_id)
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Тип файла не поддерживается: {suffix}")

    dest_dir = config.UPLOADS_DIR / session_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"source{suffix}"

    with open(dest_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    job: JobLoop = sess["job"]
    job.start_analysis(dest_path)
    return {"ok": True, "filename": file.filename}


ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
VIDEO_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
}


@app.post("/api/{session_id}/upload-url")
async def get_upload_url(session_id: str, body: dict):
    """Возвращает presigned PUT URL — браузер льёт файл напрямую в S3-хранилище, минуя сервер.

    Файлы крупнее S3_MULTIPART_THRESHOLD_BYTES (по умолчанию 4 ГиБ) не лезут в один
    PUT — у протокола S3 жёсткий лимит 5 ГиБ на объект за один PUT. Для них отдаём
    набор presigned URL на части (multipart upload)."""
    if not config.S3_ENABLED:
        raise HTTPException(status_code=503, detail="Хранилище не настроено на сервере")

    filename = str(body.get("filename", "source.mp4"))
    size = int(body.get("size", 0))
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Тип файла не поддерживается: {suffix}")

    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]
    key = f"{session_id}/source{suffix}"
    job.storage_key = key
    content_type = VIDEO_CONTENT_TYPES.get(suffix, "application/octet-stream")

    if size > config.S3_MULTIPART_THRESHOLD_BYTES:
        part_size = config.S3_MULTIPART_PART_SIZE_BYTES
        num_parts = math.ceil(size / part_size)
        try:
            upload_id = storage.create_multipart_upload(key, content_type=content_type)
            parts = [
                {
                    "part_number": n,
                    "url": storage.presigned_upload_part_url(key, upload_id, n, expires_in=config.SESSION_TTL_SEC),
                }
                for n in range(1, num_parts + 1)
            ]
        except storage.StorageError as e:
            raise HTTPException(status_code=502, detail=str(e))
        job.upload_id = upload_id
        return {"mode": "multipart", "key": key, "upload_id": upload_id, "part_size": part_size, "parts": parts}

    try:
        put_url = storage.presigned_put_url(key, content_type=content_type)
    except storage.StorageError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"mode": "single", "upload_url": put_url, "key": key, "content_type": content_type}


@app.post("/api/{session_id}/upload-complete")
async def upload_complete(session_id: str, body: dict = Body(default={})):
    """Браузер сообщает, что файл уже долетел до хранилища — запускаем анализ по presigned GET URL.

    Для multipart-загрузки в теле запроса должны быть части с ETag (их отдаёт S3
    в ответ на каждый PUT части) — без этого S3 не сможет склеить файл."""
    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]
    if not job.storage_key:
        raise HTTPException(status_code=400, detail="upload-url не был запрошен для этой сессии")

    if job.upload_id:
        parts = body.get("parts")
        if not parts:
            raise HTTPException(status_code=400, detail="Не переданы части multipart-загрузки")
        try:
            storage.complete_multipart_upload(job.storage_key, job.upload_id, parts)
        except storage.StorageError as e:
            raise HTTPException(status_code=502, detail=str(e))

    try:
        get_url = storage.presigned_get_url(job.storage_key, expires_in=config.SESSION_TTL_SEC + 3600)
    except storage.StorageError as e:
        raise HTTPException(status_code=502, detail=str(e))

    job.start_analysis(get_url)
    return {"ok": True}


@app.get("/api/{session_id}/candidates")
async def list_candidates(session_id: str):
    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]
    return {
        "status": job.status,
        "error": job.error,
        "candidates": [dataclasses.asdict(c) for c in job.candidates],
    }


@app.get("/api/{session_id}/preview")
async def preview_video(session_id: str):
    """Отдаёт исходное видео для предпросмотра кандидата (без рендера) — браузер
    сикает по таймкоду через media fragment (#t=start,end) в <video>/при навигации."""
    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]

    if config.S3_ENABLED and job.storage_key:
        try:
            url = storage.presigned_get_url(job.storage_key, expires_in=3600)
        except storage.StorageError as e:
            raise HTTPException(status_code=502, detail=str(e))
        return RedirectResponse(url)

    if isinstance(job.source_path, Path) and job.source_path.exists():
        return FileResponse(job.source_path)

    raise HTTPException(status_code=404, detail="Видео не найдено — анализ ещё не запускался?")


@app.post("/api/{session_id}/candidates/manual")
async def add_manual_candidate(session_id: str, body: dict):
    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]
    try:
        start = float(body["start"])
        end = float(body["end"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Нужны числовые поля start и end")
    if end <= start:
        raise HTTPException(status_code=400, detail="end должен быть больше start")
    candidate = job.add_manual_candidate(start, end)
    return {"candidate": dataclasses.asdict(candidate)}


@app.patch("/api/{session_id}/candidates/{candidate_id}")
async def update_candidate(session_id: str, candidate_id: str, body: dict):
    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]
    candidate = next((c for c in job.candidates if c.id == candidate_id), None)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    if "approved" in body:
        candidate.approved = bool(body["approved"])
    if "start" in body:
        candidate.start = float(body["start"])
    if "end" in body:
        candidate.end = float(body["end"])
    if "subtitle_style" in body:
        candidate.subtitle_style = str(body["subtitle_style"])
    return {"candidate": dataclasses.asdict(candidate)}


@app.post("/api/{session_id}/render")
async def start_render(session_id: str):
    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]
    job.start_render()
    return {"ok": True}


@app.post("/api/{session_id}/render/cancel")
async def cancel_render(session_id: str):
    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]
    job.cancel_render()
    return {"ok": True}


@app.get("/api/{session_id}/render/status")
async def render_status(session_id: str):
    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]
    return {
        "status": job.status,
        "error": job.error,
        "results": {cid: dataclasses.asdict(r) for cid, r in job.render_results.items()},
    }


@app.get("/api/{session_id}/download/{clip_id}")
async def download_clip(session_id: str, clip_id: str):
    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]
    result = job.render_results.get(clip_id)
    if not result or result.error or not Path(result.output_path).exists():
        raise HTTPException(status_code=404, detail="Клип не найден или не отрендерился")
    return FileResponse(result.output_path, filename=f"{clip_id}.mp4", media_type="video/mp4")


@app.get("/api/{session_id}/download_all")
async def download_all(session_id: str):
    import io
    import zipfile

    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]
    ready = [r for r in job.render_results.values() if not r.error and Path(r.output_path).exists()]
    if not ready:
        raise HTTPException(status_code=404, detail="Нет готовых клипов для скачивания")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in ready:
            zf.write(r.output_path, arcname=f"{r.clip_id}.mp4")
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=reels_{session_id[:8]}.zip"},
    )


# ── WebSocket — прогресс пайплайна ──────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    sess = _get_or_create_session(session_id)
    job: JobLoop = sess["job"]

    send_q: _queue.Queue = _queue.Queue()

    def _send(text: str, kind: str):
        send_q.put(json.dumps({"type": kind, "text": text}))

    job.on_event = _send
    _send(f"Сессия {session_id[:8]}… готова", "connected")

    # Долгая загрузка большого файла может держать соединение "молчащим" много минут —
    # прокси (в т.ч. у Render) рвут такие idle-сокеты. Ресинхронизируем статус на случай,
    # если предыдущее WS-соединение уже умерло и часть прогресса была потеряна.
    if job.status == "analyzing":
        _send("🔄 Анализ уже идёт, жду обновлений…", "progress")
    elif job.status == "ready_for_review":
        _send(f"✅ Анализ завершён, {len(job.candidates)} кандидатов на клипы", "ready")
    elif job.status == "rendering":
        _send("🔄 Рендер уже идёт, жду обновлений…", "progress")
    elif job.status == "done":
        _send("✅ Рендер завершён", "render_done")
    elif job.status == "cancelled":
        _send("⏹ Рендер был отменён", "render_done")
    elif job.status == "error" and job.error:
        _send(f"❌ {job.error}", "error")

    async def _send_loop():
        last_sent = time.time()
        while True:
            try:
                sent_any = False
                while not send_q.empty():
                    await websocket.send_text(send_q.get_nowait())
                    sent_any = True
                now = time.time()
                if sent_any:
                    last_sent = now
                elif now - last_sent > 20:
                    # Keepalive: без неё прокси рвут "молчащий" сокет во время долгой
                    # загрузки большого файла, и весь дальнейший прогресс анализа теряется.
                    await websocket.send_text(json.dumps({"type": "ping", "text": ""}))
                    last_sent = now
            except Exception:
                return
            await asyncio.sleep(0.05)

    async def _recv_loop():
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    send_task = asyncio.create_task(_send_loop())
    recv_task = asyncio.create_task(_recv_loop())
    try:
        await asyncio.gather(recv_task, send_task, return_exceptions=True)
    finally:
        send_task.cancel()
        recv_task.cancel()


# ── Очистка старых сессий ───────────────────────────────────────────────────

async def _cleanup_task():
    import shutil
    while True:
        await asyncio.sleep(3600)
        cutoff = time.time() - config.SESSION_TTL_SEC
        with _sessions_lock:
            old = [(sid, s["job"]) for sid, s in _sessions.items() if s["created"] < cutoff]
            for sid, _ in old:
                del _sessions[sid]
        for sid, job in old:
            for base in (config.UPLOADS_DIR, config.WORK_DIR, config.OUTPUT_DIR):
                shutil.rmtree(base / sid, ignore_errors=True)
            if job.storage_key and config.S3_ENABLED:
                try:
                    storage.delete_object(job.storage_key)
                except storage.StorageError:
                    pass


@app.on_event("startup")
async def startup():
    asyncio.create_task(_cleanup_task())
