"""Local web app: submission form, monitoring dashboard, control API.

The UI is a monitor, not a driver. The worker runs in the background whether
or not a browser is open; closing the tab has no effect on the batch.
"""

from __future__ import annotations

import contextlib
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config as config_module
from . import db
from .config import ROOT, load as load_config
from .logging_setup import get
from .worker import get_worker

log = get("server")

STATIC_DIR = ROOT / "static"

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


class SubmitPayload(BaseModel):
    urls: str = Field(default="", description="Newline/whitespace separated links")


class ConfigPayload(BaseModel):
    changes: dict[str, Any] = Field(default_factory=dict)


def parse_urls(blob: str) -> list[str]:
    """Extract links from pasted text; tolerates commas, quotes and bullets."""
    found: list[str] = []
    seen: set[str] = set()
    for line in (blob or "").splitlines():
        for match in URL_RE.findall(line):
            url = match.strip().strip(",;\"'<>()[]")
            if url and url not in seen:
                seen.add(url)
                found.append(url)
    return found


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    cfg = load_config(force=True)
    worker = get_worker()
    if bool(cfg.get("server.autostart_worker", True)):
        worker.start()
    try:
        yield
    finally:
        worker.stop(timeout=20.0)


app = FastAPI(title="sang_det", version="1.0.0", lifespan=lifespan)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ----------------------------------------------------------------- pages


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>sang_det</h1><p>static/index.html is missing.</p>", 500)
    return HTMLResponse(page.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- API


@app.post("/api/videos")
async def submit_videos(payload: SubmitPayload) -> JSONResponse:
    urls = parse_urls(payload.urls)
    if not urls:
        raise HTTPException(400, "No valid http(s) links found in the input.")

    added, skipped = db.add_videos(urls)
    db.log_event("info", f"Queued {added} link(s), skipped {skipped} duplicate(s)")

    worker = get_worker()
    if not worker.running:
        worker.start()

    return JSONResponse({"added": added, "skipped": skipped, "total_submitted": len(urls)})


@app.get("/api/status")
async def status() -> dict:
    cfg = load_config()
    worker = get_worker()
    videos = db.list_videos()
    totals = db.totals()

    now = time.time()
    for video in videos:
        started = video.get("started_at")
        finished = video.get("finished_at")
        video["elapsed_s"] = round((finished or now) - started, 1) if started else None
        duration = video.get("duration_s") or 0
        position = video.get("position_s") or 0
        video["progress"] = round(min(1.0, position / duration), 4) if duration > 0 else None

    return {
        "worker": worker.status(),
        "totals": totals,
        "videos": videos,
        "config": cfg.as_dict(),
        "editable_keys": sorted(config_module.EDITABLE_KEYS),
        "server_time": now,
    }


@app.get("/api/events")
async def events(limit: int = 60) -> dict:
    return {"events": db.recent_events(max(1, min(500, limit)))}


@app.get("/api/plates/recent")
async def plates_recent(limit: int = 24) -> dict:
    return {"plates": db.recent_plates(max(1, min(200, limit)))}


@app.get("/api/plates/file/{filename}")
async def plate_file(filename: str):
    """Serve one saved crop for the dashboard preview strip."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    output_dir = load_config().path("output.dir").resolve()
    path = (output_dir / filename).resolve()
    if not str(path).startswith(str(output_dir)) or not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/control/{action}")
async def control(action: str) -> dict:
    worker = get_worker()
    action = action.lower()

    if action == "pause":
        worker.pause()
    elif action == "resume":
        if not worker.running:
            worker.start()
        worker.resume()
    elif action == "start":
        worker.start()
    elif action == "retry-errors":
        count = db.requeue_all_errored()
        if not worker.running:
            worker.start()
        return {"ok": True, "requeued": count}
    else:
        raise HTTPException(400, f"Unknown action: {action}")

    return {"ok": True, "state": worker.status()["state"]}


@app.post("/api/videos/{video_id}/{action}")
async def video_action(video_id: int, action: str) -> dict:
    video = db.get_video(video_id)
    if video is None:
        raise HTTPException(404, "Video not found")

    action = action.lower()
    if action == "retry":
        db.requeue_video(video_id, reset_progress=False)
    elif action == "restart":
        db.requeue_video(video_id, reset_progress=True)
    elif action == "cancel":
        db.cancel_video(video_id)
    elif action == "delete":
        db.delete_video(video_id)
    else:
        raise HTTPException(400, f"Unknown action: {action}")

    worker = get_worker()
    if action in ("retry", "restart") and not worker.running:
        worker.start()
    return {"ok": True}


@app.get("/api/config")
async def get_config() -> dict:
    return {
        "config": load_config().as_dict(),
        "editable_keys": sorted(config_module.EDITABLE_KEYS),
        "path": str(config_module.CONFIG_PATH),
    }


@app.put("/api/config")
async def put_config(payload: ConfigPayload) -> dict:
    if not payload.changes:
        raise HTTPException(400, "No changes supplied")
    updated = config_module.update(payload.changes)
    db.log_event("info", f"Config updated: {sorted(payload.changes)}")
    return {"ok": True, "config": updated.as_dict()}


@app.get("/api/health")
async def health() -> dict:
    worker = get_worker()
    return {
        "ok": True,
        "worker_running": worker.running,
        "version": app.version,
        "output_dir": str(load_config().path("output.dir")),
    }


def run() -> None:
    """Entry point used by run.py serve."""
    import uvicorn

    cfg = load_config(force=True)
    host = str(cfg.get("server.host", "127.0.0.1"))
    port = int(cfg.get("server.port", 8000))
    log.info("Dashboard: http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


__all__ = ["app", "run", "parse_urls", "Path"]
