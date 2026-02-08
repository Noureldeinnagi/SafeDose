from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from backend_api.ids_engine import IDSEngine
from backend_api.db import SessionLocal, engine, Base
from backend_api import crud
from backend_api import models  # IMPORTANT: ensures Alert model is registered

from datetime import datetime, timezone
import time
import uuid


app = FastAPI()

# ----------------------------
# Frontend mount
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "Grad" / "Demoo" / "Demoo"
app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="app")


# ----------------------------
# IDS Engine (OLD SIM settings)
# ----------------------------
engine_ids = IDSEngine(
    window_size=20,
    stride=1,
    alert_threshold=0.50,
    sleep_sec=0.15,
    loop=False
)

LAST_EVENTS = []
MAX_EVENTS = 500
sim_task: asyncio.Task | None = None


class WSManager:
    def __init__(self):
        self.clients = set()
        self.lock = asyncio.Lock()

    async def add(self, ws: WebSocket):
        await ws.accept()
        async with self.lock:
            self.clients.add(ws)

    async def remove(self, ws: WebSocket):
        async with self.lock:
            self.clients.discard(ws)

    async def broadcast(self, data: dict):
        dead = []
        async with self.lock:
            for ws in self.clients:
                try:
                    await ws.send_json(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


manager = WSManager()


def ensure_sim_running():
    global sim_task
    if sim_task is None or sim_task.done():
        sim_task = asyncio.create_task(engine_ids.run(publish))
        print("✅ IDS simulation started.")


import time
from datetime import datetime, timezone
import uuid

async def publish(event: dict):
    # ✅ Always attach timing fields (even for status/summary)
    event.setdefault("event_id", uuid.uuid4().hex)
    event["server_ts_ms"] = int(time.time() * 1000)
    event.setdefault("detect_ts_ms", event["server_ts_ms"])
    event["server_ts_iso"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    # store in memory
    LAST_EVENTS.append(event)
    if len(LAST_EVENTS) > MAX_EVENTS:
        del LAST_EVENTS[: len(LAST_EVENTS) - MAX_EVENTS]

    # store alerts in DB
    if event.get("type") == "event" and event.get("is_alert") is True:
        def _write():
            db = SessionLocal()
            try:
                crud.create_alert_from_event(db, event)
            finally:
                db.close()

        await asyncio.to_thread(_write)

    # broadcast live
    await manager.broadcast(event)

@app.get("/api/time_test")
def time_test():
    return {"server_ts_ms": int(time.time() * 1000)}


@app.on_event("startup")
async def startup():
    # 1) Create tables
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ DB tables ready.")
    except Exception as e:
        print("❌ DB not ready:", e)

    # 2) Start simulation
    try:
        ensure_sim_running()
    except Exception as e:
        print("❌ Simulation failed to start:", e)


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def home():
    return RedirectResponse("/app/Dashboard.html")


@app.get("/api/health")
def api_health():
    return {"ok": True, "sim_running": sim_task is not None and not sim_task.done()}


@app.get("/api/events")
def api_events():
    return JSONResponse(LAST_EVENTS[-200:])


@app.get("/api/db_test")
def db_test():
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return {"db": "ok"}
    finally:
        db.close()


# ----------------------------
# Alerts API (DB)
# ----------------------------
@app.get("/api/alerts")
def api_alerts(
    search: str | None = None,
    severity: str | None = None,
    resolved: str | None = "false",
    limit: int = 200,
):
    db = SessionLocal()
    try:
        rows = crud.list_alerts(db, search=search, severity=severity, resolved=resolved, limit=limit)
        return [
            {
                "id": a.id,
                "message": a.message,
                "severity": a.severity,
                "timestamp": a.created_at.isoformat(timespec="seconds"),
                "resolved": a.resolved,
            }
            for a in rows
        ]
    finally:
        db.close()


@app.post("/api/alerts/{alert_id}/resolve")
def api_resolve(alert_id: int):
    db = SessionLocal()
    try:
        a = crud.resolve_alert(db, alert_id)
        if not a:
            return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
        return {"ok": True, "id": a.id, "resolved": a.resolved}
    finally:
        db.close()


# ----------------------------
# WebSocket
# ----------------------------
@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await manager.add(ws)
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await manager.remove(ws)

