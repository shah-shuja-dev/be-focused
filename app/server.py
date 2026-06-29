"""
server.py -- FastAPI application, Prometheus metrics, web UI.

Endpoints:
    GET /           Web UI
    GET /violations All violations JSON
    GET /today      Today violations JSON
    GET /status     Scheduler + detector state
    GET /metrics    Prometheus scrape

Run:
    uvicorn app.server:app --host 0.0.0.0 --port 8000
"""

import os, time, threading
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from prometheus_client import (
    Counter, Gauge, Histogram,
    generate_latest, CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

load_dotenv(r"config\.env")

# ── Prometheus metrics ─────────────────────────────────────────────────────────
VIOLATIONS_TOTAL  = Counter(  "focusguard_violations_total",
                               "Total focus violations detected")
VIOLATIONS_TODAY  = Gauge(    "focusguard_violations_today",
                               "Violations today")
CONFIDENCE_GAUGE  = Gauge(    "focusguard_model_confidence",
                               "Latest model confidence score")
INFERENCE_LATENCY = Histogram("focusguard_inference_latency_seconds",
                               "Per-frame inference latency",
                               buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
SCHEDULER_ACTIVE  = Gauge(    "focusguard_scheduler_active",
                               "1 if inside a check window, 0 if sleeping")
AUDIO_TRIGGERS    = Counter(  "focusguard_audio_triggers_total",
                               "Times audio was triggered")
UNFOCUSED_STREAK  = Gauge(    "focusguard_unfocused_streak_seconds",
                               "Current unfocused streak in seconds")

# ── Shared state ───────────────────────────────────────────────────────────────
_state = {
    "detector":   None,
    "scheduler":  None,
    "started_at": datetime.now().isoformat(),
}

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.inference_engine   import load_model, InferenceSession
    from app.violation_detector import ViolationDetector
    from app.audio_pipeline     import AudioPipeline
    from app.scheduler          import Scheduler

    audio = AudioPipeline()

    def on_violation(v):
        VIOLATIONS_TOTAL.inc()
        AUDIO_TRIGGERS.inc()
        audio.on_violation(v)

    detector = ViolationDetector(on_violation=on_violation)
    _state["detector"] = detector

    model = load_model()

    def check_window():
        SCHEDULER_ACTIVE.set(1)
        try:
            duration = float(os.getenv("CHECK_WINDOW_DURATION", "60"))
            with InferenceSession(model, duration=duration) as session:
                for label, conf, ts in session:
                    t0 = time.time()
                    detector.update(label, conf, ts)
                    CONFIDENCE_GAUGE.set(conf)
                    INFERENCE_LATENCY.observe(time.time() - t0)
                    UNFOCUSED_STREAK.set(detector.unfocused_duration)
                    VIOLATIONS_TODAY.set(len(detector.get_today_violations()))
        finally:
            SCHEDULER_ACTIVE.set(0)

    audio.start_reminder_loop(lambda: detector.is_unfocused)

    scheduler = Scheduler(check_window)
    _state["scheduler"] = scheduler
    threading.Thread(target=scheduler.run, daemon=True).start()

    yield

    scheduler.stop()


app = FastAPI(title="FocusGuard", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    detector  = _state["detector"]
    scheduler = _state["scheduler"]
    return templates.TemplateResponse("index.html", {
        "request":          request,
        "violations":       detector.get_today_violations() if detector else [],
        "total_today":      len(detector.get_today_violations()) if detector else 0,
        "started_at":       _state["started_at"],
        "scheduler_status": scheduler.status if scheduler else {},
    })


@app.get("/violations")
async def get_violations():
    d = _state["detector"]
    return JSONResponse({"violations": d.get_all_violations() if d else []})


@app.get("/today")
async def get_today():
    d = _state["detector"]
    return JSONResponse({"violations": d.get_today_violations() if d else []})


@app.get("/status")
async def get_status():
    d = _state["detector"]
    s = _state["scheduler"]
    return JSONResponse({
        "scheduler":        s.status if s else {},
        "violations_today": len(d.get_today_violations()) if d else 0,
        "unfocused_streak": d.unfocused_duration if d else 0,
        "started_at":       _state["started_at"],
    })


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)