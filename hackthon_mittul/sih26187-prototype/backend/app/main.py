"""
SIH26187 Central Command API — Live CCTV Feed Processing.

Provides:
- Feed management (CRUD, upload, start/stop)
- MJPEG streaming of AI-processed frames
- Live analytics and events
- AI module controls
- Virtual fence management
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import json
import os
import time
import threading

import cv2
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from .services.feed_manager import FeedManager, UPLOADS_DIR, MAX_UPLOAD_SIZE_BYTES
from .services.face_recognition import (
    FaceEngine,
    FaceMatcher,
    FaceCamera,
    WatchlistService,
)
from . import database


# --- Pydantic Models ---

class ZoneUpdate(BaseModel):
    name: str
    enabled: bool = True
    polygon: List[List[int]]

class ModuleUpdate(BaseModel):
    human_detection: Optional[bool] = None
    human_tracking: Optional[bool] = None
    vehicle_detection: Optional[bool] = None
    virtual_fence: Optional[bool] = None

class CameraFeedCreate(BaseModel):
    name: str
    device_index: int = 0

class FaceCameraStartRequest(BaseModel):
    device_index: int = 0


# --- Application Setup ---

app = FastAPI(title="SIH26187 Central Command API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def ensure_cors_and_catch_exceptions(request: Request, call_next):
    if request.method == "OPTIONS":
        res = Response(status_code=200)
        res.headers["Access-Control-Allow-Origin"] = "*"
        res.headers["Access-Control-Allow-Methods"] = "*"
        res.headers["Access-Control-Allow-Headers"] = "*"
        return res
    try:
        res = await call_next(request)
        res.headers["Access-Control-Allow-Origin"] = "*"
        return res
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
            headers={"Access-Control-Allow-Origin": "*"}
        )

# Ensure directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("config", exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize Database and handle migrations
database.init_db()

# Global Feed Manager (singleton)
feed_manager = FeedManager()

# HD Face Recognition Singletons (Phase 13)
face_engine = FaceEngine()
face_matcher = FaceMatcher()
watchlist_service = WatchlistService(engine=face_engine, matcher=face_matcher)
face_camera = FaceCamera(engine=face_engine, matcher=face_matcher)

# Pre-warm FaceEngine models in background thread on backend boot to prevent enrollment HTTP timeouts
def _prewarm_face_engine():
    try:
        time.sleep(5)
        print("[Startup] Pre-warming FaceEngine AI models...")
        _ = face_engine.app
        import gc
        gc.collect()
        print("[Startup] FaceEngine AI models ready.")
    except Exception as e:
        print(f"[Startup] FaceEngine pre-warm notice: {e}")

threading.Thread(target=_prewarm_face_engine, daemon=True, name="FaceEnginePrewarmThread").start()


# ============================================================
# NEW LIVE-FEED API
# ============================================================

@app.get("/api/feeds")
def list_feeds():
    """List all registered camera feeds."""
    return {"feeds": feed_manager.list_feeds()}


@app.post("/api/feeds/upload")
@app.post("/api/feeds")
async def create_feed(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
):
    """Upload an MP4 and register as a new camera feed."""
    feed_name = name.strip() if name and name.strip() else (file.filename or "Uploaded Feed")
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Content type check: accept video/* and application/octet-stream (common for curl/browsers)
    if file.content_type and file.content_type not in ("application/octet-stream",) and "video" not in file.content_type:
        raise HTTPException(status_code=400, detail=f"Invalid content type: {file.content_type}")

    # Stream file to disk (don't load entire file into RAM)
    safe_filename = file.filename.replace(" ", "_").replace("/", "_")
    filepath = os.path.join(UPLOADS_DIR, safe_filename)

    # Handle duplicate filenames
    base, ext_part = os.path.splitext(safe_filename)
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(UPLOADS_DIR, f"{base}_{counter}{ext_part}")
        counter += 1

    total_bytes = 0
    try:
        with open(filepath, "wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    out_file.close()
                    os.remove(filepath)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum: {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB"
                    )
                out_file.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Validate with OpenCV
    try:
        feed = feed_manager.create_feed(name=feed_name, filepath=filepath, filename=safe_filename)
    except ValueError as e:
        os.remove(filepath)
        raise HTTPException(status_code=400, detail=str(e))

    return feed


@app.post("/api/feeds/camera")
def create_camera_feed(data: CameraFeedCreate):
    """Register a device camera as a new feed."""
    if not data.name.strip():
        raise HTTPException(status_code=400, detail="Feed name is required")
    try:
        feed = feed_manager.create_camera_feed(
            name=data.name.strip(),
            device_index=data.device_index,
        )
        return feed
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/feeds/{feed_id}")
def get_feed(feed_id: str):
    """Get details of a single feed."""
    feed = feed_manager.get_feed(feed_id)
    if feed is None:
        raise HTTPException(status_code=404, detail="Feed not found")
    return feed


@app.delete("/api/feeds/{feed_id}")
def delete_feed(feed_id: str):
    """Delete a feed and its uploaded file."""
    try:
        success = feed_manager.delete_feed(feed_id)
        if not success:
            raise HTTPException(status_code=404, detail="Feed not found")
        return {"status": "deleted", "feed_id": feed_id}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/feeds/{feed_id}/start")
def start_feed(feed_id: str):
    """Start live processing of a feed."""
    try:
        feed = feed_manager.start_feed(feed_id)
        return {"status": "started", "feed": feed}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/api/feeds/{feed_id}/stop")
def stop_feed(feed_id: str):
    """Stop processing a feed."""
    try:
        feed = feed_manager.stop_feed(feed_id)
        return {"status": "stopped", "feed": feed}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/feeds/{feed_id}/analytics")
def get_feed_analytics(feed_id: str):
    """Get current live analytics for an active feed."""
    feed = feed_manager.get_feed(feed_id)
    if feed is None:
        raise HTTPException(status_code=404, detail="Feed not found")

    processor = feed_manager.get_active_processor()
    if processor is None or feed_manager.get_active_feed_id() != feed_id:
        return {
            "current_persons": 0,
            "active_tracks": 0,
            "peak_persons": 0,
            "current_vehicles": 0,
            "peak_vehicles": 0,
            "cars": 0, "motorcycles": 0, "buses": 0, "trucks": 0,
            "active_intrusions": [],
            "total_intrusion_entries": 0,
            "total_intrusion_exits": 0,
            "processing_fps": 0.0,
            "frames_processed": 0,
            "max_active_tracks": 0,
            "source_fps": 0.0,
            "timestamp": 0.0,
            "status": feed.get("status", "STOPPED"),
        }

    analytics = processor.get_analytics()
    analytics["status"] = processor.status
    analytics["modules"] = processor.get_modules()
    return analytics


@app.get("/api/feeds/{feed_id}/events")
def get_feed_events(feed_id: str):
    """Get recent events from the active processing session (from memory)."""
    feed = feed_manager.get_feed(feed_id)
    if feed is None:
        raise HTTPException(status_code=404, detail="Feed not found")

    processor = feed_manager.get_active_processor()
    if processor is None or feed_manager.get_active_feed_id() != feed_id:
        return {"events": []}

    events = processor.get_events()
    return {"events": events}


@app.post("/api/feeds/{feed_id}/modules")
def update_modules(feed_id: str, modules: ModuleUpdate):
    """Toggle AI modules on/off for the active feed."""
    processor = feed_manager.get_active_processor()
    if processor is None or feed_manager.get_active_feed_id() != feed_id:
        raise HTTPException(status_code=400, detail="Feed is not currently active")

    update = {}
    if modules.human_detection is not None:
        update["human_detection"] = modules.human_detection
    if modules.human_tracking is not None:
        update["human_tracking"] = modules.human_tracking
    if modules.vehicle_detection is not None:
        update["vehicle_detection"] = modules.vehicle_detection
    if modules.virtual_fence is not None:
        update["virtual_fence"] = modules.virtual_fence

    processor.set_modules(update)
    return {"status": "updated", "modules": processor.get_modules()}


# ============================================================
# WATCHLIST API (InsightFace Powered — Phase 13)
# ============================================================

@app.get("/api/watchlist")
def list_watchlist():
    """List all watchlist records (metadata only, no embeddings)."""
    records = watchlist_service.list_records()
    return {"records": records}


@app.post("/api/watchlist")
async def enroll_watchlist(
    photo: UploadFile = File(...),
    name: str = Form(...),
    status: str = Form("WATCHLIST"),
):
    """
    Enroll a person into the persistent watchlist using InsightFace FaceEngine.
    Requires: name, status (WATCHLIST or CRITICAL), photo with exactly one face.
    """
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Name is required")

    status_val = status.strip().upper()
    if status_val not in ("WATCHLIST", "CRITICAL"):
        raise HTTPException(
            status_code=400,
            detail="Status must be WATCHLIST or CRITICAL"
        )

    # Read image bytes
    try:
        image_bytes = await photo.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded photo")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty photo file")

    try:
        record = watchlist_service.enroll(
            name=name.strip(),
            status=status_val,
            image_bytes=image_bytes,
        )
        return record
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/watchlist/{wl_id}")
def get_watchlist_record(wl_id: str):
    """Get a single watchlist record metadata."""
    record = watchlist_service.get_record(wl_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Watchlist record not found")
    return record


@app.post("/api/watchlist/{wl_id}/enable")
def enable_watchlist_record(wl_id: str):
    """Enable a watchlist record for matching."""
    if not watchlist_service.enable_record(wl_id):
        raise HTTPException(status_code=404, detail="Watchlist record not found")
    return {"status": "enabled", "id": wl_id}


@app.post("/api/watchlist/{wl_id}/disable")
def disable_watchlist_record(wl_id: str):
    """Disable a watchlist record from matching."""
    if not watchlist_service.disable_record(wl_id):
        raise HTTPException(status_code=404, detail="Watchlist record not found")
    return {"status": "disabled", "id": wl_id}


@app.delete("/api/watchlist/{wl_id}")
def delete_watchlist_record(wl_id: str):
    """Delete a watchlist record."""
    if not watchlist_service.delete_record(wl_id):
        raise HTTPException(status_code=404, detail="Watchlist record not found")
    return {"status": "deleted", "id": wl_id}


# ============================================================
# HIGH-DEFINITION FACE CAMERA API (Phase 13)
# ============================================================

@app.post("/api/face-camera/start")
def start_face_camera(payload: Optional[FaceCameraStartRequest] = None):
    """Starts the independent HD Face Camera service."""
    dev_index = payload.device_index if payload is not None else 0
    try:
        # Re-sync watchlist records before starting
        watchlist_service.sync_matcher_from_db()
        started, err_msg = face_camera.start(device_index=dev_index, allow_fallback=True)
        return {"status": "started", "device_index": dev_index}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/face-camera/stop")
def stop_face_camera():
    """Stops the independent HD Face Camera service."""
    try:
        face_camera.stop()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/face-camera/status")
def get_face_camera_status():
    """Gets current status and telemetry of the HD Face Camera."""
    return face_camera.get_status()


@app.get("/api/face-camera/results")
def get_face_camera_results():
    """Gets latest face recognition detections snapshot."""
    return face_camera.get_latest_results()


@app.post("/api/face-camera/frame")
async def push_face_camera_frame(file: UploadFile = File(...)):
    """Receives live camera JPEG frame pushed from browser client to HD Face Camera."""
    if not face_camera.is_running():
        raise HTTPException(status_code=400, detail="HD Face Camera is not currently active")
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            face_camera.push_frame(img)
            return {"status": "success"}
        raise HTTPException(status_code=400, detail="Invalid image frame")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feeds/{feed_id}/frame")
async def push_feed_frame(feed_id: str, file: UploadFile = File(...)):
    """Receives live camera JPEG frame pushed from browser client to Primary Feed."""
    processor = feed_manager.get_active_processor()
    if processor is None or feed_manager.get_active_feed_id() != feed_id:
        raise HTTPException(status_code=400, detail="Feed is not currently active")
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            if hasattr(processor.video_source, "push_frame"):
                processor.video_source.push_frame(img)
            return {"status": "success"}
        raise HTTPException(status_code=400, detail="Invalid image frame")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/face-camera/stream")
def stream_face_camera():
    """
    MJPEG stream of HD Face Camera live frames with face overlays.
    Browser displays via: <img src="/api/face-camera/stream" />
    """
    if not face_camera.is_running():
        raise HTTPException(status_code=400, detail="HD Face Camera is not currently active")

    return StreamingResponse(
        face_camera.generate_mjpeg_stream(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.get("/api/stream/{feed_id}")
def stream_feed(feed_id: str):
    """
    MJPEG stream of AI-processed frames.
    Browser displays via: <img src="/api/stream/{feed_id}" />
    """
    feed = feed_manager.get_feed(feed_id)
    if feed is None:
        raise HTTPException(status_code=404, detail="Feed not found")

    processor = feed_manager.get_active_processor()
    if processor is None or feed_manager.get_active_feed_id() != feed_id:
        raise HTTPException(status_code=400, detail="Feed is not currently active/live")

    def generate_frames():
        target_fps = min(feed.get("fps", 30), 30)
        frame_interval = 1.0 / target_fps if target_fps > 0 else 1.0 / 30.0

        while True:
            # Check if processor is still the active one and still running
            current_proc = feed_manager.get_active_processor()
            if current_proc is None or current_proc is not processor:
                break
            if processor.status not in ("LIVE", "STARTING"):
                break

            frame = processor.get_latest_frame()
            if frame is not None:
                _, jpeg = cv2.imencode(
                    '.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + jpeg.tobytes()
                    + b'\r\n'
                )

            time.sleep(frame_interval)

    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


# ============================================================
# ZONE / FENCE MANAGEMENT (Enhanced)
# ============================================================

@app.get("/api/zones")
def get_zones():
    try:
        zones = database.get_all_zones()
        return {"zones": zones}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/zones")
def update_zones(zone_data: ZoneUpdate):
    if len(zone_data.polygon) < 3:
        raise HTTPException(status_code=400, detail="Polygon must have at least 3 points")

    try:
        # Currently we only support one active zone in this prototype 'restricted-border-zone'
        database.upsert_zone(
            zone_id="restricted-border-zone",
            name=zone_data.name,
            enabled=zone_data.enabled,
            polygon=zone_data.polygon
        )

        # Hot-reload fence in active processor
        feed_manager.reload_fence()

        return {
            "status": "success",
            "message": "Fence saved. Applied to live processing immediately."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# GENERAL ENDPOINTS
# ============================================================

# In-memory store for session acknowledgements
session_acknowledgements = set()


@app.get("/")
def read_root():
    return {"message": "SIH26187 Backend API is running"}


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/events/{event_id}/acknowledge")
def acknowledge_event(event_id: str):
    if not database.acknowledge_event(event_id):
        raise HTTPException(status_code=404, detail="Event not found")
    return {"status": "success", "event_id": event_id, "acknowledged": True}


@app.get("/api/events")
def get_all_events():
    events = database.get_events()
    return {"events": events}


@app.get("/api/events/acknowledgements")
def get_acknowledgements():
    return {"acknowledged_events": database.get_acknowledged_events()}

