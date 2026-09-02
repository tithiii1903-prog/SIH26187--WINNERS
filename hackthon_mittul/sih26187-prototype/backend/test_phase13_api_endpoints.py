"""
Phase 13 API Endpoints Verification Test.

Tests FastAPI HTTP endpoints:
- GET /api/watchlist
- POST /api/watchlist
- GET /api/watchlist/{id}
- POST /api/watchlist/{id}/enable
- POST /api/watchlist/{id}/disable
- DELETE /api/watchlist/{id}
- POST /api/face-camera/start
- POST /api/face-camera/stop
- GET /api/face-camera/status
- GET /api/face-camera/results
- GET /api/events
"""

import os
import sys
import time
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app

client = TestClient(app)

def test_api():
    print("\n--- Testing Phase 13 API Endpoints ---")

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    print("[PASS] GET /api/health")

    # 2. Watchlist List
    res = client.get("/api/watchlist")
    assert res.status_code == 200
    records = res.json().get("records", [])
    print(f"[PASS] GET /api/watchlist -> {len(records)} records")

    # 3. Watchlist Enrollment (Single Face)
    photo_path = "test_data/person_0.jpg"
    with open(photo_path, "rb") as f:
        photo_bytes = f.read()

    res = client.post(
        "/api/watchlist",
        data={"name": "API Test Person", "status": "CRITICAL"},
        files={"photo": ("person.jpg", photo_bytes, "image/jpeg")}
    )
    assert res.status_code == 200, f"Enrollment failed: {res.text}"
    enrolled = res.json()
    wl_id = enrolled["id"]
    assert enrolled["name"] == "API Test Person"
    assert enrolled["status"] == "CRITICAL"
    # Ensure private fields (embeddings, image paths) are NEVER exposed
    assert "embedding" not in enrolled
    assert "reference_image_path" not in enrolled
    assert "embedding_path" not in enrolled
    print(f"[PASS] POST /api/watchlist -> enrolled ID={wl_id}, Name={enrolled['name']}")

    # 4. Get Watchlist Record
    res = client.get(f"/api/watchlist/{wl_id}")
    assert res.status_code == 200
    rec = res.json()
    assert rec["id"] == wl_id
    assert "embedding" not in rec
    print(f"[PASS] GET /api/watchlist/{wl_id}")

    # 5. Disable Record
    res = client.post(f"/api/watchlist/{wl_id}/disable")
    assert res.status_code == 200
    assert res.json()["status"] == "disabled"
    print(f"[PASS] POST /api/watchlist/{wl_id}/disable")

    # 6. Enable Record
    res = client.post(f"/api/watchlist/{wl_id}/enable")
    assert res.status_code == 200
    assert res.json()["status"] == "enabled"
    print(f"[PASS] POST /api/watchlist/{wl_id}/enable")

    # 7. HD Face Camera Start with invalid device (truthful error check)
    res_bad = client.post("/api/face-camera/start", json={"device_index": 999})
    assert res_bad.status_code == 400
    assert "Unable to open camera device 999" in res_bad.json()["detail"]
    print("[PASS] POST /api/face-camera/start with invalid device -> 400 'Unable to open camera device 999'")

    # 8. HD Face Camera Start with valid device 0
    res = client.post("/api/face-camera/start", json={"device_index": 0})
    assert res.status_code == 200
    assert res.json()["status"] == "started"
    print("[PASS] POST /api/face-camera/start with device 0 -> 200 'started'")

    # 9. HD Face Camera Status
    res = client.get("/api/face-camera/status")
    assert res.status_code == 200
    st = res.json()
    assert st["is_running"] is True
    assert st["registered_faces_count"] >= 1
    print(f"[PASS] GET /api/face-camera/status -> is_running={st['is_running']}, enrolled={st['registered_faces_count']}")

    # 10. HD Face Camera Results
    res = client.get("/api/face-camera/results")
    assert res.status_code == 200
    results_data = res.json()
    assert "faces" in results_data
    assert "fps" in results_data
    print(f"[PASS] GET /api/face-camera/results -> faces count={len(results_data['faces'])}")

    # 11. HD Face Camera Stop
    res = client.post("/api/face-camera/stop")
    assert res.status_code == 200
    assert res.json()["status"] == "stopped"
    print("[PASS] POST /api/face-camera/stop")

    # 12. Delete Watchlist Record
    res = client.delete(f"/api/watchlist/{wl_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "deleted"
    print(f"[PASS] DELETE /api/watchlist/{wl_id}")

    # 12. Check Events
    res = client.get("/api/events")
    assert res.status_code == 200
    events = res.json().get("events", [])
    print(f"[PASS] GET /api/events -> {len(events)} events recorded")

    print("\n--- ALL PHASE 13 API ENDPOINT TESTS PASSED! ---")

if __name__ == "__main__":
    test_api()
