import time
import cv2
import requests
import threading
from fastapi.testclient import TestClient
from app.main import app, feed_manager

client = TestClient(app)

def test_api_endpoints():
    print("Testing FastAPI live endpoints...")

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200 and res.json() == {"status": "ok"}
    print("-> /api/health: OK")

    # 2. Watchlist
    res = client.get("/api/watchlist")
    assert res.status_code == 200
    print(f"-> /api/watchlist: OK ({len(res.json()['records'])} records)")

    # 3. Create & start feed
    res = client.get("/api/feeds")
    assert res.status_code == 200
    feeds = res.json()["feeds"]
    print(f"-> /api/feeds: OK ({len(feeds)} feeds)")

    # Create camera feed
    res = client.post("/api/feeds/camera", json={"name": "TestCam", "device_index": 0})
    if res.status_code == 200:
        cam_feed = res.json()
        cam_id = cam_feed["id"]
        print(f"-> Created test camera feed {cam_id}")

        # Start feed
        res = client.post(f"/api/feeds/{cam_id}/start")
        assert res.status_code == 200
        print(f"-> Started feed {cam_id}: {res.json()['status']}")

        # Read analytics
        time.sleep(1.5)
        res = client.get(f"/api/feeds/{cam_id}/analytics")
        assert res.status_code == 200
        an = res.json()
        print(f"-> Analytics: AI FPS={an.get('processing_fps')}, Stream FPS={an.get('stream_fps')}, Source FPS={an.get('source_fps')}")

        # Toggle modules
        res = client.post(f"/api/feeds/{cam_id}/modules", json={"human_detection": True, "face_watchlist": False})
        assert res.status_code == 200
        print(f"-> Toggle modules: {res.json()['modules']}")

        # Stop feed
        res = client.post(f"/api/feeds/{cam_id}/stop")
        assert res.status_code == 200
        print(f"-> Stopped feed {cam_id}")

        # Delete feed
        res = client.delete(f"/api/feeds/{cam_id}")
        assert res.status_code == 200
        print(f"-> Deleted feed {cam_id}")

    print("ALL API ENDPOINT TESTS PASSED!")

if __name__ == "__main__":
    test_api_endpoints()
