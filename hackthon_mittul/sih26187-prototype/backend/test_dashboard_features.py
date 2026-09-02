import requests
import time

BASE_URL = "http://localhost:8000"

def get_feeds():
    return requests.get(f"{BASE_URL}/api/feeds").json().get("feeds", [])

def test_feature():
    print("Getting feeds...")
    feeds = get_feeds()
    bg_video = None
    for f in feeds:
        if "walking" in f["name"].lower():
            bg_video = f
            break
    
    if not bg_video:
        print("Walking video not found, trying to find road sample...")
        for f in feeds:
            if "road" in f["name"].lower():
                bg_video = f
                break
    
    if not bg_video:
        print("No suitable feed found. Creating one...")
        r = requests.post(f"{BASE_URL}/api/feeds", data={"name": "walking"}, files={"file": ("bg.mp4", open("uploads/background_video___people___walking__.mp4", "rb"), "video/mp4")})
        print("Upload:", r.json())
        bg_video = r.json()


    feed_id = bg_video["id"]
    print(f"Selected Feed: {bg_video['name']} ({feed_id})")

    print("\nSetting virtual fence polygon covering the screen...")
    zone_payload = {
        "name": "Full Screen Zone",
        "enabled": True,
        "polygon": [[0, 0], [1280, 0], [1280, 720], [0, 720]]
    }
    r = requests.post(f"{BASE_URL}/api/zones", json=zone_payload)
    print("Set Zone:", r.json())

    print("\nStarting feed...")
    r = requests.post(f"{BASE_URL}/api/feeds/{feed_id}/start")
    print("Start:", r.json())

    print("\nEnabling all modules...")
    modules = {
        "human_detection": True,
        "human_tracking": True,
        "vehicle_detection": True,
        "virtual_fence": True
    }
    r = requests.post(f"{BASE_URL}/api/feeds/{feed_id}/modules", json=modules)
    print("Modules:", r.json())

    print("\nWaiting for 5 seconds for processing...")
    time.sleep(5)

    print("\nChecking analytics...")
    r = requests.get(f"{BASE_URL}/api/feeds/{feed_id}/analytics")
    analytics = r.json()
    print("Current Persons:", analytics.get("current_persons"))
    print("Active Tracks:", analytics.get("active_tracks"))
    print("Active Intrusions:", analytics.get("active_intrusions"))

    print("\nChecking events...")
    r = requests.get(f"{BASE_URL}/api/feeds/{feed_id}/events")
    events = r.json().get("events", [])
    
    intrusion_enter = False
    for e in events:
        if e["type"] == "INTRUSION_ENTER":
            intrusion_enter = True
            print("Found INTRUSION_ENTER:", e)
            break
    
    if intrusion_enter:
        print("\nINTRUSION VERIFICATION PASS")
    else:
        print("\nINTRUSION VERIFICATION FAIL")
    
    # Wait another 2 seconds and check if any exit occurred
    print("\nUpdating zone to empty to force INTRUSION_EXIT...")
    zone_payload = {
        "name": "Empty Zone",
        "enabled": True,
        "polygon": [[-10, -10], [-5, -10], [-5, -5]] # Offscreen
    }
    requests.post(f"{BASE_URL}/api/zones", json=zone_payload)
    time.sleep(2)
    
    print("\nChecking events for EXIT...")
    r = requests.get(f"{BASE_URL}/api/feeds/{feed_id}/events")
    events = r.json().get("events", [])
    intrusion_exit = False
    for e in events:
        if e["type"] == "INTRUSION_EXIT":
            intrusion_exit = True
            print("Found INTRUSION_EXIT:", e)
            break
    
    if intrusion_exit:
        print("INTRUSION EXIT VERIFICATION PASS")
    else:
        print("INTRUSION EXIT VERIFICATION FAIL")

    print("\nStopping feed...")
    r = requests.post(f"{BASE_URL}/api/feeds/{feed_id}/stop")
    print("Stop:", r.json())


if __name__ == "__main__":
    test_feature()
