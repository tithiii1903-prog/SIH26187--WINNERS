from app.services.feed_manager import FeedManager
import time

try:
    fm = FeedManager()
    
    print("=== TEST A: MP4 Feed ===")
    fm.create_feed("test_mp4", "dummy.mp4", "dummy.mp4")
    feeds = fm.list_feeds()
    mp4_id = feeds[-1]["id"]
    print(f"Starting MP4 feed {mp4_id}...")
    fm.start_feed(mp4_id)
    time.sleep(2)
    proc = fm.get_active_processor()
    print(f"Processor status: {proc.status if proc else 'None'}")
    
    print("=== TEST C: STOP ===")
    fm.stop_feed(mp4_id)
    time.sleep(1)
    proc = fm.get_active_processor()
    print(f"Processor status after stop: {proc.status if proc else 'None'}")
    
    print("=== TEST B: Camera Feed ===")
    fm.create_camera_feed("test_camera", 0)
    feeds = fm.list_feeds()
    cam_id = feeds[-1]["id"]
    print(f"Starting Camera feed {cam_id}...")
    fm.start_feed(cam_id)
    time.sleep(2)
    proc = fm.get_active_processor()
    print(f"Processor status: {proc.status if proc else 'None'}")
    
    print("=== TEST D: MP4 -> STOP -> Camera ===")
    # Actually just stopping and starting another
    fm.stop_feed(cam_id)
    time.sleep(1)
    fm.start_feed(mp4_id)
    time.sleep(1)
    proc = fm.get_active_processor()
    print(f"Processor status (MP4): {proc.status if proc else 'None'}")
    
    print("=== TEST E: Camera -> STOP -> MP4 ===")
    # It's currently running MP4, so let's do MP4 -> STOP -> Camera again
    fm.stop_feed(mp4_id)
    time.sleep(1)
    fm.start_feed(cam_id)
    time.sleep(1)
    proc = fm.get_active_processor()
    print(f"Processor status (Camera): {proc.status if proc else 'None'}")
    fm.stop_feed(cam_id)
    
except Exception as e:
    import traceback
    traceback.print_exc()
