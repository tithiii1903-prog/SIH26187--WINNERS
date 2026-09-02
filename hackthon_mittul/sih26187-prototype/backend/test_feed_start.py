from app.services.feed_manager import FeedManager
try:
    fm = FeedManager()
    fm.create_feed("test", "dummy.mp4", "dummy.mp4")
    feeds = fm.list_feeds()
    feed_id = feeds[-1]["id"]
    print(f"Starting feed {feed_id}...")
    fm.start_feed(feed_id)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
