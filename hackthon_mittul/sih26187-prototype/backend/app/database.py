import sqlite3
import json
import os
import time
import shutil
import uuid
from typing import List, Dict, Any, Optional

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "command_center.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs(os.path.join(DB_DIR, "watchlist_photos"), exist_ok=True)
    os.makedirs(os.path.join(DB_DIR, "watchlist_embeddings"), exist_ok=True)
    conn = get_db()
    cursor = conn.cursor()

    # Create Watchlist table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT 1,
            reference_image_path TEXT,
            embedding_path TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    ''')

    # Create Zones table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS zones (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT 1,
            polygon TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    ''')

    # Create Events table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            timestamp REAL NOT NULL,
            feed_id TEXT,
            track_id INTEGER,
            watchlist_id TEXT,
            zone_id TEXT,
            message TEXT NOT NULL,
            acknowledged BOOLEAN NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            source TEXT DEFAULT 'MAIN_CCTV',
            similarity REAL DEFAULT NULL
        )
    ''')

    # Create Migration marker table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS _migrations (
            name TEXT PRIMARY KEY,
            migrated_at REAL NOT NULL
        )
    ''')
    
    conn.commit()

    # Safe column migration for events if upgrading from older schema
    cursor.execute("PRAGMA table_info(events)")
    existing_cols = {row["name"] for row in cursor.fetchall()}
    if "source" not in existing_cols:
        cursor.execute("ALTER TABLE events ADD COLUMN source TEXT DEFAULT 'MAIN_CCTV'")
    if "similarity" not in existing_cols:
        cursor.execute("ALTER TABLE events ADD COLUMN similarity REAL DEFAULT NULL")
    conn.commit()

    # Check and perform migration
    cursor.execute("SELECT 1 FROM _migrations WHERE name = 'initial_json_migration'")
    if not cursor.fetchone():
        _migrate_json_to_db(conn)
        cursor.execute("INSERT INTO _migrations (name, migrated_at) VALUES ('initial_json_migration', ?)", (time.time(),))
        conn.commit()
        
    conn.close()


def _migrate_json_to_db(conn: sqlite3.Connection):
    print("[Database] Starting JSON to SQLite migration...")
    cursor = conn.cursor()
    now = time.time()

    # Migrate Watchlist
    watchlist_path = "config/watchlist.json"
    if os.path.exists(watchlist_path):
        try:
            with open(watchlist_path, "r") as f:
                data = json.load(f)
                records = data.get("records", {})
                for wl_id, rec in records.items():
                    cursor.execute('''
                        INSERT OR IGNORE INTO watchlist (id, name, status, enabled, reference_image_path, embedding_path, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        wl_id,
                        rec.get("name", "Unknown"),
                        rec.get("status", "WATCHLIST"),
                        rec.get("enabled", True),
                        f"config/watchlist_data/{wl_id}.jpg",
                        f"config/watchlist_data/{wl_id}.pt",
                        rec.get("created", now),
                        now
                    ))
            shutil.copy(watchlist_path, "config/backup_watchlist.json")
            print(f"[Database] Migrated {len(records)} watchlist records.")
        except Exception as e:
            print(f"[Database] Error migrating watchlist: {e}")

    # Migrate Zones
    zones_path = "config/zones.json"
    if os.path.exists(zones_path):
        try:
            with open(zones_path, "r") as f:
                data = json.load(f)
                zones = data.get("zones", [])
                for zone in zones:
                    cursor.execute('''
                        INSERT OR IGNORE INTO zones (id, name, enabled, polygon, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        zone.get("id"),
                        zone.get("name", "Zone"),
                        zone.get("enabled", True),
                        json.dumps(zone.get("polygon", [])),
                        now,
                        now
                    ))
            shutil.copy(zones_path, "config/backup_zones.json")
            print(f"[Database] Migrated {len(zones)} zones.")
        except Exception as e:
            print(f"[Database] Error migrating zones: {e}")

    conn.commit()
    print("[Database] Migration complete.")

# ----------------- ASYNC EVENT WRITER -----------------
import queue
import threading

_event_queue: queue.Queue = queue.Queue()
_writer_stop_event = threading.Event()
_writer_thread: Optional[threading.Thread] = None
_writer_lock = threading.Lock()


def _event_writer_loop():
    """Background worker thread that batches and persists events to SQLite."""
    while not _writer_stop_event.is_set():
        try:
            item = _event_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        batch = [item]
        # Drain any other pending events in queue for batch insertion
        while len(batch) < 50:
            try:
                batch.append(_event_queue.get_nowait())
            except queue.Empty:
                break

        if batch:
            try:
                conn = get_db()
                cursor = conn.cursor()
                cursor.executemany('''
                    INSERT INTO events (id, event_type, timestamp, feed_id, track_id, watchlist_id, zone_id, message, created_at, source, similarity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch)
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[Database Async Writer] Error writing events batch: {e}")
            finally:
                for _ in batch:
                    _event_queue.task_done()


def _ensure_writer_started():
    global _writer_thread
    with _writer_lock:
        if _writer_thread is None or not _writer_thread.is_alive():
            _writer_stop_event.clear()
            _writer_thread = threading.Thread(target=_event_writer_loop, daemon=True, name="DBEventWriter")
            _writer_thread.start()


def flush_events(timeout: float = 1.0):
    """Wait for all pending events in queue to be written to disk."""
    if _writer_thread is not None and _writer_thread.is_alive():
        try:
            _event_queue.join()
        except Exception:
            pass


# ----------------- EVENTS API -----------------

def insert_event(
    event_type: str,
    timestamp: float,
    message: str,
    feed_id: Optional[str] = None,
    track_id: Optional[int] = None,
    watchlist_id: Optional[str] = None,
    zone_id: Optional[str] = None,
    source: str = "MAIN_CCTV",
    similarity: Optional[float] = None,
    sync: bool = False
) -> str:
    """
    Inserts an event into SQLite.
    By default, uses the non-blocking async queue for microsecond return.
    If sync=True, executes immediately.
    """
    event_id = str(uuid.uuid4())
    record = (
        event_id,
        event_type,
        timestamp,
        feed_id,
        track_id,
        watchlist_id,
        zone_id,
        message,
        time.time(),
        source,
        similarity
    )

    if sync:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (id, event_type, timestamp, feed_id, track_id, watchlist_id, zone_id, message, created_at, source, similarity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', record)
        conn.commit()
        conn.close()
    else:
        _ensure_writer_started()
        _event_queue.put(record)

    return event_id

def insert_face_event(
    event_type: str,
    timestamp: float,
    watchlist_id: Optional[str],
    name: str,
    status: str,
    similarity: float,
    source: str = "HD Face Camera"
) -> str:
    """Inserts a dedicated face recognition event into SQLite events table."""
    if event_type == "FACE_WATCHLIST_MATCH":
        msg = f"HD Face Camera: Matched {name} ({status}) with {similarity * 100:.1f}% similarity"
    elif event_type == "FACE_WATCHLIST_MATCH_CLEARED":
        msg = f"HD Face Camera: Match cleared for {name} ({status})"
    else:
        msg = f"HD Face Camera: {event_type} - {name} ({status})"

    return insert_event(
        event_type=event_type,
        timestamp=timestamp,
        message=msg,
        feed_id="hd_face_camera",
        watchlist_id=watchlist_id,
        source=source,
        similarity=round(float(similarity), 4)
    )

def get_events(limit: int = 100) -> List[Dict[str, Any]]:
    flush_events(timeout=0.1)
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    events = []
    for row in rows:
        item = dict(row)
        item["acknowledged"] = bool(item.get("acknowledged", 0))
        events.append(item)
    return events

def acknowledge_event(event_id: str) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE events SET acknowledged = 1 WHERE id = ?', (event_id,))
    changes = cursor.rowcount
    conn.commit()
    conn.close()
    return changes > 0

def get_acknowledged_events() -> List[str]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM events WHERE acknowledged = 1')
    rows = cursor.fetchall()
    conn.close()
    return [row["id"] for row in rows]

# ----------------- ZONES API -----------------

def get_all_zones() -> List[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM zones')
    rows = cursor.fetchall()
    conn.close()
    zones = []
    for row in rows:
        z = dict(row)
        z["polygon"] = json.loads(z["polygon"])
        z["enabled"] = bool(z["enabled"])
        zones.append(z)
    return zones

def get_zone(zone_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM zones WHERE id = ?', (zone_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        z = dict(row)
        z["polygon"] = json.loads(z["polygon"])
        z["enabled"] = bool(z["enabled"])
        return z
    return None

def upsert_zone(zone_id: str, name: str, enabled: bool, polygon: List[List[int]]) -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.cursor()
    now = time.time()
    poly_str = json.dumps(polygon)
    
    cursor.execute('SELECT 1 FROM zones WHERE id = ?', (zone_id,))
    if cursor.fetchone():
        cursor.execute('''
            UPDATE zones SET name = ?, enabled = ?, polygon = ?, updated_at = ?
            WHERE id = ?
        ''', (name, int(enabled), poly_str, now, zone_id))
    else:
        cursor.execute('''
            INSERT INTO zones (id, name, enabled, polygon, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (zone_id, name, int(enabled), poly_str, now, now))
    conn.commit()
    conn.close()
    return {"id": zone_id, "name": name, "enabled": enabled, "polygon": polygon}

def delete_zone(zone_id: str) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM zones WHERE id = ?', (zone_id,))
    changes = cursor.rowcount
    conn.commit()
    conn.close()
    return changes > 0

# ----------------- WATCHLIST API -----------------

def get_watchlist_records() -> List[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM watchlist ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    records = []
    for row in rows:
        r = dict(row)
        r["enabled"] = bool(r["enabled"])
        r["created"] = r["created_at"]
        records.append(r)
    return records

def get_watchlist_record(wl_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM watchlist WHERE id = ?', (wl_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        r = dict(row)
        r["enabled"] = bool(r["enabled"])
        r["created"] = r["created_at"]
        return r
    return None

def insert_watchlist_record(
    wl_id: str,
    name: str,
    status: str,
    enabled: bool = True,
    reference_image_path: Optional[str] = None,
    embedding_path: Optional[str] = None
) -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.cursor()
    now = time.time()
    ref_path = reference_image_path or f"data/watchlist_photos/{wl_id}.jpg"
    emb_path = embedding_path or f"data/watchlist_embeddings/{wl_id}.npy"
    
    cursor.execute('''
        INSERT INTO watchlist (id, name, status, enabled, reference_image_path, embedding_path, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (wl_id, name, status, int(enabled), ref_path, emb_path, now, now))
    conn.commit()
    conn.close()
    return get_watchlist_record(wl_id)

def update_watchlist_record(
    wl_id: str,
    name: Optional[str] = None,
    status: Optional[str] = None,
    enabled: Optional[bool] = None,
    reference_image_path: Optional[str] = None,
    embedding_path: Optional[str] = None
) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    updates = []
    params = []
    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if enabled is not None:
        updates.append("enabled = ?")
        params.append(int(enabled))
    if reference_image_path is not None:
        updates.append("reference_image_path = ?")
        params.append(reference_image_path)
    if embedding_path is not None:
        updates.append("embedding_path = ?")
        params.append(embedding_path)

    if not updates:
        conn.close()
        return False

    updates.append("updated_at = ?")
    params.append(time.time())
    params.append(wl_id)

    sql = f"UPDATE watchlist SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(sql, tuple(params))
    changes = cursor.rowcount
    conn.commit()
    conn.close()
    return changes > 0

def update_watchlist_enabled(wl_id: str, enabled: bool) -> bool:
    return update_watchlist_record(wl_id, enabled=enabled)

def delete_watchlist_record(wl_id: str) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM watchlist WHERE id = ?', (wl_id,))
    changes = cursor.rowcount
    conn.commit()
    conn.close()
    return changes > 0
