"""
Watchlist Service — Persistent Watchlist Management and InsightFace Single-Face Enrollment.

Responsibilities:
- Manage persistent watchlist records in backend/data/command_center.db SQLite database.
- Enforce strict single-face enrollment validation (0 faces -> reject, >1 face -> reject, exactly 1 -> enroll).
- Store private reference images in data/watchlist_photos/ (never publicly served).
- Store 512D ArcFace embeddings in data/watchlist_embeddings/ as .npy files (never exposed via API).
- Synchronize active enabled records with FaceMatcher in real-time.
- Handle backward-compatible migration of legacy records on startup.
"""

import os
import uuid
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
import cv2
import numpy as np

from .face_engine import FaceEngine
from .face_matcher import FaceMatcher
from ... import database

PHOTOS_DIR = os.path.join(database.DB_DIR, "watchlist_photos")
EMBEDDINGS_DIR = os.path.join(database.DB_DIR, "watchlist_embeddings")


class WatchlistService:
    """
    Manages persistent watchlist enrollment, enable/disable toggles, deletion,
    and runtime synchronization with the FaceMatcher engine.
    """

    def __init__(
        self,
        engine: Optional[FaceEngine] = None,
        matcher: Optional[FaceMatcher] = None
    ):
        self.engine = engine if engine is not None else FaceEngine()
        self.matcher = matcher if matcher is not None else FaceMatcher()
        self._lock = threading.Lock()

        # Ensure private storage directories exist
        os.makedirs(PHOTOS_DIR, exist_ok=True)
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

        # Load persisted watchlist records from SQLite and synchronize with FaceMatcher
        self.sync_matcher_from_db()

    def sync_matcher_from_db(self):
        """
        Loads all watchlist records from SQLite command_center.db into the in-memory FaceMatcher.
        Migrates legacy records if reference photos exist.
        """
        with self._lock:
            records = database.get_watchlist_records()
            loaded_count = 0

            for rec in records:
                wl_id = str(rec["id"])
                name = rec.get("name", "Unknown")
                status = rec.get("status", "WATCHLIST")
                enabled = bool(rec.get("enabled", True))
                emb_path = rec.get("embedding_path", "")
                ref_path = rec.get("reference_image_path", "")

                emb = None

                # 1. Try loading existing .npy embedding
                if emb_path and emb_path.endswith(".npy") and os.path.exists(emb_path):
                    try:
                        emb = np.load(emb_path)
                    except Exception as e:
                        print(f"[WatchlistService] Error loading embedding {emb_path}: {e}")

                # 2. Check if .npy exists in standard directory
                std_emb_path = os.path.join(EMBEDDINGS_DIR, f"{wl_id}.npy")
                if emb is None and os.path.exists(std_emb_path):
                    try:
                        emb = np.load(std_emb_path)
                        database.update_watchlist_record(wl_id, embedding_path=std_emb_path)
                    except Exception as e:
                        print(f"[WatchlistService] Error loading std embedding {std_emb_path}: {e}")

                # 3. If no .npy but reference image exists, re-extract with InsightFace (migration)
                if emb is None and ref_path and os.path.exists(ref_path):
                    try:
                        img = cv2.imread(ref_path)
                        if img is not None:
                            success, extracted_emb, _, count = self.engine.extract_single_face(img)
                            if success and extracted_emb is not None:
                                emb = extracted_emb
                                np.save(std_emb_path, emb)
                                database.update_watchlist_record(
                                    wl_id,
                                    embedding_path=std_emb_path
                                )
                                print(f"[WatchlistService] Migrated legacy record {wl_id} ({name}) to ArcFace .npy")
                    except Exception as e:
                        print(f"[WatchlistService] Error migrating record {wl_id}: {e}")

                # Register in FaceMatcher if embedding is available
                if emb is not None:
                    self.matcher.register_face(
                        person_id=wl_id,
                        name=name,
                        status=status,
                        embedding=emb,
                        enabled=enabled
                    )
                    loaded_count += 1
                else:
                    print(f"[WatchlistService] Warning: No embedding available for watchlist record {wl_id} ({name})")

            print(f"[WatchlistService] Synchronized {loaded_count} face records into FaceMatcher")

    def enroll(
        self,
        name: str,
        status: str,
        image_bytes: bytes
    ) -> Dict[str, Any]:
        """
        Enrolls a new person into the watchlist:
        1. Validates name and status (WATCHLIST or CRITICAL).
        2. Decodes photo and runs InsightFace FaceEngine.
        3. Strict single-face validation (0 -> reject, >1 -> reject, exactly 1 -> enroll).
        4. Saves private photo and 512D ArcFace embedding (.npy).
        5. Inserts record into SQLite command_center.db.
        6. Registers record with FaceMatcher.
        7. Returns safe metadata dictionary without embedding or private paths.
        """
        if not name or not name.strip():
            raise ValueError("Name is required")

        status_norm = status.strip().upper()
        if status_norm not in ("WATCHLIST", "CRITICAL"):
            raise ValueError("Status must be WATCHLIST or CRITICAL")

        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Empty photo file")

        # Decode image using OpenCV
        img_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError("Failed to decode image. Please provide a valid JPEG or PNG image.")

        # Validate single face with InsightFace
        success, embedding, err_msg, face_count = self.engine.extract_single_face(img)
        if not success or embedding is None:
            raise ValueError(err_msg or "Face detection failed")

        wl_id = str(uuid.uuid4())[:8]

        # Define private file paths
        photo_path = os.path.join(PHOTOS_DIR, f"{wl_id}.jpg")
        emb_path = os.path.join(EMBEDDINGS_DIR, f"{wl_id}.npy")

        # Write private photo and embedding to disk
        try:
            cv2.imwrite(photo_path, img)
            np.save(emb_path, embedding)
        except Exception as e:
            if os.path.exists(photo_path):
                os.remove(photo_path)
            if os.path.exists(emb_path):
                os.remove(emb_path)
            raise RuntimeError(f"Failed to save enrollment data: {e}")

        # Insert record into SQLite DB
        with self._lock:
            record = database.insert_watchlist_record(
                wl_id=wl_id,
                name=name.strip(),
                status=status_norm,
                enabled=True,
                reference_image_path=photo_path,
                embedding_path=emb_path
            )

            # Register with FaceMatcher
            self.matcher.register_face(
                person_id=wl_id,
                name=name.strip(),
                status=status_norm,
                embedding=embedding,
                enabled=True
            )

        print(f"[WatchlistService] Enrolled '{name.strip()}' ({status_norm}) with ID {wl_id}")

        return {
            "id": record["id"],
            "name": record["name"],
            "status": record["status"],
            "enabled": bool(record["enabled"]),
            "created": record["created_at"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
        }

    def list_records(self) -> List[Dict[str, Any]]:
        """Returns all watchlist records metadata without exposing embeddings or private paths."""
        records = database.get_watchlist_records()
        sanitized = []
        for r in records:
            sanitized.append({
                "id": r["id"],
                "name": r["name"],
                "status": r["status"],
                "enabled": bool(r["enabled"]),
                "created": r.get("created_at", r.get("created", time.time())),
                "created_at": r.get("created_at", time.time()),
                "updated_at": r.get("updated_at", time.time()),
            })
        return sanitized

    def get_record(self, wl_id: str) -> Optional[Dict[str, Any]]:
        """Gets a single watchlist record metadata."""
        rec = database.get_watchlist_record(wl_id)
        if rec is None:
            return None
        return {
            "id": rec["id"],
            "name": rec["name"],
            "status": rec["status"],
            "enabled": bool(rec["enabled"]),
            "created": rec.get("created_at", time.time()),
            "created_at": rec.get("created_at", time.time()),
            "updated_at": rec.get("updated_at", time.time()),
        }

    def enable_record(self, wl_id: str) -> bool:
        """Enables a watchlist record for active face matching."""
        with self._lock:
            rec = database.get_watchlist_record(wl_id)
            if rec is None:
                return False
            success = database.update_watchlist_enabled(wl_id, True)
            if success:
                # Reload / register into FaceMatcher
                emb_path = rec.get("embedding_path", "")
                if emb_path and os.path.exists(emb_path):
                    try:
                        emb = np.load(emb_path)
                        self.matcher.register_face(
                            person_id=wl_id,
                            name=rec["name"],
                            status=rec["status"],
                            embedding=emb,
                            enabled=True
                        )
                    except Exception as e:
                        print(f"[WatchlistService] Error loading embedding during enable: {e}")
            return success

    def disable_record(self, wl_id: str) -> bool:
        """Disables a watchlist record from active face matching."""
        with self._lock:
            rec = database.get_watchlist_record(wl_id)
            if rec is None:
                return False
            success = database.update_watchlist_enabled(wl_id, False)
            if success:
                # Mark disabled in FaceMatcher
                emb_path = rec.get("embedding_path", "")
                if emb_path and os.path.exists(emb_path):
                    try:
                        emb = np.load(emb_path)
                        self.matcher.register_face(
                            person_id=wl_id,
                            name=rec["name"],
                            status=rec["status"],
                            embedding=emb,
                            enabled=False
                        )
                    except Exception:
                        pass
            return success

    def delete_record(self, wl_id: str) -> bool:
        """Deletes a watchlist record and removes its private files."""
        with self._lock:
            rec = database.get_watchlist_record(wl_id)
            if rec is None:
                return False

            ref_path = rec.get("reference_image_path", "")
            emb_path = rec.get("embedding_path", "")

            # Delete from SQLite DB
            deleted = database.delete_watchlist_record(wl_id)

            # Unregister from FaceMatcher
            self.matcher.unregister_face(wl_id)

            # Delete private files from disk
            if ref_path and os.path.exists(ref_path):
                try:
                    os.remove(ref_path)
                except Exception:
                    pass

            if emb_path and os.path.exists(emb_path):
                try:
                    os.remove(emb_path)
                except Exception:
                    pass

            std_emb = os.path.join(EMBEDDINGS_DIR, f"{wl_id}.npy")
            if os.path.exists(std_emb):
                try:
                    os.remove(std_emb)
                except Exception:
                    pass

            std_photo = os.path.join(PHOTOS_DIR, f"{wl_id}.jpg")
            if os.path.exists(std_photo):
                try:
                    os.remove(std_photo)
                except Exception:
                    pass

            return deleted
