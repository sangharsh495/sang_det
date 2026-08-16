"""SQLite job store.

Holds the video queue, per-video progress, saved-crop provenance and an event
log. WAL mode so the dashboard can read while the worker writes. Every piece
of run state lives here, which is what makes an interrupted 15-hour batch
resumable instead of restartable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Iterable

from .config import ROOT

DB_PATH = ROOT / "data" / "sang_det.db"

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"

_local = threading.local()
_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    url               TEXT NOT NULL,
    title             TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',
    error             TEXT,
    frames_processed  INTEGER NOT NULL DEFAULT 0,
    plates_saved      INTEGER NOT NULL DEFAULT 0,
    detections_seen   INTEGER NOT NULL DEFAULT 0,
    rejected_conf     INTEGER NOT NULL DEFAULT 0,
    rejected_size     INTEGER NOT NULL DEFAULT 0,
    rejected_blur     INTEGER NOT NULL DEFAULT 0,
    rejected_dupe     INTEGER NOT NULL DEFAULT 0,
    duration_s        REAL,
    position_s        REAL NOT NULL DEFAULT 0,
    attempts          INTEGER NOT NULL DEFAULT 0,
    added_at          REAL NOT NULL,
    started_at        REAL,
    finished_at       REAL,
    heartbeat_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);

CREATE TABLE IF NOT EXISTS plates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id     INTEGER NOT NULL,
    filename     TEXT NOT NULL,
    frame_index  INTEGER NOT NULL,
    timestamp_s  REAL NOT NULL,
    confidence   REAL NOT NULL,
    box          TEXT NOT NULL,
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    blur_score   REAL,
    phash        TEXT,
    bytes        INTEGER,
    saved_at     REAL NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(id)
);
CREATE INDEX IF NOT EXISTS idx_plates_video ON plates(video_id);
CREATE INDEX IF NOT EXISTS idx_plates_saved ON plates(saved_at);

CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id  INTEGER,
    level     TEXT NOT NULL,
    message   TEXT NOT NULL,
    at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_at ON events(at);

CREATE TABLE IF NOT EXISTS state (
    key    TEXT PRIMARY KEY,
    value  TEXT
);
"""


def connect() -> sqlite3.Connection:
    """Thread-local connection. Each worker thread gets its own handle."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


def init() -> None:
    conn = connect()
    with _write_lock:
        conn.executescript(SCHEMA)
    # Any job left mid-flight by a hard kill goes back in the queue. Its
    # position_s checkpoint survives, so it resumes rather than restarts.
    reclaim_orphans()


def reclaim_orphans() -> int:
    conn = connect()
    with _write_lock:
        cur = conn.execute(
            "UPDATE videos SET status=?, started_at=NULL WHERE status=?",
            (STATUS_PENDING, STATUS_PROCESSING),
        )
    return cur.rowcount or 0


# --------------------------------------------------------------------- state


def set_state(key: str, value: Any) -> None:
    conn = connect()
    with _write_lock:
        conn.execute(
            "INSERT INTO state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )


def get_state(key: str, default: Any = None) -> Any:
    row = connect().execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (TypeError, ValueError):
        return default


# -------------------------------------------------------------------- videos


def add_videos(urls: Iterable[str]) -> tuple[int, int]:
    """Insert URLs, skipping ones already queued or completed. -> (added, skipped)"""
    conn = connect()
    added = skipped = 0
    now = time.time()
    with _write_lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for url in urls:
                url = url.strip()
                if not url:
                    continue
                existing = conn.execute(
                    "SELECT id FROM videos WHERE url=? AND status IN (?,?,?)",
                    (url, STATUS_PENDING, STATUS_PROCESSING, STATUS_DONE),
                ).fetchone()
                if existing:
                    skipped += 1
                    continue
                conn.execute(
                    "INSERT INTO videos(url, status, added_at) VALUES(?,?,?)",
                    (url, STATUS_PENDING, now),
                )
                added += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return added, skipped


def claim_next_video() -> dict[str, Any] | None:
    """Atomically take the oldest pending job. None if the queue is empty."""
    conn = connect()
    now = time.time()
    with _write_lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT * FROM videos WHERE status=? ORDER BY id LIMIT 1",
                (STATUS_PENDING,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                "UPDATE videos SET status=?, started_at=COALESCE(started_at,?), "
                "heartbeat_at=?, attempts=attempts+1, error=NULL WHERE id=?",
                (STATUS_PROCESSING, now, now, row["id"]),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    claimed = dict(row)
    # `row` was read before the UPDATE, so reflect the incremented value here.
    # The retry check compares against this, and an off-by-one would spend an
    # extra full attempt on every dead link in the batch.
    claimed["attempts"] = int(claimed.get("attempts") or 0) + 1
    return claimed


def update_video(video_id: int, **fields: Any) -> None:
    if not fields:
        return
    allowed = {
        "title", "status", "error", "frames_processed", "plates_saved",
        "detections_seen", "rejected_conf", "rejected_size", "rejected_blur",
        "rejected_dupe", "duration_s", "position_s", "started_at",
        "finished_at", "heartbeat_at",
    }
    cols = {k: v for k, v in fields.items() if k in allowed}
    if not cols:
        return
    sql = "UPDATE videos SET " + ", ".join(f"{k}=?" for k in cols) + " WHERE id=?"
    with _write_lock:
        connect().execute(sql, (*cols.values(), video_id))


def bump_video(video_id: int, **deltas: int) -> None:
    """Increment counters without a read-modify-write race."""
    allowed = {
        "frames_processed", "plates_saved", "detections_seen", "rejected_conf",
        "rejected_size", "rejected_blur", "rejected_dupe",
    }
    cols = {k: v for k, v in deltas.items() if k in allowed and v}
    if not cols:
        return
    sql = (
        "UPDATE videos SET "
        + ", ".join(f"{k}={k}+?" for k in cols)
        + ", heartbeat_at=? WHERE id=?"
    )
    with _write_lock:
        connect().execute(sql, (*cols.values(), time.time(), video_id))


def get_video(video_id: int) -> dict[str, Any] | None:
    row = connect().execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
    return dict(row) if row else None


def list_videos(limit: int = 500) -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT * FROM videos ORDER BY "
        "CASE status WHEN 'processing' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END, id",
        (),
    ).fetchmany(limit)
    return [dict(r) for r in rows]


def requeue_video(video_id: int, reset_progress: bool = False) -> None:
    with _write_lock:
        if reset_progress:
            connect().execute(
                "UPDATE videos SET status=?, error=NULL, finished_at=NULL, "
                "position_s=0, frames_processed=0, attempts=0 WHERE id=?",
                (STATUS_PENDING, video_id),
            )
        else:
            connect().execute(
                "UPDATE videos SET status=?, error=NULL, finished_at=NULL WHERE id=?",
                (STATUS_PENDING, video_id),
            )


def requeue_all_errored() -> int:
    with _write_lock:
        cur = connect().execute(
            "UPDATE videos SET status=?, error=NULL, finished_at=NULL, attempts=0 "
            "WHERE status=?",
            (STATUS_PENDING, STATUS_ERROR),
        )
    return cur.rowcount or 0


def cancel_video(video_id: int) -> None:
    with _write_lock:
        connect().execute(
            "UPDATE videos SET status=?, finished_at=? WHERE id=? AND status IN (?,?)",
            (STATUS_CANCELLED, time.time(), video_id, STATUS_PENDING, STATUS_PROCESSING),
        )


def delete_video(video_id: int) -> None:
    with _write_lock:
        conn = connect()
        conn.execute("DELETE FROM plates WHERE video_id=?", (video_id,))
        conn.execute("DELETE FROM videos WHERE id=?", (video_id,))


# -------------------------------------------------------------------- plates


def record_plate(
    video_id: int,
    filename: str,
    frame_index: int,
    timestamp_s: float,
    confidence: float,
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    blur_score: float | None,
    phash: str | None,
    nbytes: int | None,
) -> None:
    with _write_lock:
        connect().execute(
            "INSERT INTO plates(video_id, filename, frame_index, timestamp_s, "
            "confidence, box, width, height, blur_score, phash, bytes, saved_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                video_id, filename, frame_index, timestamp_s, confidence,
                json.dumps(list(box)), width, height, blur_score, phash,
                nbytes, time.time(),
            ),
        )


def recent_plates(limit: int = 40) -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT p.*, v.url AS source_url FROM plates p "
        "LEFT JOIN videos v ON v.id = p.video_id "
        "ORDER BY p.id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def totals() -> dict[str, Any]:
    conn = connect()
    row = conn.execute(
        "SELECT "
        " COUNT(*) AS videos,"
        " SUM(status='done') AS done,"
        " SUM(status='pending') AS pending,"
        " SUM(status='processing') AS processing,"
        " SUM(status='error') AS errored,"
        " SUM(status='cancelled') AS cancelled,"
        " COALESCE(SUM(frames_processed),0) AS frames,"
        " COALESCE(SUM(plates_saved),0) AS plates,"
        " COALESCE(SUM(detections_seen),0) AS detections,"
        " COALESCE(SUM(rejected_conf),0) AS rejected_conf,"
        " COALESCE(SUM(rejected_size),0) AS rejected_size,"
        " COALESCE(SUM(rejected_blur),0) AS rejected_blur,"
        " COALESCE(SUM(rejected_dupe),0) AS rejected_dupe"
        " FROM videos"
    ).fetchone()
    out = {k: (row[k] or 0) for k in row.keys()}
    size = conn.execute("SELECT COALESCE(SUM(bytes),0) AS b FROM plates").fetchone()
    out["output_bytes"] = size["b"] or 0
    return out


# -------------------------------------------------------------------- events


def log_event(level: str, message: str, video_id: int | None = None) -> None:
    with _write_lock:
        connect().execute(
            "INSERT INTO events(video_id, level, message, at) VALUES(?,?,?,?)",
            (video_id, level, message[:2000], time.time()),
        )


def recent_events(limit: int = 100) -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def prune_events(keep: int = 5000) -> None:
    with _write_lock:
        connect().execute(
            "DELETE FROM events WHERE id NOT IN "
            "(SELECT id FROM events ORDER BY id DESC LIMIT ?)",
            (keep,),
        )
