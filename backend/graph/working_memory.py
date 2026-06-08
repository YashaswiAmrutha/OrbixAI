"""
OrbixAI working memory — the SQLite session buffer (Docs/architecture §5, §9).

This is the *fast* half of the two-tier memory. It holds the running conversation
(transcript = the model's short-term context) plus an `extract_queue` recording which
turns still need to be promoted into long-term Neo4j memory. It deliberately does
**not** touch Neo4j — it is the cheap staging buffer that *feeds* it, so a reply never
waits on a graph write and the system survives Neo4j being briefly down.

Cadence (who calls what, and when):
  - on each user/assistant turn .... add_message()        — instant, on the critical path
  - after the reply is streamed ..... add_message(enqueue) — stages the turn (off-path)
  - background extraction worker .... pending_batch() -> (NER gate -> extract -> write
                                      to Neo4j) -> mark_done()/mark_failed()

The DB lives at backend/graph/session.db (gitignored). WAL mode + busy_timeout make
it safe for the FastAPI request thread and a background worker thread to share it.

Schema (architecture §5.2 / §9.1):
    sessions      (id, started_at, last_at)
    messages      (id, session_id, role, text, ts)
    extract_queue (msg_id, session_id, status, tries)   status: pending|done|failed
"""

import time
import uuid
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "session.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  INTEGER NOT NULL,
    last_at     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,            -- user | assistant | system
    text        TEXT NOT NULL,
    ts          INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS extract_queue (
    msg_id      INTEGER PRIMARY KEY,      -- one queue row per message
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending | done | failed
    tries       INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (msg_id) REFERENCES messages(id)
);
CREATE INDEX IF NOT EXISTS idx_queue_status ON extract_queue(status, tries);
"""


def _now_ms() -> int:
    return int(time.time() * 1000)


@contextmanager
def _conn():
    """A short-lived connection in WAL mode. Commits on success, rolls back on error."""
    con = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=30000;")
    con.execute("PRAGMA foreign_keys=ON;")
    try:
        con.execute("BEGIN;")
        yield con
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise
    finally:
        con.close()


def init_db() -> None:
    """Create the tables/indexes if they don't exist. Idempotent."""
    con = sqlite3.connect(DB_PATH, timeout=30)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.executescript(_SCHEMA)
        con.commit()
        logger.info("Working memory ready at %s", DB_PATH)
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# sessions
# --------------------------------------------------------------------------- #
def start_session(session_id: str | None = None) -> str:
    """Create (or touch) a session and return its id. Generates a uuid if omitted."""
    sid = session_id or uuid.uuid4().hex
    now = _now_ms()
    with _conn() as con:
        con.execute(
            "INSERT INTO sessions (id, started_at, last_at) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET last_at = excluded.last_at",
            (sid, now, now),
        )
    return sid


def touch_session(session_id: str) -> None:
    """Advance a session's last_at (call on activity)."""
    with _conn() as con:
        con.execute("UPDATE sessions SET last_at = ? WHERE id = ?",
                    (_now_ms(), session_id))


# --------------------------------------------------------------------------- #
# messages + queueing
# --------------------------------------------------------------------------- #
def add_message(session_id: str, role: str, text: str,
                enqueue: bool | None = None) -> int:
    """
    Append a turn to the transcript and return its message id.

    `enqueue` controls whether the turn is staged for long-term extraction.
    Default: only user turns are enqueued (they carry most durable facts); pass
    enqueue=True to also stage assistant/system turns, or False to skip.
    """
    if enqueue is None:
        enqueue = (role == "user")
    ts = _now_ms()
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO messages (session_id, role, text, ts) VALUES (?, ?, ?, ?)",
            (session_id, role, text, ts),
        )
        msg_id = cur.lastrowid
        con.execute("UPDATE sessions SET last_at = ? WHERE id = ?", (ts, session_id))
        if enqueue:
            con.execute(
                "INSERT OR IGNORE INTO extract_queue (msg_id, session_id, status) "
                "VALUES (?, ?, 'pending')",
                (msg_id, session_id),
            )
    return msg_id


def get_messages(session_id: str, limit: int = 50) -> list[dict]:
    """Return the most recent turns for a session, oldest-first (for LLM context)."""
    with _conn() as con:
        rows = con.execute(
            "SELECT id, role, text, ts FROM messages "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


# --------------------------------------------------------------------------- #
# extraction queue (consumed by the background worker)
# --------------------------------------------------------------------------- #
def pending_batch(limit: int = 20, max_tries: int = 3) -> list[dict]:
    """
    Fetch a batch of turns awaiting extraction, joined with their message text.
    Skips rows that have already failed `max_tries` times.
    """
    with _conn() as con:
        rows = con.execute(
            "SELECT q.msg_id, q.session_id, q.tries, m.role, m.text, m.ts "
            "FROM extract_queue q JOIN messages m ON m.id = q.msg_id "
            "WHERE q.status = 'pending' AND q.tries < ? "
            "ORDER BY q.msg_id ASC LIMIT ?",
            (max_tries, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_done(msg_id: int) -> None:
    with _conn() as con:
        con.execute("UPDATE extract_queue SET status = 'done' WHERE msg_id = ?",
                    (msg_id,))


def mark_failed(msg_id: int, max_tries: int = 3) -> None:
    """Increment the try counter; flip to 'failed' once it hits max_tries."""
    with _conn() as con:
        con.execute(
            "UPDATE extract_queue SET tries = tries + 1, "
            "status = CASE WHEN tries + 1 >= ? THEN 'failed' ELSE 'pending' END "
            "WHERE msg_id = ?",
            (max_tries, msg_id),
        )


def queue_stats() -> dict:
    """Counts by status — for monitoring the background pipeline."""
    with _conn() as con:
        rows = con.execute(
            "SELECT status, COUNT(*) AS c FROM extract_queue GROUP BY status"
        ).fetchall()
    return {r["status"]: r["c"] for r in rows}


def claim_pending(limit: int = 20, session_id: str | None = None,
                  max_tries: int = 3) -> list[dict]:
    """
    Atomically claim a batch of pending turns: select + flip to 'processing' in one
    transaction, so several workers (in-app + drain-on-startup) never grab the same
    row. Returns the claimed rows joined with their message text.
    """
    with _conn() as con:  # _conn wraps BEGIN/COMMIT -> the claim is atomic
        where = "q.status = 'pending' AND q.tries < ?"
        params: list = [max_tries]
        if session_id:
            where += " AND q.session_id = ?"
            params.append(session_id)
        params.append(limit)
        rows = con.execute(
            f"SELECT q.msg_id, q.session_id, q.tries, m.role, m.text, m.ts "
            f"FROM extract_queue q JOIN messages m ON m.id = q.msg_id "
            f"WHERE {where} ORDER BY q.msg_id ASC LIMIT ?",
            params,
        ).fetchall()
        for r in rows:
            con.execute("UPDATE extract_queue SET status = 'processing' WHERE msg_id = ?",
                        (r["msg_id"],))
    return [dict(r) for r in rows]


def reset_stale_processing() -> int:
    """Return any rows stuck in 'processing' (from a crash mid-extract) to 'pending'.
    Call on startup so nothing is permanently stranded."""
    with _conn() as con:
        cur = con.execute(
            "UPDATE extract_queue SET status = 'pending' WHERE status = 'processing'")
        return cur.rowcount


def pending_count(session_id: str | None = None) -> int:
    """How many turns still await extraction (pending or in-flight)."""
    with _conn() as con:
        if session_id:
            row = con.execute(
                "SELECT COUNT(*) AS c FROM extract_queue "
                "WHERE status IN ('pending','processing') AND session_id = ?",
                (session_id,)).fetchone()
        else:
            row = con.execute(
                "SELECT COUNT(*) AS c FROM extract_queue "
                "WHERE status IN ('pending','processing')").fetchone()
    return row["c"]


def prune(keep_days: int = 30) -> dict:
    """
    Keep SQLite bounded: drop 'done' queue rows and messages older than keep_days
    that are no longer awaiting extraction. Never deletes pending/processing turns
    (their facts haven't reached Neo4j yet). Returns counts removed.
    """
    cutoff = _now_ms() - keep_days * 86_400_000
    with _conn() as con:
        q = con.execute("DELETE FROM extract_queue WHERE status = 'done'").rowcount
        m = con.execute(
            "DELETE FROM messages WHERE ts < ? AND id NOT IN "
            "(SELECT msg_id FROM extract_queue WHERE status IN ('pending','processing'))",
            (cutoff,)).rowcount
    return {"queue_rows_removed": q, "messages_removed": m}


def vacuum() -> None:
    """Reclaim file space after pruning (must run outside a transaction)."""
    con = sqlite3.connect(DB_PATH, timeout=30)
    try:
        con.execute("VACUUM;")
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# self-test: `python backend/graph/working_memory.py`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_db()
    sid = start_session()
    print(f"session: {sid}")
    add_message(sid, "user", "I moved to Pune last week.")          # enqueued
    add_message(sid, "assistant", "Got it — noting your move to Pune.")  # not enqueued
    add_message(sid, "user", "thanks!")                              # enqueued (gate drops later)
    print("transcript:")
    for m in get_messages(sid):
        print(f"  [{m['role']}] {m['text']}")
    print("pending extraction batch:")
    for item in pending_batch():
        print(f"  msg {item['msg_id']} ({item['role']}): {item['text']}")
    mark_done(pending_batch()[0]["msg_id"])
    print("queue stats:", queue_stats())
