"""
Per-turn metrics logging — the raw data for the base-vs-fine-tuned model
comparison (research paper). Append-only JSONL, one line per agent turn, so
it's trivial to load into pandas later. Not a database: no queries, no
rotation — `eval_harness.py` is the only thing that needs to read it back,
and it does so by loading the whole file.

File lives at backend/metrics/runs.jsonl (gitignored, like session.db).
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

_METRICS_DIR = Path(__file__).resolve().parents[1] / "metrics"
_RUNS_FILE = _METRICS_DIR / "runs.jsonl"
_lock = Lock()


def record_turn(*, model: str, mode: str, session_id: str, trace: list[dict],
                success: bool, elapsed_seconds: float, steps: int,
                error: str | None = None, label: str | None = None) -> None:
    """
    Append one row for a completed agent turn.

    `label` is an optional caller-supplied tag (e.g. the benchmark prompt id
    from eval_harness.py) so repeated runs of the same prompt can be grouped;
    None for ordinary interactive turns.
    """
    row = {
        "ts": time.time(),
        "model": model,
        "tool_calling_mode": mode,
        "session_id": session_id,
        "label": label,
        "success": bool(success),
        "elapsed_seconds": elapsed_seconds,
        "steps": steps,
        "tools_called": [t.get("tool") for t in (trace or [])],
        "tool_errors": [t.get("tool") for t in (trace or [])
                        if isinstance(t.get("result"), dict) and t["result"].get("error")],
        "error": error,
    }
    try:
        with _lock:
            _METRICS_DIR.mkdir(parents=True, exist_ok=True)
            with _RUNS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
    except Exception as e:  # noqa: BLE001 — metrics must never break a real turn
        logger.error("metrics.record_turn failed to write: %s", e)


def load_runs() -> list[dict]:
    """Read every recorded row (for eval_harness.py's summary and any later analysis)."""
    if not _RUNS_FILE.exists():
        return []
    rows = []
    with _RUNS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
