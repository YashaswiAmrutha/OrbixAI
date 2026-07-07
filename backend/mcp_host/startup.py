"""
App startup orchestration for OrbixAI.

On launch we must guarantee the pieces the agent depends on:
  1. Ollama is serving        -> if not, launch `ollama serve` detached and wait.
  2. Working memory is ready   -> init the SQLite schema.
  3. Neo4j status is reported  -> we can't auto-start Neo4j Desktop; we log clearly
     if it's down.

(Drain-on-startup was removed with the old SQLite-queue write pathway; durable memory
writes now happen per turn via the LangGraph background write path.)

Called from FastAPI's lifespan in main.py. Safe to call more than once.
"""

import sys
import time
import logging
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))

import working_memory as wm   # type: ignore  # noqa: E402

logger = logging.getLogger(__name__)


def ensure_ollama(wait_seconds: int = 30) -> bool:
    """Return True if Ollama responds; otherwise launch `ollama serve` and wait."""
    import ollama
    try:
        ollama.list()
        logger.info("Ollama already running.")
        return True
    except Exception:
        logger.info("Ollama not responding — starting `ollama serve`…")

    try:
        # Detached so it keeps running for the app's lifetime and isn't tied to a shell.
        flags = 0
        kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            flags = (subprocess.CREATE_NEW_PROCESS_GROUP
                     | getattr(subprocess, "DETACHED_PROCESS", 0))
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(["ollama", "serve"], creationflags=flags, **kwargs)
    except FileNotFoundError:
        logger.error("`ollama` not on PATH — install Ollama or start it manually.")
        return False

    for _ in range(wait_seconds):
        time.sleep(1)
        try:
            ollama.list()
            logger.info("Ollama is up.")
            return True
        except Exception:
            continue
    logger.error("Ollama did not become ready in %ds.", wait_seconds)
    return False


def neo4j_status() -> bool:
    try:
        from connection import verify  # type: ignore
        ok = verify()
        logger.info("Neo4j reachable." if ok else
                    "Neo4j NOT reachable — start the DB; queued writes will drain later.")
        return ok
    except Exception as e:  # noqa: BLE001
        logger.error("Neo4j check failed: %s", e)
        return False


def run_startup() -> dict:
    """Full startup sequence. Returns a small status report."""
    report: dict = {}
    report["ollama"] = ensure_ollama()
    wm.init_db()
    report["neo4j"] = neo4j_status()

    # NOTE: drain-on-startup has been removed along with the old SQLite-queue write
    # pathway. Durable memory writes now happen through the LangGraph background write
    # path (orchestration.extraction_executor) per turn, not via a startup queue drain.
    logger.info("Startup report: %s", report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    print(run_startup())
