"""
Neo4j connection for OrbixAI's memory graph.

Reads credentials from the gitignored root `.env`. Supports both the lowercase
keys used in this project (`neo4j_connection_url`, `neo4j_username`,
`neo4j_password`) and the conventional uppercase ones
(`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`).
"""

import os
import logging
from pathlib import Path
from functools import lru_cache

from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)

# Root .env (D:\...\OrbixAI\.env) — two levels up from this file.
_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _ROOT / ".env"


def _load_env_file() -> None:
    """Load the root .env into os.environ without overwriting existing vars."""
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _cfg() -> tuple[str, str, str]:
    _load_env_file()
    uri = os.environ.get("neo4j_connection_url") or os.environ.get("NEO4J_URI")
    user = os.environ.get("neo4j_username") or os.environ.get("NEO4J_USERNAME")
    pwd = os.environ.get("neo4j_password") or os.environ.get("NEO4J_PASSWORD")
    if not all([uri, user, pwd]):
        raise RuntimeError(
            "Neo4j credentials missing. Set neo4j_connection_url / neo4j_username "
            "/ neo4j_password in the root .env (or NEO4J_URI / NEO4J_USERNAME / "
            "NEO4J_PASSWORD)."
        )
    return uri, user, pwd


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    """Return a cached singleton Neo4j driver."""
    uri, user, pwd = _cfg()
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    logger.info("Neo4j driver created for %s", uri)
    return driver


def verify() -> bool:
    """Return True if the database is reachable, else log and return False."""
    try:
        get_driver().verify_connectivity()
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("Neo4j not reachable: %s", e)
        return False


def close() -> None:
    """Close the cached driver (call on shutdown)."""
    if get_driver.cache_info().currsize:
        get_driver().close()
        get_driver.cache_clear()
