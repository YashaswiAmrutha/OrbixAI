"""
Apply the OrbixAI memory-graph schema, or inspect the live database.

Usage:
    python backend/graph/init_schema.py            # apply schema.cypher
    python backend/graph/init_schema.py --inspect  # print labels/rels/props/counts
    python backend/graph/init_schema.py --both     # inspect, then apply

Statements in schema.cypher are separated by ';'. Lines starting with '//' are
comments. Everything is idempotent (IF NOT EXISTS), so re-running is safe.
"""

import sys
import logging
from pathlib import Path

from connection import get_driver, verify  # type: ignore  (run from this dir)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("init_schema")

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.cypher"


def _statements(text: str):
    """Yield Cypher statements, stripping // comment lines and blank lines."""
    cleaned = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        cleaned.append(line)
    for stmt in "\n".join(cleaned).split(";"):
        stmt = stmt.strip()
        if stmt:
            yield stmt


def apply_schema() -> None:
    text = SCHEMA_FILE.read_text(encoding="utf-8")
    ok = fail = 0
    with get_driver().session() as s:
        for stmt in _statements(text):
            label = stmt.split("\n")[0][:70]
            try:
                s.run(stmt)
                ok += 1
                log.info("  ok   %s", label)
            except Exception as e:  # noqa: BLE001
                fail += 1
                log.error("  FAIL %s\n       %s", label, e)
    log.info("\nSchema applied: %d ok, %d failed.", ok, fail)


def inspect() -> None:
    with get_driver().session() as s:
        labels = [r["label"] for r in s.run(
            "CALL db.labels() YIELD label RETURN label ORDER BY label")]
        rels = [r["relationshipType"] for r in s.run(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN relationshipType ORDER BY relationshipType")]
        props = [r["propertyKey"] for r in s.run(
            "CALL db.propertyKeys() YIELD propertyKey "
            "RETURN propertyKey ORDER BY propertyKey")]

        log.info("\nLABELS (%d): %s", len(labels), ", ".join(labels) or "—")
        log.info("\nRELATIONSHIPS (%d): %s", len(rels), ", ".join(rels) or "—")
        log.info("\nPROPERTY KEYS (%d): %s", len(props), ", ".join(props) or "—")

        log.info("\nNODE COUNTS:")
        for lbl in labels:
            c = s.run(f"MATCH (n:`{lbl}`) RETURN count(n) AS c").single()["c"]
            log.info("  %-16s %d", lbl, c)

        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        edges = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        log.info("\nTOTAL: %d nodes, %d relationships", nodes, edges)

        log.info("\nCONSTRAINTS:")
        for r in s.run("SHOW CONSTRAINTS YIELD name, type RETURN name, type"):
            log.info("  %-28s %s", r["name"], r["type"])
        log.info("\nINDEXES:")
        for r in s.run("SHOW INDEXES YIELD name, type, labelsOrTypes, properties "
                       "RETURN name, type, labelsOrTypes, properties"):
            log.info("  %-22s %-10s %s %s", r["name"], r["type"],
                     r["labelsOrTypes"], r["properties"])


def main() -> int:
    if not verify():
        log.error("Cannot reach Neo4j — is it running at the configured URI?")
        return 1
    arg = sys.argv[1] if len(sys.argv) > 1 else "--apply"
    if arg == "--inspect":
        inspect()
    elif arg == "--both":
        inspect()
        log.info("\n--- applying schema ---")
        apply_schema()
    else:
        apply_schema()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
