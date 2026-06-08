"""
End-to-end pipeline test: query -> LLM -> MCP tool calls -> Neo4j read/write.

Proves the whole loop works:
  1. a recall question        -> model calls a read tool, answers from the graph
  2. a contradiction stated   -> model calls remember_fact, the graph is updated
  3. direct DB check          -> the new value really landed (overwrite-in-place)
  4. a follow-up question     -> model recalls the NEW value
  5. cleanup                  -> restore the original value, wipe test history

Needs: Neo4j running, the agent model pulled. Run from backend/mcp_host/:
    python test_e2e.py
"""

import sys
import asyncio
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))

import agent  # type: ignore
import memory as mem  # type: ignore


def _hr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8")

    original = (mem.get_fact("user:self", "home_city") or {}).get("value")
    print(f"starting home_city in graph: {original!r}")
    sid = "e2e-test-session"

    try:
        _hr("1) RECALL question — 'Where do I live?'")
        r1 = await agent.run_turn("Where do I live?", session_id=sid)
        _show(r1)
        assert any(t["readonly"] for t in r1["trace"]), "expected a read tool call"

        _hr("2) CONTRADICTION — 'Actually, I just moved to Pune.'")
        r2 = await agent.run_turn("Actually, I just moved to Pune.", session_id=sid)
        _show(r2)
        wrote = [t for t in r2["trace"] if t["tool"] in ("remember_fact", "correct")]
        assert wrote, "expected a remember_fact/correct write call"

        _hr("3) DIRECT DB CHECK — did the graph overwrite in place?")
        fact = mem.get_fact("user:self", "home_city")
        hist = mem.fact_history("user:self", "home_city")
        print("  get_fact(home_city):", fact)
        print("  fact_history       :", hist)
        assert fact and "pune" in str(fact["value"]).lower(), "home_city should now be Pune"

        _hr("4) FOLLOW-UP — 'Remind me, which city do I live in now?'")
        r3 = await agent.run_turn("Remind me, which city do I live in now?", session_id=sid)
        _show(r3)
        assert "pune" in r3["reply"].lower(), "reply should reflect the new value Pune"

        print("\n✅ PIPELINE OK: query -> LLM -> MCP -> Neo4j read & write all verified.")
        return 0
    finally:
        _hr("CLEANUP — restoring original value + wiping test history")
        from connection import get_driver  # type: ignore
        with get_driver().session() as s:
            s.run("MATCH (f:Fact {key:'fact:user:self:home_city'}) "
                  "OPTIONAL MATCH (f)-[:HAS_OBSERVATION]->(o) DETACH DELETE o")
            if original is not None:
                s.run("MATCH (f:Fact {key:'fact:user:self:home_city'}) "
                      "SET f.value=$v, f.current=true, f.updated_at=timestamp()",
                      v=original)
        print("  restored home_city ->", (mem.get_fact("user:self", "home_city") or {}).get("value"))


def _show(out: dict):
    for t in out["trace"]:
        tag = "READ " if t["readonly"] else "WRITE"
        print(f"  [{tag}] {t['tool']}({t['args']}) -> {str(t['result'])[:110]}")
    print("  REPLY:", out["reply"])


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
