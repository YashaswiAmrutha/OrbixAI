"""
Benchmark harness for the base-vs-fine-tuned model comparison (research paper).

Runs a fixed set of prompts through the real agent loop (`mcp_host.agent.run_turn`)
against WHICHEVER model is currently active (`llm/model_registry.py`) — switch the
active model, run this, switch again, run again; each run's rows land in
`backend/metrics/runs.jsonl` tagged with the model that answered.

Usage (from backend/):
    python eval_harness.py                  # 1 pass over all prompts
    python eval_harness.py --repeats 3       # 3 passes (for latency variance)
    python eval_harness.py --model llama3.1:8b   # switch model first, then run

Requires Ollama + Neo4j reachable, same as the interactive agent.
"""

import sys
import time
import asyncio
import argparse
import logging
import statistics as stats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "mcp_host"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "llm"))

# Fixed benchmark set — one prompt per capability area named in the plan.
# Keep ids stable across runs; they're the grouping key in the metrics log.
# Expects seed data compatible with the demo graph (a contact this app already
# knows about); adjust the email/name if your Neo4j seed data differs.
BENCHMARK_PROMPTS: list[dict] = [
    {"id": "memory_recall_self",   "text": "Who am I and where do I live?"},
    {"id": "memory_recall_family", "text": "Who is my spouse?"},
    {"id": "send_email",           "text": "Send an email to test@example.com with subject 'Hello' and body 'This is a test.'"},
    {"id": "create_meeting",       "text": "Set up a Google Meet with test@example.com about a project sync."},
    {"id": "create_calendar_event","text": "Add a calendar event titled 'Dentist appointment' on 2026-08-01."},
    {"id": "list_calendar_events", "text": "What's on my calendar coming up?"},
    {"id": "list_emails",          "text": "Show me my latest emails."},
    {"id": "add_task",             "text": "Remind me to submit the report by Friday."},
    {"id": "web_search",           "text": "Search the web for the current version of Python."},
    {"id": "plain_chat",           "text": "What's a good way to stay focused while studying?"},
]


async def _run_once(prompt: dict, session_prefix: str) -> dict:
    from mcp_host.agent import run_turn
    t0 = time.monotonic()
    try:
        out = await run_turn(prompt["text"], session_id=f"{session_prefix}-{prompt['id']}",
                             label=prompt["id"])
        return {"id": prompt["id"], "ok": True, "elapsed": out["elapsed_seconds"],
                "model": out.get("model"), "mode": out.get("mode"), "steps": out["steps"]}
    except Exception as e:  # noqa: BLE001 — record and keep going
        return {"id": prompt["id"], "ok": False, "elapsed": time.monotonic() - t0,
                "model": None, "mode": None, "steps": 0, "error": str(e)}


async def _amain(repeats: int) -> None:
    import uuid
    from graph.working_memory import init_db
    import model_registry

    init_db()
    active = model_registry.get_active()
    print(f"Active model: {active} ({model_registry.tool_calling_mode(active)})\n")

    session_prefix = f"eval-{uuid.uuid4().hex[:8]}"
    results: list[dict] = []
    for i in range(repeats):
        print(f"--- pass {i + 1}/{repeats} ---")
        for prompt in BENCHMARK_PROMPTS:
            r = await _run_once(prompt, f"{session_prefix}-{i}")
            results.append(r)
            status = "OK  " if r["ok"] else "FAIL"
            print(f"  {status} {r['id']:<22} {r['elapsed']:.2f}s  steps={r['steps']}"
                  + (f"  ERROR: {r.get('error')}" if not r["ok"] else ""))

    # summary
    print("\n=== summary ===")
    by_id: dict[str, list[dict]] = {}
    for r in results:
        by_id.setdefault(r["id"], []).append(r)
    n_ok = sum(1 for r in results if r["ok"])
    print(f"model={active}  total={len(results)}  success={n_ok}/{len(results)} "
         f"({100 * n_ok / max(1, len(results)):.0f}%)")
    for pid, rows in by_id.items():
        elapsed = [r["elapsed"] for r in rows]
        ok = sum(1 for r in rows if r["ok"])
        avg = stats.mean(elapsed) if elapsed else 0.0
        print(f"  {pid:<22} success={ok}/{len(rows)}  avg={avg:.2f}s")
    print(f"\nRaw rows appended to backend/metrics/runs.jsonl (label = prompt id, "
         f"model = {active}).")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1, help="passes over the benchmark set")
    parser.add_argument("--model", default=None,
                        help="switch the active model before running (e.g. llama3.1:8b)")
    args = parser.parse_args()

    if args.model:
        import model_registry
        model_registry.set_active(args.model)
        print(f"Switched active model to {args.model}")

    asyncio.run(_amain(args.repeats))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
