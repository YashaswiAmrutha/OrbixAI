"""
OrbixAI Tasks MCP server.

A tiny persistent to-do list the agent can manage as tool calls, so requests like
"remind me to send the report" or the natural follow-ups of a bigger task ("prepare
for the meeting") land on a real task list instead of being forgotten. Backed by a
JSON file (backend/tasks_store.json). The host surfaces add_task results to the
frontend To-Do panel, so tasks the agent creates show up in the UI automatically.

Tools
  read  (readOnlyHint):  list_tasks
  write (action)      :  add_task · complete_task · delete_task

Run (stdio transport):
    python backend/mcp_servers/tasks_server.py
"""

import json
import uuid
from pathlib import Path
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

_STORE = Path(__file__).resolve().parents[1] / "tasks_store.json"

mcp = FastMCP(
    "orbix-tasks",
    instructions=(
        "OrbixAI to-do list. Use add_task when the user asks to be reminded of "
        "something or when a larger request implies a follow-up, list_tasks to see "
        "what's pending, and complete_task/delete_task to update it. Keep task text "
        "short and actionable."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_ACTION = ToolAnnotations(readOnlyHint=False, idempotentHint=False)
_DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True)


def _load() -> list:
    if _STORE.exists():
        try:
            return json.loads(_STORE.read_text(encoding="utf-8")).get("tasks", [])
        except Exception:  # noqa: BLE001
            return []
    return []


def _save(tasks: list) -> None:
    _STORE.write_text(json.dumps({"tasks": tasks}, indent=2, ensure_ascii=False),
                      encoding="utf-8")


@mcp.tool(annotations=_ACTION)
def add_task(text: str) -> dict:
    """
    Add a to-do item. Returns the created task {id, text, done, created_at}. Use this
    whenever the user wants to remember or be reminded of something, or when finishing
    a request leaves an obvious next step.
    """
    text = (text or "").strip()
    if not text:
        return {"error": "task text is required"}
    tasks = _load()
    task = {"id": uuid.uuid4().hex[:8], "text": text, "done": False,
            "created_at": datetime.utcnow().isoformat()}
    tasks.insert(0, task)
    _save(tasks)
    return {**task, "added": True}


@mcp.tool(annotations=_READONLY)
def list_tasks(include_done: bool = False) -> dict:
    """List to-do items. By default only pending ones. Returns {tasks, count}."""
    tasks = _load()
    if not include_done:
        tasks = [t for t in tasks if not t.get("done")]
    return {"tasks": tasks, "count": len(tasks)}


@mcp.tool(annotations=_ACTION)
def complete_task(task_id: str = "", text: str = "") -> dict:
    """
    Mark a task done, found by its id OR by matching its text. Returns
    {completed: true, task} or an error if no match.
    """
    tasks = _load()
    tid = (task_id or "").strip()
    needle = (text or "").strip().lower()
    for t in tasks:
        if (tid and t["id"] == tid) or (needle and needle in t["text"].lower()):
            t["done"] = True
            _save(tasks)
            return {"completed": True, "task": t}
    return {"error": "no matching task found"}


@mcp.tool(annotations=_DESTRUCTIVE)
def delete_task(task_id: str = "", text: str = "") -> dict:
    """Delete a task by id OR matching text. Returns {deleted: true} or an error."""
    tasks = _load()
    tid = (task_id or "").strip()
    needle = (text or "").strip().lower()
    for i, t in enumerate(tasks):
        if (tid and t["id"] == tid) or (needle and needle in t["text"].lower()):
            tasks.pop(i)
            _save(tasks)
            return {"deleted": True}
    return {"error": "no matching task found"}


if __name__ == "__main__":
    mcp.run()  # stdio transport
