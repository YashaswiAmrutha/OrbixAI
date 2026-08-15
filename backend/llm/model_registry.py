"""
OrbixAI active-model registry.

Which Ollama model the agent talks to is now a runtime setting, not a constant
frozen at process start — this is what lets the research comparison (base
llama3.1:8b vs. the fine-tuned gpraneeth555/llama-3-13k) be done by *toggling*
one model at a time instead of running both simultaneously (this machine can't
carry that load). A toggle here takes effect on the very next agent turn.

Same shape as `mcp_host/registry.py`: a small static CATALOG + a persisted
{"active": ...} state file (`backend/model_state.json`, gitignored like
`mcp_state.json`).
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

_BACKEND = Path(__file__).resolve().parents[1]
_STATE_FILE = _BACKEND / "model_state.json"
_lock = Lock()

# The models this app knows about. `tool_calling` records whether the model's
# Ollama template understands native tool schemas (checked via `ollama show
# <model> --modelfile`): llama3.1:8b does; the fine-tune's template is a bare
# prompt block with no {{ .Tools }}/{{ .ToolCalls }} — it answers in a
# `### Output: {"tool":..., "action":..., "parameters":...}` text block instead
# (confirmed by a live test call). `mcp_host/legacy_tool_adapter.py` is what
# actually branches on this value.
CATALOG: list[dict] = [
    {
        "id": "llama3.1:8b",
        "label": "Base — Llama 3.1 8B",
        "tool_calling": "native",
    },
    {
        "id": "gpraneeth555/llama-3-13k:latest",
        "label": "Fine-tuned — gpraneeth555/Llama-3-13k",
        "tool_calling": "legacy",
    },
    {
        "id": "qwen3.5:9b",
        "label": "Qwen 3.5 9B",
        # Assumed "native" (most current Qwen releases ship an Ollama template
        # with proper {{ .Tools }}/{{ .ToolCalls }} support) — NOT yet verified
        # against this specific tag the way the two entries above were (`ollama
        # show qwen3.5:9b --modelfile`). If it turns out to be legacy-style like
        # the fine-tune, flip this to "legacy" — legacy_tool_adapter.py handles
        # that branch already, nothing else needs to change.
        "tool_calling": "native",
    },
]
_BY_ID = {m["id"]: m for m in CATALOG}

# Fallback default if nothing has been toggled yet — keeps existing behavior
# (agent.py used to read this same env var directly) for anyone who hasn't
# touched the new toggle at all.
_DEFAULT_MODEL = os.environ.get("ORBIX_AGENT_MODEL", "llama3.1:8b")


def _default_state() -> dict:
    return {"active": _DEFAULT_MODEL}


def load_state() -> dict:
    state = _default_state()
    try:
        if _STATE_FILE.exists():
            saved = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            active = saved.get("active")
            if isinstance(active, str) and active:
                state["active"] = active
    except Exception as e:  # noqa: BLE001 — never let a bad state file break startup
        logger.error("Failed to read model_state.json (%s); using default", e)
    return state


def save_state(state: dict) -> None:
    with _lock:
        try:
            _STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to write model_state.json: %s", e)


def get_active() -> str:
    """The model id the agent should use on its next turn."""
    return load_state()["active"]


def set_active(model_id: str) -> dict:
    """
    Switch the active model. Accepts any non-empty id (not just catalog
    entries) so an ad-hoc `ORBIX_AGENT_MODEL`-style override still works —
    unknown ids just default to "native" tool-calling behavior (see
    tool_calling_mode below), same as the app's prior single-model behavior.
    """
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("model id is required")
    state = load_state()
    previous = state.get("active")
    state["active"] = model_id
    save_state(state)
    logger.info("Active agent model switched to %s", model_id)
    if previous and previous != model_id:
        _release_model(previous)
    return describe_active()


def _release_model(model_id: str) -> bool:
    """
    Evict a model from Ollama after switching away from it.

    The agent runs with keep_alive = -1 so a model stays resident indefinitely — the
    right trade within a session, since a cold load costs 100-200s on CPU. But nothing
    ever released it on a SWITCH, so both models ended up resident at once. Measured on
    the development machine: llama3.1:8b holds ~5.5 GB of 15.4 GB total, and loading the
    ~4.9 GB fine-tuned model on top drove free memory to ~0.6 GB and pushed the machine
    into swapping — which is what made the fine-tuned model appear to "run forever"
    rather than anything about the model itself. Unloading one model was measured to
    return free RAM from 0.73 GB to 6.97 GB.

    Best-effort by design: a failure here costs memory, never correctness, so it must
    never raise into a model switch the user asked for.
    """
    try:
        import subprocess
        subprocess.run(
            ["ollama", "stop", model_id],
            capture_output=True, timeout=120, stdin=subprocess.DEVNULL,
        )
        logger.info("Released previous model %s from memory", model_id)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not release previous model %s: %s", model_id, e)
        return False


def tool_calling_mode(model_id: str) -> str:
    """'native' (Ollama tools= param) or 'legacy' (text-parsed tool calls)."""
    entry = _BY_ID.get(model_id)
    return entry["tool_calling"] if entry else "native"


def list_models() -> list[dict]:
    """Catalog decorated with which one is currently active, for the UI."""
    active = get_active()
    out = []
    for m in CATALOG:
        out.append({**m, "active": m["id"] == active})
    if active not in _BY_ID:
        # a manually-set / env-overridden model not in the catalog — surface it
        # too so the UI doesn't silently show nothing as selected.
        out.append({"id": active, "label": active, "tool_calling": "native", "active": True})
    return out


def describe_active() -> dict:
    active = get_active()
    entry = _BY_ID.get(active, {"id": active, "label": active, "tool_calling": "native"})
    return {**entry, "active": True}
