"""
Routing logic for LangGraph — determines which module to call based on user query.

Adapts existing IntentClassifier with simple rule-based routing.
Three tiers:
  1. Rule-based regex patterns (instant, high confidence)
  2. Optional LLM fallback for ambiguous queries (low confidence)
  3. Default to general_chat
"""

import os
import re
import logging
from typing import Tuple
from .graph_state import OrbixState

logger = logging.getLogger(__name__)

# Rule-based intent detection patterns
_INTENT_PATTERNS = {
    "send_email": [
        r"send\s+(?:an?\s+)?email",
        r"email\s+(?:to\s+)?[\w\-]+@[\w\-]+\.\w+",
        # Removed: r"(?:tell|write|reply)\s+(?:to\s+)?\w+" — matched almost any
        # sentence with "tell"/"write"/"reply" + a word, with no requirement for
        # "email" or an address anywhere. Confirmed live: "did i tell u about the
        # new friend..." (a pure recall question) got tagged send_email. The two
        # patterns above already cover real send-email phrasing without this.
    ],
    "travel_planner": [
        r"plan\s+(?:a\s+)?(?:trip|travel|vacation)",
        r"(?:trip|travel|vacation)\s+to\s+\w+",
        r"where\s+should\s+i\s+(?:go|travel|visit)",
    ],
    "create_meeting": [
        r"(?:schedule|create|book)\s+(?:a\s+)?(?:meeting|call|hangout)",
        r"(?:meet|call)\s+(?:with\s+)?[\w\-]+@[\w\-]+\.\w+",
    ],
    "get_emails": [
        r"(?:show|get|retrieve)\s+(?:my\s+)?emails",
        r"(?:check|read)\s+(?:my\s+)?(?:inbox|mail)",
    ],
}


# Map intent → LangGraph module node.
# Gmail / Meet / calendar actions now run through the MCP agent (the "chat" module):
# the model plans and executes them as tool calls with reasoning/chaining, instead of
# a hardcoded action_node. Only travel keeps its own dedicated branch.
_MODULE_MAP = {
    "send_email": "chat",
    "create_meeting": "chat",
    "meeting_and_email": "chat",
    "schedule_meeting": "chat",
    "get_emails": "chat",
    "travel_planner": "travel",
    "general_chat": "chat",
}


def _regex_route(user_query: str):
    """Fallback rule-based detection. Returns (intent, confidence)."""
    ql = user_query.lower()
    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, ql, re.IGNORECASE):
                return intent, 0.95
    return "general_chat", 0.5


def route_query(state: OrbixState) -> OrbixState:
    """
    Route user query to the appropriate module.

    This only ever needs to decide ONE thing: travel vs. everything-else — every
    non-travel intent maps to "chat" in _MODULE_MAP regardless of its specific
    label (chat_node's agent loop does its own tool-selection independent of
    this classification; state["classification"]/state["intent"] are otherwise
    only echoed back to the frontend as a cosmetic status label — see
    PERFORMANCE_AUDIT.md item 3). So:

    Tier 1: regex rules (instant, no model call) — handles travel/email/meeting
            phrasing directly, and anything it doesn't recognize is, by the same
            logic, not a travel query either (the travel patterns are broad),
            so it's safe to resolve straight to "general_chat" without escalating.
    Tier 2: the separate fine-tuned classifier model — NOT called by default
            anymore. Calling a second model here used to evict the agent loop's
            already-warm llama3.1:8b context on literally every turn (confirmed:
            ~110s of pure prompt reprocessing per turn on this CPU-only setup,
            for a decision regex already made correctly). Kept only as an
            explicit opt-in (ORBIX_USE_LLM_ROUTER=1) for anyone who wants the
            extra nuance and can afford the cost.
    """
    user_query = state["user_query"]

    # Tier 1: regex — resolves the travel-vs-chat decision for the vast majority
    # of queries with zero LLM cost.
    intent, confidence = _regex_route(user_query)
    logger.info(f"Regex router → intent={intent} (confidence={confidence:.2f})")
    classification = {}

    # Tier 2 (opt-in only): consult the fine-tuned classifier for extra nuance.
    # Off by default — see docstring above for why.
    if os.environ.get("ORBIX_USE_LLM_ROUTER") == "1":
        try:
            from intent_workflow.intent_classifier import IntentClassifier
            classification = IntentClassifier.classify(user_query) or {}
            llm_intent = classification.get("intent")
            if llm_intent:
                intent = llm_intent
                confidence = float(classification.get("confidence", 0.9) or 0.9)
                logger.info(f"LLM classifier → intent={intent} (confidence={confidence:.2f})")
        except Exception as e:
            logger.warning(f"LLM classifier failed ({e}); keeping regex result")

    # The MCP agent reads the raw user query and decides which tools to call, so no
    # param normalization / intent-upgrade is needed here anymore — routing only picks
    # the branch (chat vs travel).
    module_name = _MODULE_MAP.get(intent, "chat")

    state["intent"] = intent
    state["confidence"] = confidence
    state["module_name"] = module_name
    state["classification"] = classification if isinstance(classification, dict) else {}
    state["step"] = f"route_query → {module_name}"

    logger.info(f"Routed query to {module_name} (intent={intent}, confidence={confidence:.2f})")
    return state


def determine_module(state: OrbixState) -> str:
    """
    Conditional edge function for LangGraph.
    Returns the next node name based on module_name in state.
    """
    module = state.get("module_name", "chat")
    node_map = {
        "chat": "chat_node",
        "travel": "travel_node",
    }
    return node_map.get(module, "chat_node")
