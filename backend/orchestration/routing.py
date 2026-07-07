"""
Routing logic for LangGraph — determines which module to call based on user query.

Adapts existing IntentClassifier with simple rule-based routing.
Three tiers:
  1. Rule-based regex patterns (instant, high confidence)
  2. Optional LLM fallback for ambiguous queries (low confidence)
  3. Default to general_chat
"""

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
        r"(?:tell|write|reply)\s+(?:to\s+)?\w+",
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


# Map intent → LangGraph module node
_MODULE_MAP = {
    "send_email": "action",
    "create_meeting": "action",
    "meeting_and_email": "action",
    "schedule_meeting": "action",
    "get_emails": "action",
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

    Tier 1: fine-tuned IntentClassifier (LLM) — same brain the legacy /chat path
            uses, so routing quality matches. Its full output (parameters,
            email_content, travel_plan) is stashed in state["classification"]
            so action_node can reuse it without a second LLM call.
    Tier 2: regex rules as a fast fallback when the LLM is unavailable.
    """
    user_query = state["user_query"]

    intent = None
    confidence = 0.0
    classification = {}

    # Tier 1: LLM classifier
    try:
        from intent_workflow.intent_classifier import IntentClassifier
        classification = IntentClassifier.classify(user_query) or {}
        intent = classification.get("intent")
        confidence = float(classification.get("confidence", 0.9) or 0.9)
        if intent:
            logger.info(f"LLM classifier → intent={intent} (confidence={confidence:.2f})")
    except Exception as e:
        logger.warning(f"LLM classifier failed ({e}); falling back to regex routing")

    # Tier 2: regex fallback
    if not intent:
        intent, confidence = _regex_route(user_query)
        logger.info(f"Regex router → intent={intent} (confidence={confidence:.2f})")

    # Normalize params so action_node has both recipient_email / attendee_email
    params = classification.get("parameters", {}) if isinstance(classification, dict) else {}
    if "attendee_email" in params and "recipient_email" not in params:
        params["recipient_email"] = params["attendee_email"]
    elif "recipient_email" in params and "attendee_email" not in params:
        params["attendee_email"] = params["recipient_email"]

    # Auto-upgrade create_meeting → meeting_and_email when an email is present
    if intent == "create_meeting" and (params.get("attendee_email") or params.get("recipient_email")):
        intent = "meeting_and_email"
        if isinstance(classification, dict):
            classification["intent"] = intent

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
        "action": "action_node",
    }
    return node_map.get(module, "chat_node")
