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


def route_query(state: OrbixState) -> OrbixState:
    """
    Route user query to the appropriate module.
    
    Returns state with:
      - intent: the detected intent name
      - confidence: 0.0-1.0 (how confident we are)
      - module_name: "chat" | "travel" | "action"
    
    Routing logic:
      1. Rule-based regex patterns (fast, high confidence)
      2. If no match, default to "general_chat"
      3. (Phase 2 option: LLM fallback for ambiguous queries)
    """
    user_query = state["user_query"].lower()
    
    # Tier 1: Rule-based detection
    best_intent = None
    best_confidence = 0.0
    
    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, user_query, re.IGNORECASE):
                # Rule-based hit → high confidence
                best_intent = intent
                best_confidence = 0.95
                break
        
        if best_confidence > 0.9:
            break
    
    # If no rule matches, default to general_chat
    if not best_intent:
        best_intent = "general_chat"
        best_confidence = 0.5
    
    # Map intent to module
    module_map = {
        "send_email": "action",
        "create_meeting": "action",
        "meeting_and_email": "action",
        "schedule_meeting": "action",
        "get_emails": "action",
        "travel_planner": "travel",
        "general_chat": "chat",
    }
    
    module_name = module_map.get(best_intent, "chat")
    
    state["intent"] = best_intent
    state["confidence"] = best_confidence
    state["module_name"] = module_name
    state["step"] = f"route_query → {module_name}"
    
    logger.info(
        f"Routed query to {module_name} (intent={best_intent}, confidence={best_confidence:.2f})"
    )
    
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
