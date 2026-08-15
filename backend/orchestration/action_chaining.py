"""
Phase 4: Action Chaining — Multi-intent workflow support

Handles:
1. Follow-up actions after module completion
2. Multi-intent queries (e.g., "Plan trip to Paris AND email my team")
3. Context passing between modules
4. Sequential vs parallel execution
"""

import logging
from typing import Dict, List, Any, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ActionChain:
    """Represents a chain of actions to execute."""
    intents: List[str]  # e.g., ["travel_planner", "send_email"]
    contexts: Dict[str, Any]  # Shared context between intents
    execution_mode: Literal["sequential", "parallel"] = "sequential"


class ActionChainExecutor:
    """Orchestrates multi-intent workflows with context passing."""

    def __init__(self):
        self.chains = {}

    def detect_chain(self, user_query: str) -> ActionChain:
        """
        Detect multi-intent queries and build action chain.
        
        Examples:
        - "Plan trip to Paris and email my team" → [travel_planner, send_email]
        - "Send email and create meeting" → [send_email, create_meeting]
        - "Plan trip" → [travel_planner] (no chain)
        """
        intents = []
        execution_mode = "sequential"

        # Multi-intent detection using keyword patterns
        query_lower = user_query.lower()

        # Travel intentions
        if any(w in query_lower for w in ["trip", "travel", "vacation", "destination", "hotel", "flight"]):
            intents.append("travel_planner")

        # Email intentions
        if any(w in query_lower for w in ["send email", "email", "mail", "message"]):
            intents.append("send_email")

        # Meeting intentions — NOT "calendar"/"event": those just mean "put this on my
        # calendar," which is a create_calendar_event action, not a Google Meet. Confirmed
        # live: "email me the plan, add it to my calendar" was misclassified as a meeting
        # request and made the agent create an unrequested real Google Meet.
        if any(w in query_lower for w in ["create meeting", "meeting", "schedule a call",
                                          "video call", "gmeet", "google meet"]):
            intents.append("create_meeting")

        # Preference for execution mode
        if any(w in query_lower for w in ["and", "also", "then"]):
            execution_mode = "sequential"  # "and then" → sequential
        elif any(w in query_lower for w in ["with", "&", "plus"]):
            execution_mode = "sequential"  # "with" → can be sequential or parallel

        return ActionChain(
            intents=list(dict.fromkeys(intents)) or ["general_chat"],  # Remove duplicates
            contexts={},
            execution_mode=execution_mode
        )

    def build_context(self, query: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build shared context from previous module results.
        This context is passed to subsequent modules for better understanding.
        
        Example: If travel_planner returns {to_city: "Paris", itinerary: "..."},
                 send_email can reference this in the email body.
        """
        context = {
            "original_query": query,
            "travel": previous_results.get("travel", {}),
            "action": previous_results.get("action", {}),
            "chat": previous_results.get("chat", {}),
        }
        return context

    def route_chain(self, action_chain: ActionChain) -> List[str]:
        """
        Given an action chain, determine execution order.
        
        Rules:
        - Travel planner should run first (builds destination context)
        - Then send email / create meeting (can reference travel details)
        - General chat can run anytime
        """
        preferred_order = {
            "travel_planner": 1,
            "send_email": 2,
            "create_meeting": 2,
            "meeting_and_email": 2,
            "get_emails": 3,
            "general_chat": 4,
        }

        sorted_intents = sorted(
            action_chain.intents,
            key=lambda x: preferred_order.get(x, 99)
        )

        logger.info(f"Routing chain: {sorted_intents} (mode: {action_chain.execution_mode})")
        return sorted_intents

    def extract_parameters_for_chain(self, query: str, previous_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parameters for the next intent in the chain.
        Can use results from previous intents.
        
        Example: If travel result has {to_city: "Paris"},
                 use that to create smart email like "Here's your Paris itinerary..."
        """
        parameters = {}

        # If previous result was from travel, extract destination
        if "travel" in previous_result and previous_result["travel"]:
            travel_data = previous_result["travel"]
            destination = travel_data.get("entities", {}).get("to_city")
            if destination:
                parameters["destination"] = destination
                parameters["trip_context"] = travel_data.get("itinerary", "")

        return parameters


# Singleton executor
_chain_executor = None


def get_chain_executor() -> ActionChainExecutor:
    """Get or create the action chain executor."""
    global _chain_executor
    if _chain_executor is None:
        _chain_executor = ActionChainExecutor()
    return _chain_executor
