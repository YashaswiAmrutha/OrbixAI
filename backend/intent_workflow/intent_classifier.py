"""
Intent Classifier - powered by gpraneeth555/llama-3-13k via the OrbixAI orchestrator.
Single-shot: classifies intent, extracts all parameters, generates email content,
and builds travel plans — all from one model call.
"""

import logging
from typing import Dict, List, Any
from llm.hf_client import orchestrate

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classifies user queries and extracts all actionable data in one pass."""

    INTENTS = {
        "send_email": {
            "description": "Send an email with AI-generated subject and body",
            "required_fields": ["recipient_email"],
            "optional_fields": ["subject", "body", "recipient_name"]
        },
        "create_meeting": {
            "description": "Create a Google Meet without sending an email",
            "required_fields": ["event_title"],
            "optional_fields": ["event_description", "attendee_email"]
        },
        "meeting_and_email": {
            "description": "Create a Google Meet and send an invitation email",
            "required_fields": ["attendee_email", "event_title"],
            "optional_fields": ["event_description", "subject", "body"]
        },
        "schedule_meeting": {
            "description": "Schedule a meeting at a specific time",
            "required_fields": ["attendee_email", "event_title"],
            "optional_fields": ["start_time", "end_time", "event_description"]
        },
        "get_emails": {
            "description": "Retrieve recent emails from inbox",
            "required_fields": [],
            "optional_fields": ["max_results"]
        },
        "travel_planner": {
            "description": "Plan a trip — generate itinerary, recommendations, budget",
            "required_fields": ["destination"],
            "optional_fields": [
                "origin", "departure_date", "return_date",
                "travelers", "preferences"
            ]
        },
        "general_chat": {
            "description": "General conversation or question",
            "required_fields": [],
            "optional_fields": []
        }
    }

    @staticmethod
    def classify(user_query: str) -> Dict[str, Any]:
        """
        Classify a user query using gpraneeth555/llama-3-13k.

        Returns a dict with:
          intent, confidence, parameters, email_content, travel_plan, explanation
        """
        result = orchestrate(user_query)

        # Validate / normalise intent
        intent = result.get("intent", "general_chat")
        if intent not in IntentClassifier.INTENTS:
            logger.warning(f"Unknown intent '{intent}', defaulting to general_chat")
            result["intent"] = "general_chat"
            result["confidence"] = 0.4

        # Clean up parameters — remove nulls/empty
        raw_params = result.get("parameters", {})
        if isinstance(raw_params, dict):
            result["parameters"] = {
                k: v for k, v in raw_params.items()
                if v is not None and v != "" and v != []
            }
        else:
            result["parameters"] = {}

        # Ensure email_content is always present
        if "email_content" not in result or not isinstance(result["email_content"], dict):
            result["email_content"] = {"subject": None, "body": None}

        # Ensure travel_plan is always present
        if "travel_plan" not in result or not isinstance(result["travel_plan"], dict):
            result["travel_plan"] = {}

        logger.info(
            f"[Classifier] intent={result['intent']} "
            f"confidence={result.get('confidence', 0):.2f} "
            f"source={result.get('_source', '?')}"
        )
        return result

    @staticmethod
    def get_required_fields(intent: str) -> List[str]:
        return IntentClassifier.INTENTS.get(intent, {}).get("required_fields", [])

    @staticmethod
    def validate_parameters(intent: str, parameters: Dict[str, Any]):
        required = IntentClassifier.get_required_fields(intent)
        missing = [f for f in required if not parameters.get(f)]
        if missing:
            return False, f"Missing required fields for '{intent}': {', '.join(missing)}"
        return True, ""
