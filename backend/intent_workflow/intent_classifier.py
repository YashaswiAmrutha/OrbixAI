"""
Intent Classifier - Detects user intent and determines the workflow
Uses LLM to understand user queries and classify them into actionable intents
"""

import json
import logging
from typing import Dict, List, Any
from llm.ollama_client import generate_response

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Classifies user queries into structured intents with required parameters"""
    
    # Define all possible intents
    INTENTS = {
        "send_email": {
            "description": "Send an email with optional LLM-generated content",
            "required_fields": ["recipient_email"],
            "optional_fields": ["subject", "body", "recipient_name"]
        },
        "create_meeting": {
            "description": "Create a Google Meet without sending email",
            "required_fields": ["event_title"],
            "optional_fields": ["event_description", "attendee_email"]
        },
        "meeting_and_email": {
            "description": "Create Google Meet and send invitation email",
            "required_fields": ["attendee_email", "event_title"],
            "optional_fields": ["event_description", "subject"]
        },
        "schedule_meeting": {
            "description": "Schedule a meeting with specific time",
            "required_fields": ["attendee_email", "event_title"],
            "optional_fields": ["start_time", "end_time", "event_description"]
        },
        "get_emails": {
            "description": "Retrieve recent emails",
            "required_fields": [],
            "optional_fields": ["max_results"]
        },
        "general_chat": {
            "description": "General conversation or query (not email/meeting related)",
            "required_fields": [],
            "optional_fields": []
        }
    }
    
    @staticmethod
    def classify(user_query: str) -> Dict[str, Any]:
        """
        Classify user query and extract relevant parameters
        
        Args:
            user_query (str): The user's query/message
            
        Returns:
            Dict with keys:
                - intent: str (one of INTENTS.keys())
                - confidence: float (0-1)
                - parameters: Dict (extracted parameters)
                - explanation: str (why this intent was chosen)
        """
        
        try:
            # Create a prompt for the LLM to classify the intent
            classification_prompt = f"""Analyze this user query and classify it into one of these intents:

AVAILABLE INTENTS:
- send_email: Send an email (extract: recipient_email, subject, body content)
- create_meeting: Create Google Meet only (extract: event_title, description)
- meeting_and_email: Create meeting AND send email with invite link (extract: attendee_email, event_title, subject)
- schedule_meeting: Schedule meeting with time (extract: attendee_email, event_title, start_time)
- get_emails: Retrieve emails (extract: max_results if specified)
- general_chat: General conversation/question

USER QUERY: "{user_query}"

RESPOND IN JSON FORMAT ONLY:
{{
    "intent": "<one of the above>",
    "confidence": <0.0-1.0>,
    "parameters": {{
        "recipient_email": "<email if mentioned>",
        "attendee_email": "<email if mentioned>",
        "event_title": "<title if mentioned>",
        "subject": "<email subject if mentioned>",
        "body": "<email body/content if mentioned>",
        "event_description": "<description if mentioned>",
        "start_time": "<time if mentioned>",
        "max_results": <number if mentioned>,
        "recipient_name": "<contact name if mentioned>"
    }},
    "explanation": "<brief reason for this classification>"
}}

Return ONLY valid JSON, no other text."""
            
            response = generate_response(classification_prompt)
            
            # Parse the JSON response
            # Clean up response if it has markdown code blocks
            response_clean = response.strip()
            if response_clean.startswith("```"):
                response_clean = response_clean.split("```")[1]
                if response_clean.startswith("json"):
                    response_clean = response_clean[4:]
                response_clean = response_clean.strip()
            
            result = json.loads(response_clean)
            
            # Validate intent
            if result.get("intent") not in IntentClassifier.INTENTS:
                logger.warning(f"Invalid intent returned: {result.get('intent')}, defaulting to general_chat")
                result["intent"] = "general_chat"
                result["confidence"] = 0.5
            
            # Filter out None/empty parameters
            result["parameters"] = {
                k: v for k, v in result.get("parameters", {}).items() 
                if v is not None and v != ""
            }
            
            logger.info(f"Classified query as '{result['intent']}' (confidence: {result.get('confidence', 0)})")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {
                "intent": "general_chat",
                "confidence": 0.3,
                "parameters": {},
                "explanation": "Failed to classify - defaulting to general chat"
            }
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return {
                "intent": "general_chat",
                "confidence": 0.3,
                "parameters": {},
                "explanation": f"Classification error: {str(e)}"
            }
    
    @staticmethod
    def get_required_fields(intent: str) -> List[str]:
        """Get required fields for an intent"""
        return IntentClassifier.INTENTS.get(intent, {}).get("required_fields", [])
    
    @staticmethod
    def validate_parameters(intent: str, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate if required parameters are present for an intent
        
        Returns:
            (is_valid, error_message)
        """
        required = IntentClassifier.get_required_fields(intent)
        missing = [f for f in required if not parameters.get(f)]
        
        if missing:
            return False, f"Missing required fields for '{intent}': {', '.join(missing)}"
        
        return True, ""
