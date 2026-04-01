"""
HuggingFace Inference Client for gpraneeth555/llama-3-13k
Used as the primary orchestrator: classifies intent, extracts parameters,
generates email content, and plans travel — all in one JSON response.
Falls back to Ollama llama3.1:8b if the HF API is unavailable.
"""

import os
import json
import re
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)

HF_MODEL_ID = "gpraneeth555/llama-3-13k"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# ──────────────────────────────────────────────
# Orchestrator system prompt
# ──────────────────────────────────────────────
ORCHESTRATOR_SYSTEM_PROMPT = """You are OrbixAI's intelligent task orchestrator. \
Analyze the user message below and return ONLY a single valid JSON object — no prose, no markdown fences.

AVAILABLE INTENTS:
  send_email         - User wants to send an email to someone
  create_meeting     - User wants to create a Google Meet (no email)
  meeting_and_email  - User wants to create a meeting AND email the invite
  schedule_meeting   - User wants to schedule a meeting at a specific time
  get_emails         - User wants to read / check their inbox
  travel_planner     - User wants a travel itinerary or recommendations
  general_chat       - Everything else (questions, conversation, etc.)

RULES:
1. For send_email / meeting_and_email / schedule_meeting — ALWAYS populate email_content.subject and email_content.body with complete, professional text.
2. For create_meeting / schedule_meeting — populate parameters.event_title and parameters.event_description.
3. For travel_planner — populate travel_plan with a full day-by-day itinerary, recommendations, tips, budget_estimate, and best_time.
4. Extract every entity mentioned (emails, names, dates, places, preferences).
5. Omit fields that are genuinely not present — use null, not empty strings.
6. confidence is a float 0.0–1.0 reflecting how certain you are of the classification.

JSON SCHEMA (return exactly this shape):
{
  "intent": "<one of the above intents>",
  "confidence": 0.95,
  "parameters": {
    "recipient_email": null,
    "attendee_email": null,
    "recipient_name": null,
    "event_title": null,
    "event_description": null,
    "start_time": null,
    "max_results": null,
    "origin": null,
    "destination": null,
    "departure_date": null,
    "return_date": null,
    "travelers": null,
    "preferences": []
  },
  "email_content": {
    "subject": null,
    "body": null
  },
  "travel_plan": {
    "itinerary": null,
    "recommendations": [],
    "tips": null,
    "budget_estimate": null,
    "best_time": null
  },
  "explanation": "<one sentence rationale>"
}"""


def _get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _extract_json(text: str) -> dict:
    """Extract the first JSON object found in a string."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # Find first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON object found in model output")


def call_hf_model(prompt: str, max_new_tokens: int = 1024) -> str:
    """
    Call gpraneeth555/llama-3-13k via the HuggingFace Inference API.
    Raises on failure so the caller can fall back to Ollama.
    """
    token = _get_hf_token()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.15,
            "do_sample": False,
            "return_full_text": False,
            "stop": ["</s>", "[/INST]"]
        }
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)
    response.raise_for_status()

    result = response.json()
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "")
    if isinstance(result, dict):
        if "error" in result:
            raise RuntimeError(f"HF API error: {result['error']}")
        return result.get("generated_text", "")
    return ""


def _build_orchestrator_prompt(user_query: str) -> str:
    """Wrap the user query with the orchestrator system prompt."""
    return (
        f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n"
        f"USER MESSAGE: \"{user_query}\"\n\n"
        "JSON RESPONSE:"
    )


def _ollama_fallback(user_query: str) -> str:
    """Fall back to Ollama llama3.1:8b with the same orchestrator prompt."""
    import ollama
    prompt = _build_orchestrator_prompt(user_query)
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


def orchestrate(user_query: str) -> dict:
    """
    Main entry point. Sends the user query to gpraneeth555/llama-3-13k and
    returns a parsed dict with intent, parameters, email_content, travel_plan.
    Falls back to Ollama on any HF failure.
    """
    prompt = _build_orchestrator_prompt(user_query)

    raw_text = None
    source = "hf"

    try:
        raw_text = call_hf_model(prompt)
        logger.info("[Orchestrator] gpraneeth555/llama-3-13k responded")
    except Exception as e:
        logger.warning(f"[Orchestrator] HF model unavailable ({e}), falling back to Ollama")
        source = "ollama"
        try:
            raw_text = _ollama_fallback(user_query)
        except Exception as e2:
            logger.error(f"[Orchestrator] Ollama fallback also failed: {e2}")
            return _default_result(user_query)

    try:
        result = _extract_json(raw_text)
        result["_source"] = source
        logger.info(f"[Orchestrator] intent={result.get('intent')} confidence={result.get('confidence')} source={source}")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"[Orchestrator] JSON parse failed: {e}\nRaw: {raw_text[:300]}")
        return _default_result(user_query)


def _default_result(user_query: str) -> dict:
    """Safe fallback when all LLM calls fail."""
    return {
        "intent": "general_chat",
        "confidence": 0.0,
        "parameters": {},
        "email_content": {"subject": None, "body": None},
        "travel_plan": {},
        "explanation": "Orchestration failed — defaulting to general chat",
        "_source": "fallback"
    }
