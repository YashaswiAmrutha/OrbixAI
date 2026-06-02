"""
OrbixAI Orchestrator — powered by the local fine-tuned Ollama model.

The fine-tuned model (gpraneeth555/Llama-3-13k) returns tool calls in the form:

    ### Tool Calls:  (or ### Output: / ### Action: / ### Tools:)
    [{"tool": "gmeet", "action": "create_meeting", "parameters": {...}},
     {"tool": "gmail", "action": "send_email",     "parameters": {...}}]

This module parses those tool calls and maps them to OrbixAI's internal
intent schema so the existing workflow pipeline stays untouched.

Fallback chain:  fine-tuned model → llama3.1:8b → safe default
"""

import json
import re
import logging

logger = logging.getLogger(__name__)

ORCHESTRATOR_MODEL = "gpraneeth555/Llama-3-13k:latest"
FALLBACK_MODEL     = "llama3.1:8b"

# ── GPU workaround: model must run on CPU (CUDA runtime issue on this machine) ─
_OLLAMA_OPTIONS = {"temperature": 0.1, "num_predict": 1024, "num_gpu": 0}
_CHAT_OPTIONS   = {"temperature": 0.7, "num_predict": 512,  "num_gpu": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fast-path extractor
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_extract(user_query: str) -> dict | None:
    """
    Instant intent extraction via regex + keyword matching.
    Runs before the model — covers the vast majority of clear commands.
    Returns None only when the query is genuinely ambiguous.
    """
    q      = user_query.lower()
    emails = re.findall(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', user_query)

    wants_meet   = any(w in q for w in ('meet', 'google meet', 'gmeet', 'video call', 'conference', 'schedule'))
    wants_email  = any(w in q for w in ('email', 'mail', 'send', 'message'))
    wants_travel = any(w in q for w in ('trip', 'travel', 'itinerary', 'plan a trip', 'visit', 'fly to', 'going to', 'holiday', 'vacation'))
    wants_inbox  = any(w in q for w in ('check email', 'read email', 'my inbox', 'show email', 'latest email', 'check my email', 'inbox'))

    # If none of the known intent signals present, hand off to model
    if not (wants_meet or wants_email or wants_travel or wants_inbox or emails):
        return None

    params: dict = {}
    if emails:
        params['attendee_email']  = emails[0]
        params['recipient_email'] = emails[0]

    # Meeting title from "about X" / "regarding X" / "for X" / "discuss X"
    title_match = re.search(
        r'(?:about|regarding|for|discuss(?:ing)?|on)\s+(?:our\s+)?([^,.\n]{3,60})',
        user_query, re.IGNORECASE
    )
    if title_match:
        params['event_title'] = title_match.group(1).strip().title()
    else:
        params.setdefault('event_title', 'Meeting')

    # Destination extraction for travel
    if wants_travel:
        dest_m = re.search(
            r'(?:to|trip to|travel to|visit|going to|fly to)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)',
            user_query
        )
        if dest_m:
            params['destination'] = dest_m.group(1)

    # Determine intent
    if wants_meet and (wants_email or emails):
        intent = 'meeting_and_email'
    elif wants_meet:
        intent = 'create_meeting'
    elif wants_email and emails:
        intent = 'send_email'
    elif wants_travel and params.get('destination'):
        intent = 'travel_planner'
    elif wants_inbox:
        intent = 'get_emails'
    else:
        return None  # Not confident enough — let model decide

    logger.info(f"[Orchestrator] Rule-based fast-path → intent={intent}")
    return {
        'intent':        intent,
        'confidence':    0.88,
        'parameters':    params,
        'email_content': {'subject': None, 'body': None},
        'travel_plan':   {},
        'explanation':   f'Rule-based extraction: {intent}',
        '_source':       'rule_based',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool-call parser
# ─────────────────────────────────────────────────────────────────────────────

# Tools that indicate the model wants to ask the user rather than act — ignore these
_NON_ACTIONABLE_TOOLS = frozenset({
    "user_interaction", "ask_clarification", "clarification",
    "confirm", "user_input", "ask_user", "weather_api",
})


def _extract_tool_calls(text: str) -> list:
    """
    Pull the JSON tool-call payload out of the model's raw output.
    Handles: JSON array [...], JSON object {...}, and plain-text tool names.
    Filters out non-actionable tool calls (user_interaction, ask_clarification, etc.).
    """
    raw_calls = []

    # Try JSON array first  (most common: ### Tool Calls: [...])
    arr = re.search(r'\[[\s\S]*?\]', text)
    if arr:
        try:
            parsed = json.loads(arr.group())
            if isinstance(parsed, list):
                raw_calls = parsed
        except json.JSONDecodeError:
            pass

    # Single JSON object  (### Action: {...})
    if not raw_calls:
        obj = re.search(r'\{[\s\S]*?\}', text)
        if obj:
            try:
                parsed = json.loads(obj.group())
                if isinstance(parsed, dict):
                    raw_calls = [parsed]
            except json.JSONDecodeError:
                pass

    # Plain-text tool reference  (### Tools:\ngmail.search_emails)
    if not raw_calls:
        lower = text.lower()
        if "search_emails" in lower or "gmail.search" in lower:
            raw_calls = [{"tool": "gmail", "action": "search_emails"}]

    # Filter non-actionable tool calls
    actionable = [
        tc for tc in raw_calls
        if str(tc.get("tool", "")).lower() not in _NON_ACTIONABLE_TOOLS
        and str(tc.get("action", "")).lower() not in _NON_ACTIONABLE_TOOLS
    ]
    return actionable


def _map_to_intent(tool_calls: list, raw_text: str) -> dict:
    """
    Convert fine-tuned-model tool calls → OrbixAI intent schema.

    Intent mapping:
      gmeet create_meeting  +  gmail send_email  →  meeting_and_email
      gmeet create_meeting  only                 →  create_meeting
      gmail send_email      only                 →  send_email
      search_flights / hotels / travel           →  travel_planner
      gmail search_emails / get_emails           →  get_emails
      (nothing matched)                          →  general_chat
    """
    has_meet   = False
    has_email  = False
    has_travel = False
    has_inbox  = False
    params     = {}
    email_subj = None
    email_body = None

    for tc in tool_calls:
        tool   = str(tc.get("tool",   "")).lower()
        action = str(tc.get("action", "")).lower()
        p      = tc.get("parameters", {}) or {}

        # ── detect intent signals ──────────────────────────────────────────
        if tool == "gmeet" or action in ("create_meeting", "schedule_meeting"):
            has_meet = True
        if tool == "gmail" and action in ("send_email", "send", "email"):
            has_email = True
        if tool in ("search_flights", "search_hotels", "flights", "hotels") \
                or action in ("search_flights", "search_hotels", "plan_trip"):
            has_travel = True
        if action in ("search_emails", "get_emails", "list_emails", "inbox"):
            has_inbox = True

        # ── extract parameters ─────────────────────────────────────────────
        # Attendees / recipients
        for key in ("attendees", "attendee"):
            if key in p:
                val = p[key]
                email = val[0] if isinstance(val, list) else val
                params["attendee_email"]  = email
                params["recipient_email"] = email

        for key in ("recipient", "recipients", "to"):
            if key in p:
                val = p[key]
                email = val[0] if isinstance(val, list) else val
                params.setdefault("recipient_email", email)
                params.setdefault("attendee_email",  email)

        # Meeting title
        for key in ("title", "event_title", "summary", "meeting_title"):
            if key in p:
                params.setdefault("event_title", p[key])

        # Description
        for key in ("description", "event_description", "details"):
            if key in p:
                params.setdefault("event_description", p[key])

        # Time
        for key in ("start_time", "time", "datetime", "date"):
            if key in p:
                params.setdefault("start_time", p[key])

        # Travel
        for key in ("origin", "from", "departure_city", "source"):
            if key in p:
                params.setdefault("origin", p[key])
        for key in ("destination", "to", "arrival_city"):
            if key in p:
                params.setdefault("destination", p[key])
                has_travel = True
        for key in ("departure_date", "depart", "start_date"):
            if key in p:
                params.setdefault("departure_date", p[key])
        for key in ("return_date", "return", "end_date"):
            if key in p:
                params.setdefault("return_date", p[key])

        # Email subject/body
        if "subject" in p:
            email_subj = p["subject"]
        for key in ("body", "message", "content", "body_keywords"):
            if key in p:
                val = p[key]
                if isinstance(val, list):
                    val = " ".join(val)
                email_body = val

    # ── determine final intent ─────────────────────────────────────────────
    if has_meet and has_email:
        intent = "meeting_and_email"
    elif has_meet:
        intent = "create_meeting"
    elif has_email:
        intent = "send_email"
    elif has_travel:
        intent = "travel_planner"
    elif has_inbox:
        intent = "get_emails"
    else:
        intent = "general_chat"

    # Default meeting title if missing
    if intent in ("create_meeting", "meeting_and_email", "schedule_meeting"):
        params.setdefault("event_title", "Meeting")

    return {
        "intent":        intent,
        "confidence":    0.92,
        "parameters":    params,
        "email_content": {"subject": email_subj, "body": email_body},
        "travel_plan":   {},
        "explanation":   f"Parsed from fine-tuned model tool calls ({intent})",
        "_source":       "ollama_finetuned",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (same signatures as before — zero changes needed in callers)
# ─────────────────────────────────────────────────────────────────────────────

def orchestrate(user_query: str) -> dict:
    """
    Classify intent and extract parameters from a user query.

    PRIMARY:   fine-tuned gpraneeth555/Llama-3-13k — its tool-call JSON drives the automation.
    FALLBACK:  rule-based regex — only when the model is unavailable or returns non-actionable calls.
    """
    import ollama

    raw_text = None
    source   = "ollama_finetuned"

    # ── 1. Fine-tuned model (primary — always run this first) ─────────────
    try:
        resp     = ollama.chat(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "user", "content": user_query}],
            options=_OLLAMA_OPTIONS,
        )
        raw_text = resp["message"]["content"].strip()
        logger.info(f"[Orchestrator] {ORCHESTRATOR_MODEL} responded")
    except Exception as e:
        logger.warning(f"[Orchestrator] Fine-tuned model unavailable ({e})")

    # ── 2. Parse model's tool calls ────────────────────────────────────────
    if raw_text:
        tool_calls = _extract_tool_calls(raw_text)
        if tool_calls:
            result = _map_to_intent(tool_calls, raw_text)
            result["_source"] = source
            logger.info(
                f"[Orchestrator] Tool-call intent={result['intent']} "
                f"confidence={result['confidence']}"
            )
            return result

        # Model responded but returned no actionable tool calls (e.g. ask_clarification)
        # → fall through to rule-based so the action still executes
        logger.info("[Orchestrator] Model returned no actionable tools — trying rule-based")

    # ── 3. Rule-based fallback ─────────────────────────────────────────────
    fast = _rule_based_extract(user_query)
    if fast and fast["intent"] != "general_chat":
        logger.info(f"[Orchestrator] Rule-based fallback → intent={fast['intent']}")
        return fast

    # ── 4. General chat ────────────────────────────────────────────────────
    return {
        "intent":        "general_chat",
        "confidence":    0.5,
        "parameters":    {},
        "email_content": {"subject": None, "body": None},
        "travel_plan":   {},
        "explanation":   "No actionable intent detected",
        "_source":       source or "fallback",
    }


def chat_response(user_query: str) -> str:
    """
    Generate a plain-text conversational reply.
    Uses fine-tuned model first, falls back to llama3.1:8b.
    """
    import ollama

    for model in (ORCHESTRATOR_MODEL, FALLBACK_MODEL):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": user_query}],
                options=_CHAT_OPTIONS,
            )
            text = resp["message"]["content"].strip()
            if text:
                logger.info(f"[ChatResponse] Used {model}")
                # Strip any accidental tool-call output
                text = re.sub(r'###\s*(Tool Calls?|Output|Action|Tools?):?.*', '', text,
                              flags=re.IGNORECASE | re.DOTALL).strip()
                return text or "I'm here to help — what would you like to do?"
        except Exception as e:
            logger.warning(f"[ChatResponse] {model} failed: {e}")

    return (
        "Ollama is not running. Start it with `ollama serve` in a terminal "
        "and make sure the model is pulled."
    )


def _default_result(user_query: str) -> dict:
    return {
        "intent":        "general_chat",
        "confidence":    0.0,
        "parameters":    {},
        "email_content": {"subject": None, "body": None},
        "travel_plan":   {},
        "explanation":   "Orchestration failed — defaulting to general chat",
        "_source":       "fallback",
    }
