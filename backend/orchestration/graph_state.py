"""
OrbixState — TypedDict definition for LangGraph workflow state.

Contains:
  - Input: user_query, session_id
  - Routing: intent, confidence, module_name
  - Processing: module_output, retrieved_facts, transcript
  - Action chaining: follow_up_actions
  - Async tasks: extraction_tasks[], cache_tasks[]
"""

from typing import TypedDict, List, Dict, Any, Optional


class OrbixState(TypedDict, total=False):
    """
    State container for LangGraph workflow.
    All fields are optional (total=False) to allow incremental population.
    """
    
    # === Input ===
    user_query: str                    # Raw user input
    session_id: str                    # Unique session identifier
    
    # === Routing ===
    intent: str                        # "general_chat" | "travel_planning" | "action" | "combined"
    confidence: float                  # 0.0 to 1.0
    module_name: str                   # Which module to call: "chat" | "travel" | "action"
    classification: Dict[str, Any]     # Full IntentClassifier output (params/email_content/travel_plan)
    
    # === Context Retrieval ===
    transcript: List[Dict[str, str]]   # SQLite conversation history [{role, text}, ...]
    retrieved_facts: Dict[str, Any]    # Neo4j facts: {recent_trips, contacts, preferences}
    
    # === Processing ===
    module_output: Dict[str, Any]      # {"response": str, "data": any, "module": str}
    calendar_events: List[Dict[str, Any]]  # Events to auto-sync into the frontend calendar
    todo_items: List[Dict[str, Any]]       # Todo items to auto-add in the frontend
    
    # === Action Chaining (Phase 4) ===
    follow_up_actions: List[str]       # Actions to chain after module completes
    execution_mode: str                # "sequential" | "parallel"
    follow_up_context: Dict[str, Any]  # Shared context for follow-up actions
    
    # === Async Task Queues (Phase 3) ===
    extraction_tasks: List[Dict[str, Any]]  # [{type: "extract_from_turn" | "extract_trip" | "extract_action", ...}]
    cache_tasks: List[Dict[str, Any]]       # [{type: "travel_plan", trip_id: ..., itinerary_json: ...}]
    
    # === Error Handling ===
    errors: List[Dict[str, str]]       # [{task_type: str, error: str}]
    
    # === Metadata ===
    created_at: int                    # Timestamp when state created
    updated_at: int                    # Timestamp of last update
    step: str                          # Current step name (for debugging)


def new_state(user_query: str, session_id: str) -> OrbixState:
    """Create a fresh state for a new query."""
    import time
    now = int(time.time())
    return {
        "user_query": user_query,
        "session_id": session_id,
        "intent": "",
        "confidence": 0.0,
        "module_name": "",
        "classification": {},
        "transcript": [],
        "retrieved_facts": {},
        "module_output": {},
        "calendar_events": [],
        "todo_items": [],
        "follow_up_actions": [],
        "execution_mode": "sequential",
        "follow_up_context": {},
        "extraction_tasks": [],
        "cache_tasks": [],
        "errors": [],
        "created_at": now,
        "updated_at": now,
        "step": "init",
    }
