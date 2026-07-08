"""
OrbixAI LangGraph Workflow — Phase 2 (Module Integration)

Implements the core workflow graph:
  1. route_query() — determine which module to call
  2. prepare_context() — fetch from SQLite + Neo4j
  3. Module nodes (chat_node, travel_node, action_node) — integrated with real logic
  4. format_response() — prepare output
  5. background_tasks() — async extraction & caching (Phase 3)

Phase 2: Real module integration with agent, travel planner, and action executor.
"""

import logging
import re
import time

from langgraph.graph import StateGraph, START, END

from .graph_state import OrbixState, new_state
from .routing import route_query, determine_module
from graph.working_memory import add_message, get_messages, start_session

logger = logging.getLogger(__name__)


# ============================================================================ #
# Progress streaming
# ---------------------------------------------------------------------------- #
# Nodes emit short human-readable steps; the API layer registers a callback per
# session_id and forwards them to the frontend as SSE "thinking" events.
# ============================================================================ #

_PROGRESS_EMITTERS = {}


def register_progress_emitter(session_id, fn):
    """Register a callback(str) that receives progress steps for a session."""
    if session_id and callable(fn):
        _PROGRESS_EMITTERS[session_id] = fn


def unregister_progress_emitter(session_id):
    _PROGRESS_EMITTERS.pop(session_id, None)


def _emit(state, msg):
    """Emit a progress step to the registered listener (if any) + the log."""
    fn = _PROGRESS_EMITTERS.get(state.get("session_id"))
    if fn:
        try:
            fn(msg)
        except Exception:
            pass
    logger.info("[workflow] %s", msg)


def _travel_calendar_events(itinerary: str, destination: str, departure_date: str = None) -> list:
    """Parse a day-wise itinerary into calendar event dicts (mirrors main.py)."""
    from datetime import datetime, timedelta
    events = []
    base_date = None
    if departure_date:
        try:
            base_date = datetime.strptime(departure_date, "%Y-%m-%d")
        except Exception:
            pass
    if not base_date:
        base_date = datetime.utcnow()

    pattern = re.compile(r'day\s+(\d+)[:\s\-–]+([^\n]+)', re.IGNORECASE)
    matches = pattern.findall(itinerary or "")
    if matches:
        for day_str, title in matches:
            day_num = int(day_str)
            event_date = base_date + timedelta(days=day_num - 1)
            events.append({
                "title": f"Day {day_num}: {title.strip()[:70]}",
                "date": event_date.strftime("%Y-%m-%d"),
                "time": "",
                "description": f"Trip to {destination}",
                "type": "travel",
                "source": "ai_travel",
            })
    else:
        events.append({
            "title": f"Trip to {destination}",
            "date": base_date.strftime("%Y-%m-%d"),
            "time": "",
            "description": (itinerary or "")[:400],
            "type": "travel",
            "source": "ai_travel",
        })
    return events


def _calendar_events_from_trace(trace: list) -> tuple[list, list]:
    """
    Turn calendar-affecting agent tool calls into (calendar_events, todo_items) so the
    frontend can auto-sync them — mirrors what action_node/travel_node used to surface.

    - create_calendar_event returns the stored event dict directly → use as-is.
    - create_meet returns {meet_link, event_id, event_title, attendee_email}; the gsuite
      tool doesn't add it to the local calendar, so synthesize a `meeting` event here.
    """
    from datetime import datetime
    cal_events, todos = [], []
    for t in trace or []:
        name = t.get("tool")
        res = t.get("result")
        if not isinstance(res, dict) or res.get("error"):
            continue
        if name == "create_calendar_event" and res.get("id"):
            cal_events.append(res)
            todos.append({"text": f"Follow up: {res.get('title', 'event')}",
                          "source": "ai_agent"})
        elif name == "create_meet" and res.get("meet_link"):
            title = res.get("event_title", "Meeting")
            cal_events.append({
                "title": title,
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "time": "",
                "description": (f"Attendee: {res.get('attendee_email', 'N/A')}\n"
                                f"Meet link: {res.get('meet_link')}"),
                "type": "meeting",
                "source": "ai_meeting",
            })
            todos.append({"text": f"Prepare for: {title}", "source": "ai_meeting"})
        elif name == "add_task" and res.get("added") and res.get("text"):
            # The agent added a to-do via the tasks MCP server — mirror it into the
            # frontend To-Do panel so the UI stays in sync with what it created.
            todos.append({"text": res["text"], "source": "ai_agent"})
    return cal_events, todos


# ============================================================================ #
# Node Implementations
# ============================================================================ #

def prepare_context(state: OrbixState) -> OrbixState:
    """
    Fetch context from both SQLite and Neo4j before passing to module.
    
    Phase 2: Fetch actual data from databases.
    Phase 4: Check for multi-intent queries and chain actions
    Phase 5: Retrieve user profile, trips, contacts, preferences from Neo4j
    """
    session_id = state["session_id"]
    user_query = state["user_query"]

    # Ensure the SQLite session row exists before any node writes a message.
    # travel_node / action_node call add_message() directly, and messages has a
    # FOREIGN KEY to sessions(id) — a client-supplied session_id that was never
    # registered via start_session() would otherwise crash the write (and the
    # node's except block would discard an already-computed result).
    try:
        start_session(session_id)
    except Exception as e:
        logger.error(f"Failed to ensure session {session_id}: {e}")

    try:
        # Tier 1: Fetch SQLite transcript (recent messages for context)
        transcript = get_messages(session_id, limit=20)
        state["transcript"] = [
            {"role": m["role"], "text": m["text"]} 
            for m in transcript
        ]
        logger.info(f"Retrieved {len(transcript)} messages from SQLite for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to retrieve transcript: {e}")
        state["transcript"] = []
    
    # Tier 2: Fetch Neo4j facts (Phase 5 - Real Data)
    try:
        from orchestration.memory_integration import get_memory_integration
        memory = get_memory_integration()
        
        # Fetch user profile, recent trips, contacts, preferences
        recent_trips = memory.get_recent_trips(limit=3)
        contacts = memory.get_contacts(limit=10)
        preferences = memory.get_preferences()
        
        state["retrieved_facts"] = {
            "recent_trips": recent_trips,
            "contacts": contacts,
            "preferences": preferences,
        }
        logger.info(f"Retrieved Neo4j context: {len(recent_trips)} trips, {len(contacts)} contacts")
    except Exception as e:
        logger.error(f"Failed to fetch Neo4j context: {e}")
        state["retrieved_facts"] = {"recent_trips": [], "contacts": [], "preferences": {}}
    
    # Tier 3: Detect multi-intent queries (Phase 4 - Action Chaining)
    try:
        from orchestration.action_chaining import get_chain_executor
        chain_executor = get_chain_executor()
        
        action_chain = chain_executor.detect_chain(user_query)
        if len(action_chain.intents) > 1:
            logger.info(f"Multi-intent detected: {action_chain.intents}")
            state["follow_up_actions"] = chain_executor.route_chain(action_chain)
            state["execution_mode"] = action_chain.execution_mode
        else:
            state["follow_up_actions"] = []
            state["execution_mode"] = "sequential"
    except Exception as e:
        logger.error(f"Failed to detect action chains: {e}")
        state["follow_up_actions"] = []
        state["execution_mode"] = "sequential"
    
    state["step"] = "prepare_context"
    return state


async def chat_node(state: OrbixState) -> OrbixState:
    """
    General chat module — integrates with agent.py's run_turn().

    Calls the real memory-aware agent loop (MCP tools + Neo4j recall). This is an
    async node so we can simply await run_turn() instead of nesting event loops.
    """
    session_id = state["session_id"]
    user_query = state["user_query"]

    try:
        _emit(state, "Thinking…")
        logger.info(f"Chat node: calling agent.run_turn() for '{user_query[:50]}...'")

        from mcp_host.agent import run_turn

        # Stream the agent's tool steps into the thinking bubble as they happen.
        # run_turn awaits on_event, so it must be a coroutine.
        async def _on_event(ev):
            step = ev.get("step") if isinstance(ev, dict) else None
            if step:
                _emit(state, step)

        result = await run_turn(user_query, session_id=session_id, on_event=_on_event)

        response = result.get("reply", "I couldn't generate a response")
        trace = result.get("trace", [])
        # run_turn may mint a session_id when none was supplied — keep it
        state["session_id"] = result.get("session_id", session_id)

        # Surface any calendar-affecting actions the agent took (create_meet /
        # create_calendar_event) so the frontend's calendar/todo auto-sync keeps
        # working now that actions run inside the agent instead of action_node.
        cal_events, todos = _calendar_events_from_trace(trace)
        if cal_events:
            state["calendar_events"] = cal_events
        if todos:
            state["todo_items"] = todos

        state["module_output"] = {
            "response": response,
            "module": "chat",
            "data": {"agent_response": response, "trace_steps": len(trace)}
        }

        # Enqueue extraction from the response (agent already persists the turn)
        state["extraction_tasks"].append({
            "type": "extract_from_turn",
            "user_text": user_query,
            "assistant_text": response,
            "session_id": state["session_id"],
        })

        logger.info(f"Chat node completed successfully with {len(trace)} tool steps")

    except Exception as e:
        logger.error(f"Chat node error: {e}", exc_info=True)
        state["module_output"] = {
            "response": f"I encountered an error: {str(e)}. Please try again.",
            "module": "chat",
            "data": {"error": str(e)}
        }
        state["errors"].append({
            "step": "chat_node",
            "error": str(e),
            "timestamp": time.time()
        })
    
    state["step"] = "chat_node"
    return state


async def travel_node(state: OrbixState) -> OrbixState:
    """
    Travel planner module — integrates with travel_planner.py.

    Async node: awaits the real plan_trip() pipeline (MCP travel server → Amadeus
    flights/hotels, OSM attractions, LLM itinerary), then produces a rich reply
    plus calendar events + todos for the frontend to auto-sync.
    """
    session_id = state["session_id"]
    user_query = state["user_query"]

    try:
        logger.info(f"Travel node: planning trip for '{user_query[:50]}...'")
        _emit(state, "Planning your trip…")

        from google_service.travel_planner import plan_trip

        def emit_progress(msg):
            _emit(state, msg)

        # plan_trip is a coroutine — await it directly on the event loop.
        result = await plan_trip(query=user_query, emit=emit_progress)

        # Check for errors in the result
        if "error" in result:
            raise ValueError(result["error"])

        # Format response for user
        entities = result.get("entities", {})
        itinerary = result.get("itinerary", "")
        flights = result.get("flights", [])
        hotels = result.get("hotels", [])
        attractions = result.get("attractions", [])

        to_city = entities.get("to_city", "your destination")

        parts = [f"## Travel Plan for {to_city}\n"]
        if flights:
            f = flights[0]
            parts.append(
                "**✈ Best Flight:** "
                f"{f.get('currency','')} {f.get('price','')} | "
                f"{f.get('departure','')}→{f.get('arrival','')} | {f.get('duration','')}"
            )
        if hotels:
            h = hotels[0]
            parts.append(
                f"**🏨 Top Hotel:** {h.get('name','')} — "
                f"{h.get('currency','')} {h.get('price','')}/night"
            )
        if attractions:
            top = ", ".join(a.get("name", "") for a in attractions[:5])
            parts.append(f"**📍 Top Attractions:** {top}")
        parts.append("\n### Itinerary\n" + itinerary)
        response = "\n\n".join(parts)

        # Persist itinerary as calendar events + surface them for frontend sync
        try:
            from calendar_store import bulk_create_events
            ev_defs = _travel_calendar_events(itinerary, to_city, entities.get("check_in"))
            state["calendar_events"] = bulk_create_events(ev_defs)
        except Exception as e:
            logger.error(f"Travel calendar creation failed: {e}")
            state["calendar_events"] = []
        state["todo_items"] = [{"text": f"Plan trip to {to_city}", "source": "ai_travel"}]

        state["module_output"] = {
            "response": response,
            "module": "travel",
            "data": {
                "entities": entities,
                "flights": flights,
                "hotels": hotels,
                "attractions": attractions,
                "itinerary": itinerary
            }
        }

        # Save messages to SQLite
        add_message(session_id, "user", user_query)
        add_message(session_id, "assistant", response)

        # Enqueue for Neo4j storage (Phase 3)
        state["extraction_tasks"].append({
            "type": "extract_trip_to_neo4j",
            "from_city": entities.get("from_city", "Unknown"),
            "to_city": to_city,
            "check_in": entities.get("check_in"),
            "check_out": entities.get("check_out"),
            "num_adults": entities.get("num_adults", 1),
            "itinerary": itinerary,
            "flights": flights,
            "hotels": hotels,
            "attractions": attractions,
            "session_id": session_id,
        })
        
        logger.info(f"Travel node completed successfully")
        
    except Exception as e:
        logger.error(f"Travel node error: {e}", exc_info=True)
        state["module_output"] = {
            "response": f"Sorry, I couldn't plan your trip: {str(e)}",
            "module": "travel",
            "data": {"error": str(e)}
        }
        state["errors"].append({
            "step": "travel_node",
            "error": str(e),
            "timestamp": time.time()
        })
    
    state["step"] = "travel_node"
    return state


def format_response(state: OrbixState) -> OrbixState:
    """Format the module output for sending to user."""
    response = state.get("module_output", {}).get("response", "No response")
    state["module_output"]["formatted"] = response
    state["step"] = "format_response"
    return state


def background_tasks(state: OrbixState) -> OrbixState:
    """
    Phase 3: Execute extraction tasks in parallel (async).
    Phase 4: Handle follow-up actions if multi-intent was detected.
    Phase 5: Store extracted data to Neo4j.
    
    This node returns immediately after queueing async tasks.
    Actual execution happens in background job queue.
    """
    session_id = state["session_id"]
    num_extraction = len(state.get("extraction_tasks", []))
    num_cache = len(state.get("cache_tasks", []))
    follow_ups = state.get("follow_up_actions", [])
    
    logger.info(
        f"Background tasks: {num_extraction} extraction, {num_cache} cache, "
        f"{len(follow_ups)} follow-ups"
    )

    # ======================================================================
    # PHASE 3: Execute Extraction Tasks (Async in Background)
    # ======================================================================
    if num_extraction > 0:
        try:
            from orchestration.extraction_executor import get_executor
            executor = get_executor()
            
            # Queue extraction tasks to run asynchronously
            # In a real implementation, these would be queued to a Celery/RQ job queue
            # For now, we'll spawn them as async tasks
            import threading
            
            def run_extractions():
                try:
                    import asyncio
                    # Create new event loop in thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Execute all extraction tasks in parallel
                    result = loop.run_until_complete(
                        executor.execute_tasks(state["extraction_tasks"], session_id)
                    )
                    
                    logger.info(f"Extraction completed: {result['success']} success, {result['failed']} failed")
                    loop.close()
                except Exception as e:
                    logger.error(f"Background extraction error: {e}", exc_info=True)
            
            # Start extraction in background thread (non-blocking)
            extraction_thread = threading.Thread(target=run_extractions, daemon=True)
            extraction_thread.start()
            logger.info("Extraction tasks queued for background execution")
            
        except Exception as e:
            logger.error(f"Error queueing extraction tasks: {e}")

    # ======================================================================
    # PHASE 4: Handle Follow-up Actions (Action Chaining)
    # ======================================================================
    if follow_ups and len(follow_ups) > 1:
        try:
            # Get the current module's output for context passing
            current_module = state.get("module_name")
            current_output = state.get("module_output", {})
            
            # Build context from current result
            from orchestration.action_chaining import get_chain_executor
            chain_executor = get_chain_executor()
            
            context = chain_executor.build_context(
                state["user_query"],
                {current_module: current_output}
            )
            
            # Queue follow-up actions
            state["follow_up_context"] = context
            logger.info(f"Follow-up actions queued: {follow_ups}")
            
        except Exception as e:
            logger.error(f"Error handling follow-up actions: {e}")

    # ======================================================================
    # PHASE 5: Trigger Neo4j Storage for Critical Data
    # ======================================================================
    try:
        from orchestration.memory_integration import get_memory_integration
        memory = get_memory_integration()
        
        # If this was a travel module response, trigger trip deduplication + storage
        if state.get("module_name") == "travel" and state.get("module_output"):
            travel_data = state.get("module_output", {}).get("data", {})
            if travel_data.get("entities"):
                entities = travel_data["entities"]
                trip_data = {
                    "from_city": entities.get("from_city"),
                    "to_city": entities.get("to_city"),
                    "check_in": entities.get("check_in"),
                    "check_out": entities.get("check_out"),
                    "num_adults": entities.get("num_adults", 1),
                    "num_nights": entities.get("num_nights", 1),
                    "flights": travel_data.get("flights", []),
                    "hotels": travel_data.get("hotels", []),
                    "attractions": travel_data.get("attractions", []),
                }
                
                # Deduplicate and store (Phase 5)
                def store_trip():
                    try:
                        trip_id = memory.deduplicate_trip(trip_data)
                        if trip_id:
                            logger.info(f"Trip {trip_id} stored to Neo4j")
                    except Exception as e:
                        logger.error(f"Error storing trip: {e}")
                
                threading.Thread(target=store_trip, daemon=True).start()
        
        # If action module, link email to contacts
        if state.get("module_name") == "action":
            action_output = state.get("module_output", {}).get("data", {})
            if action_output.get("email_sent"):
                recipient = action_output.get("recipient")
                def link_contact():
                    try:
                        memory.link_email_to_contact(recipient)
                    except Exception as e:
                        logger.error(f"Error linking contact: {e}")
                
                threading.Thread(target=link_contact, daemon=True).start()
                
    except Exception as e:
        logger.error(f"Error triggering Neo4j storage: {e}")
    
    state["step"] = "background_tasks"
    return state


# ============================================================================ #
# Build LangGraph Workflow
# ============================================================================ #

def build_workflow():
    """
    Construct the LangGraph workflow graph.
    
    Topology:
      START → route_query → prepare_context → [module node] → format_response → background_tasks → END
    """
    graph = StateGraph(OrbixState)
    
    # Add all nodes. Gmail/Meet/calendar actions no longer have a dedicated node —
    # the chat_node (MCP agent) executes them as tool calls.
    graph.add_node("route_query", route_query)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("chat_node", chat_node)
    graph.add_node("travel_node", travel_node)
    graph.add_node("format_response", format_response)
    graph.add_node("background_tasks", background_tasks)

    # Define edges
    graph.add_edge(START, "route_query")
    graph.add_edge("route_query", "prepare_context")

    # Conditional routing to module nodes (chat handles general + all actions)
    graph.add_conditional_edges(
        "prepare_context",
        determine_module,  # routing function
        {
            "chat_node": "chat_node",
            "travel_node": "travel_node",
        }
    )

    # All module nodes go to format_response
    graph.add_edge("chat_node", "format_response")
    graph.add_edge("travel_node", "format_response")

    # Format → background tasks → end
    graph.add_edge("format_response", "background_tasks")
    graph.add_edge("background_tasks", END)
    
    return graph.compile()


# Singleton workflow instance
_workflow = None


def get_workflow():
    """Get or create the compiled workflow."""
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


async def run_workflow(user_query: str, session_id: str) -> OrbixState:
    """
    Execute the workflow end-to-end.
    
    Returns the final state with:
      - module_output.formatted: response to send to user
      - extraction_tasks: queued for background processing
      - cache_tasks: queued for SQLite caching
    """
    workflow = get_workflow()
    initial_state = new_state(user_query, session_id)

    logger.info(f"Starting workflow for session {session_id}")

    # Async invoke — chat_node / travel_node are async coroutine nodes.
    final_state = await workflow.ainvoke(
        initial_state,
        config={"recursion_limit": 50}
    )

    logger.info(f"Workflow completed for session {session_id}")
    return final_state
