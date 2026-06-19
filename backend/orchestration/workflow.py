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
import json
import asyncio
import time
from typing import Literal
from uuid import uuid4

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .graph_state import OrbixState, new_state
from .routing import route_query, determine_module
from graph.working_memory import add_message, get_messages

logger = logging.getLogger(__name__)


# ============================================================================ #
# Node Implementations (Phase 1 Stubs)
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


def chat_node(state: OrbixState) -> OrbixState:
    """
    General chat module — integrates with agent.py's run_turn().
    
    Phase 2: Call the real agent loop with MCP tools.
    - Run agent_loop via run_turn() to get intelligent responses
    - Capture response and enqueue extraction tasks
    """
    session_id = state["session_id"]
    user_query = state["user_query"]
    
    try:
        logger.info(f"Chat node: calling agent.run_turn() for '{user_query[:50]}...'")
        
        # Import here to avoid circular imports
        from mcp_host.agent import run_turn
        
        # run_turn() is async, so we need to run it in the event loop
        # Since LangGraph nodes are sync, we use asyncio.run() or nest the call
        import asyncio
        try:
            # Try to get existing loop for nested async contexts
            loop = asyncio.get_running_loop()
            # If we get here, we're already in async context - create task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = executor.submit(asyncio.run, run_turn(user_query, session_id)).result()
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            result = asyncio.run(run_turn(user_query, session_id))
        
        response = result.get("reply", "I couldn't generate a response")
        trace = result.get("trace", [])
        
        state["module_output"] = {
            "response": response,
            "module": "chat",
            "data": {"agent_response": response, "trace_steps": len(trace)}
        }
        
        # Save to SQLite (working_memory already saved by agent)
        add_message(session_id, "assistant", response)
        
        # Enqueue extraction from the response (Phase 3 background task)
        state["extraction_tasks"].append({
            "type": "extract_from_turn",
            "user_text": user_query,
            "assistant_text": response,
            "session_id": session_id,
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


def travel_node(state: OrbixState) -> OrbixState:
    """
    Travel planner module — integrates with travel_planner.py.
    
    Phase 2: Call the real travel planner pipeline.
    - Extract travel parameters from query
    - Call plan_trip() to fetch flights, hotels, attractions, itinerary
    - Return structured itinerary
    - Enqueue for Neo4j storage
    """
    session_id = state["session_id"]
    user_query = state["user_query"]
    
    try:
        logger.info(f"Travel node: planning trip for '{user_query[:50]}...'")
        
        from google_service.travel_planner import plan_trip
        
        # plan_trip() takes a query and optional emit callback for progress
        # Returns dict with entities, flights, hotels, attractions, itinerary, error (if any)
        def emit_progress(msg):
            logger.info(f"Travel progress: {msg}")
        
        result = plan_trip(query=user_query, emit=emit_progress)
        
        # Check for errors in the result
        if "error" in result:
            raise ValueError(result["error"])
        
        # Format response for user
        entities = result.get("entities", {})
        itinerary = result.get("itinerary", "")
        flights = result.get("flights", [])
        hotels = result.get("hotels", [])
        attractions = result.get("attractions", [])
        
        to_city = entities.get("to_city", "Unknown")
        
        response = f"I've planned a trip to {to_city} for you!\n\n{itinerary}"
        if flights:
            response += f"\n\n**Flights:** Found {len(flights)} option(s)"
        if hotels:
            response += f"\n**Hotels:** Found {len(hotels)} option(s)"
        if attractions:
            response += f"\n**Attractions:** {len(attractions)} places to visit"
        
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


def action_node(state: OrbixState) -> OrbixState:
    """
    Action executor module — handles send_email, create_meeting, get_emails, etc.
    
    Phase 2: Call the real workflow executor with intent classification.
    - Uses intent_classifier to extract parameters (from LLM)
    - Calls appropriate service (gmail, calendar, etc.)
    - Handles errors with retry queueing
    """
    session_id = state["session_id"]
    user_query = state["user_query"]
    intent = state.get("intent", "unknown")
    
    try:
        logger.info(f"Action node: executing {intent} for '{user_query[:50]}...'")
        
        from intent_workflow.intent_classifier import IntentClassifier
        from google_service.gmail_client import GmailClient
        from google_service.mail_generator import MailGenerator
        import calendar_store
        
        # Classify and extract parameters
        classification = IntentClassifier.classify(user_query)
        extracted_intent = classification.get("intent", intent)
        parameters = classification.get("parameters", {})
        
        response = None
        action_data = {}
        
        # Execute based on extracted intent
        if extracted_intent == "send_email":
            gmail = GmailClient()
            recipient_email = parameters.get("recipient_email")
            if not recipient_email:
                raise ValueError("Recipient email is required to send an email")
            
            recipient_name = parameters.get("recipient_name", recipient_email)
            
            # Use pre-generated content from classifier if available
            email_content = classification.get("email_content", {})
            if not email_content.get("subject") or not email_content.get("body"):
                # Fall back to mail_generator if not available
                email_content = MailGenerator.generate_mail_content(
                    user_prompt=user_query,
                    recipient_name=recipient_name,
                    prefilled=email_content
                )
            
            # Send the email
            result = gmail.send_email(
                to=recipient_email,
                subject=email_content.get("subject", ""),
                body=email_content.get("body", "")
            )
            
            response = f"Email sent successfully to {recipient_email}!"
            action_data = {"email_sent": True, "recipient": recipient_email}
            
        elif extracted_intent == "create_meeting":
            event_title = parameters.get("event_title")
            if not event_title:
                raise ValueError("Event title is required to create a meeting")
            
            # Create calendar event
            event = calendar_store.create_event(
                title=event_title,
                date=parameters.get("date", ""),
                time=parameters.get("time", ""),
                description=parameters.get("event_description", ""),
                type="meeting"
            )
            
            response = f"Meeting '{event_title}' created successfully!"
            action_data = {"event_created": True, "event_id": event.get("id")}
            
        elif extracted_intent == "meeting_and_email":
            gmail = GmailClient()
            event_title = parameters.get("event_title")
            attendee_email = parameters.get("attendee_email")
            
            if not event_title or not attendee_email:
                raise ValueError("Event title and attendee email are required")
            
            # Step 1: Create meeting
            event = calendar_store.create_event(
                title=event_title,
                date=parameters.get("date", ""),
                time=parameters.get("time", ""),
                description=parameters.get("event_description", ""),
                type="meeting"
            )
            
            # Step 2: Generate and send invitation email
            email_content = classification.get("email_content", {})
            if not email_content.get("subject") or not email_content.get("body"):
                email_content = MailGenerator.generate_mail_content(
                    user_prompt=user_query,
                    recipient_name=attendee_email,
                    prefilled=email_content
                )
            
            gmail.send_email(
                to=attendee_email,
                subject=email_content.get("subject", ""),
                body=email_content.get("body", "")
            )
            
            response = f"Meeting created and invitation sent to {attendee_email}!"
            action_data = {"event_created": True, "email_sent": True, "event_id": event.get("id")}
            
        elif extracted_intent == "get_emails":
            gmail = GmailClient()
            max_results = int(parameters.get("max_results", 5))
            
            # Fetch recent emails
            emails = gmail.get_emails(max_results=max_results)
            
            # Format for user
            if emails:
                email_list = "\n".join([
                    f"- From: {e.get('from')}, Subject: {e.get('subject')}"
                    for e in emails
                ])
                response = f"Here are your {len(emails)} recent emails:\n{email_list}"
            else:
                response = "You don't have any recent emails."
            
            action_data = {"emails_retrieved": True, "count": len(emails) if emails else 0}
            
        else:
            response = f"I don't recognize the action '{extracted_intent}'. Please try again."
            action_data = {"error": f"Unknown action: {extracted_intent}"}
        
        state["module_output"] = {
            "response": response,
            "module": "action",
            "data": action_data
        }
        
        # Save messages to SQLite
        add_message(session_id, "user", user_query)
        add_message(session_id, "assistant", response)
        
        # Enqueue for extraction (Phase 3)
        state["extraction_tasks"].append({
            "type": "extract_action",
            "action": extracted_intent,
            "parameters": parameters,
            "session_id": session_id,
        })
        
        logger.info(f"Action node completed: {extracted_intent}")
        
    except Exception as e:
        logger.error(f"Action node error: {e}", exc_info=True)
        state["module_output"] = {
            "response": f"I couldn't complete that action: {str(e)}",
            "module": "action",
            "data": {"error": str(e)}
        }
        state["errors"].append({
            "step": "action_node",
            "error": str(e),
            "timestamp": time.time()
        })
    
    state["step"] = "action_node"
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
    
    # Add all nodes
    graph.add_node("route_query", route_query)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("chat_node", chat_node)
    graph.add_node("travel_node", travel_node)
    graph.add_node("action_node", action_node)
    graph.add_node("format_response", format_response)
    graph.add_node("background_tasks", background_tasks)
    
    # Define edges
    graph.add_edge(START, "route_query")
    graph.add_edge("route_query", "prepare_context")
    
    # Conditional routing to module nodes
    graph.add_conditional_edges(
        "prepare_context",
        determine_module,  # routing function
        {
            "chat_node": "chat_node",
            "travel_node": "travel_node",
            "action_node": "action_node",
        }
    )
    
    # All module nodes go to format_response
    graph.add_edge("chat_node", "format_response")
    graph.add_edge("travel_node", "format_response")
    graph.add_edge("action_node", "format_response")
    
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
    
    # Run synchronously (LangGraph handles async internally if needed)
    final_state = workflow.invoke(
        initial_state,
        config={"recursion_limit": 50}
    )
    
    logger.info(f"Workflow completed for session {session_id}")
    return final_state
