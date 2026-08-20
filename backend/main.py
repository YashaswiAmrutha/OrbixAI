from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pathlib import Path
from llm.ollama_client import generate_response
from google_service.gmail_client import GmailClient
from google_service.mail_generator import MailGenerator
from google_service.travel_planner import plan_trip
from intent_workflow import IntentClassifier
from calendar_store import get_events, create_event, bulk_create_events, update_event, delete_event
from faster_whisper import WhisperModel
from graph.working_memory import (
    init_db as init_working_memory,
    cleanup_expired_messages,
    reset_stale_processing,
    queue_stats,
    add_message,
    get_messages,
    start_session,
)
from orchestration.workflow import run_workflow
import asyncio
import tempfile
import json
import re
from datetime import datetime, timedelta
import logging
import os

# Ollama uses the M1 Pro's Metal GPU automatically. Do not set num_gpu=0 or an
# invalid visible-device override here; both force CPU inference.
# Allow OAuth over HTTP for local development
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Load HuggingFace token from .env if present
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

logger = logging.getLogger(__name__)
# DEBUG logging from urllib3/httpcore prints full request URLs, including API
# query parameters. Keep normal operational detail without leaking credentials
# into the terminal.
logging.basicConfig(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Frontend directory (used for serving static files and index.html)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

_MAINTENANCE_INTERVAL_S = int(os.environ.get("ORBIX_MAINTENANCE_INTERVAL_S", str(15 * 60)))


def _run_maintenance():
    """
    Single shared maintenance routine (called by the scheduled loop below and by
    POST /workflow/cleanup) — one code path, not two.

    Order matters: drain any pending extract_queue rows into Neo4j FIRST (the durable
    retry safety net for agent.py's fast per-turn extraction — if that fast path fails,
    e.g. Neo4j briefly down, the row stays 'pending' here instead of the fact being
    lost outright), and only THEN expire old SQLite messages. cleanup_expired_messages()
    already refuses to delete a session with pending/processing extract_queue rows;
    draining first is what makes that check actually meaningful.
    """
    try:
        reset_stale_processing()  # crash recovery: anything stuck 'processing' -> 'pending'
    except Exception as e:
        logger.error(f"reset_stale_processing failed: {e}")

    try:
        from mcp_host import extractor
        drain_result = extractor.drain_all()
        logger.info(f"Maintenance drain: {drain_result}")
    except Exception as e:
        logger.error(f"extract_queue drain failed: {e}")

    try:
        cleanup_result = cleanup_expired_messages()
        logger.info(f"Expired messages cleanup: {cleanup_result}")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

    # Repair any graph node that reached Neo4j without an identity key. `key` is the
    # sole MERGE target for every memory write, so a keyless node is unreachable by
    # recall() and can never be deduplicated — the next mention of the same entity
    # creates a second node. Neo4j Community cannot enforce this at the database
    # level (property-existence and node-key constraints are Enterprise-only), so
    # the invariant is maintained here instead. Raw-Cypher writers have been fixed
    # to set keys directly; this is the backstop for anything that regresses.
    try:
        from graph import memory as _mem
        backfill_result = _mem.backfill_missing_keys()
        if backfill_result.get("assigned") or backfill_result.get("relabeled"):
            logger.info(f"Graph key backfill: {backfill_result}")
    except Exception as e:
        logger.error(f"Graph key backfill failed: {e}")


async def _maintenance_loop():
    """Background loop: periodically drain extract_queue, then expire old messages.
    Runs for the app's lifetime; cancelled cleanly on shutdown via lifespan()."""
    while True:
        await asyncio.sleep(_MAINTENANCE_INTERVAL_S)
        try:
            await asyncio.to_thread(_run_maintenance)
        except Exception as e:
            logger.error(f"Maintenance loop iteration failed: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On launch: ensure Ollama is serving, initialize SQLite schema, start the
    periodic maintenance loop. Never blocks fatally."""
    try:
        # Initialize SQLite working memory (creates tables if needed)
        init_working_memory()
        logger.info("Working memory SQLite initialized")
    except Exception as e:
        logger.error(f"Working memory init failed: {e}", exc_info=True)

    try:
        from mcp_host import startup
        await asyncio.to_thread(startup.run_startup)
    except Exception as e:
        logger.error("Startup orchestration failed (continuing): %s", e, exc_info=True)

    maintenance_task = asyncio.create_task(_maintenance_loop())
    try:
        yield
    finally:
        maintenance_task.cancel()
        try:
            await maintenance_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load whisper model lazily
model = None

# Initialize Gmail client (lazy initialization)
gmail_client = None

def get_model():
    global model
    if model is None:
        logger.info("Loading Whisper model on CPU...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully")
    return model

def get_gmail_client():
    """Get or create Gmail client. Returns None if not authenticated (needs OAuth)."""
    global gmail_client
    if gmail_client is None:
        try:
            gmail_client = GmailClient()
            logger.info("Gmail client created")
        except Exception as e:
            logger.error(f"Failed to create Gmail client: {str(e)}")
            gmail_client = None
            return None
    # If credentials expired, try to reload token (might have been refreshed)
    if not gmail_client.is_authenticated():
        gmail_client._try_load_token()
    return gmail_client if gmail_client.is_authenticated() else gmail_client


# ============ CALENDAR HELPERS ============
def extract_travel_calendar_events(itinerary: str, destination: str,
                                   departure_date: str = None) -> list:
    """Parse day-wise itinerary text into calendar event dicts."""
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
            # Strip trailing markdown emphasis markers (e.g. "...Landmarks**" from
            # a "**Day 1: ...Landmarks**" bold heading line) so they don't leak
            # into the calendar event title.
            clean_title = re.sub(r'[\*_#]+$', '', title.strip()).strip()
            events.append({
                "title": f"Day {day_num}: {clean_title[:70]}",
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

@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/auth/status")
def auth_status():
    """Check if Gmail is authenticated"""
    client = get_gmail_client()
    if client and client.is_authenticated():
        return {"authenticated": True}
    return {"authenticated": False, "auth_url": "/auth/login"}

@app.get("/auth/profile")
def auth_profile():
    """Return the logged-in user's display name and email for personalization."""
    client = get_gmail_client()
    if not client or not client.is_authenticated():
        return {"authenticated": False, "name": None, "email": None}
    try:
        prof = client.get_user_profile()
        return {"authenticated": True, "name": prof.get("name"), "email": prof.get("email")}
    except Exception as e:
        logger.error(f"auth_profile error: {e}")
        return {"authenticated": True, "name": None, "email": None}

@app.get("/auth/login")
def auth_login():
    """Start OAuth flow — redirects browser to Google login"""
    client = get_gmail_client()
    if not client:
        return {"error": "Gmail client could not be created"}
    auth_url, state = client.get_auth_url()
    return RedirectResponse(url=auth_url)

@app.get("/auth/callback")
def auth_callback(request: Request):
    """OAuth callback — Google redirects here after login"""
    try:
        client = get_gmail_client()
        if not client:
            return {"error": "Gmail client could not be created"}
        # Pass the full callback URL to exchange the code for tokens
        client.handle_auth_callback(str(request.url))
        logger.info("OAuth callback successful, redirecting to app")
        return RedirectResponse(url="/")
    except Exception as e:
        logger.error(f"OAuth callback error: {str(e)}")
        return {"error": f"Authentication failed: {str(e)}"}

@app.get("/health")
def health_check():
    return {
        "message": "Orbii Backend API is running",
        "status": "ok",
        "build": "optimized-2026-08-20.3",
    }


@app.get("/memory/health")
def memory_health():
    """
    Visibility into the two-tier memory pipeline: how many SQLite turns are still
    waiting to be promoted into Neo4j (by status — pending/processing/failed), and
    whether Neo4j itself is reachable. `failed` rows previously only surfaced as a
    log line; they're counted here so a stuck extractor doesn't go unnoticed.
    """
    stats = queue_stats()
    try:
        from mcp_host import extractor
        neo4j_reachable = extractor.mem_reachable()
    except Exception as e:
        neo4j_reachable = False
        logger.error(f"memory_health: reachability check failed: {e}")

    # Graph integrity: unkeyed nodes are silently unreachable and undedupable, and
    # Community edition can't constrain against them — so surface the count rather
    # than letting it drift unnoticed. Non-zero means some writer bypassed the
    # keyed-upsert discipline; the maintenance loop repairs it on its next pass.
    unkeyed = None
    if neo4j_reachable:
        try:
            from graph import memory as _mem
            unkeyed = _mem.count_unkeyed()
        except Exception as e:
            logger.error(f"memory_health: unkeyed count failed: {e}")

    return {"queue": stats, "neo4j_reachable": neo4j_reachable,
            "graph": {"unkeyed_entities": unkeyed,
                      "healthy": (unkeyed == 0) if unkeyed is not None else None}}


# ============ MCP INTEGRATIONS (enable/disable servers) ============
# The frontend Integrations page reads and toggles the MCP server roster here.
# Toggling only records intent in mcp_state.json; the agent picks up the enabled set
# on its next turn. A disabled/misconfigured server is simply skipped by the host, so
# nothing here can break the running backend.

@app.get("/mcp/servers")
def mcp_list_servers():
    """List every known MCP integration with its enabled state + config (secrets redacted)."""
    try:
        from mcp_host import registry
        return {"servers": registry.list_servers()}
    except Exception as e:
        logger.error(f"mcp_list_servers error: {e}", exc_info=True)
        return {"servers": [], "error": str(e)}


@app.get("/model/list")
def model_list():
    """Catalog of known agent models + which one is active (for the model toggle UI)."""
    try:
        from llm import model_registry
        return {"models": model_registry.list_models()}
    except Exception as e:
        logger.error(f"model_list error: {e}", exc_info=True)
        return {"models": [], "error": str(e)}


@app.post("/model/active")
def model_set_active(payload: dict):
    """
    Switch the active agent model. Body: {"model": "llama3.1:8b"}.
    Takes effect on the next agent turn — no restart needed (agent.py reads
    the registry fresh each call, same pattern as the MCP server toggles).
    """
    try:
        from llm import model_registry
        model_id = (payload.get("model") or "").strip()
        if not model_id:
            return {"success": False, "error": "model is required"}
        updated = model_registry.set_active(model_id)
        return {"success": True, "model": updated}
    except Exception as e:
        logger.error(f"model_set_active error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.post("/mcp/servers/{server_id}")
def mcp_update_server(server_id: str, payload: dict):
    """
    Enable/disable a server and/or update its config.
    Body: {"enabled": bool?, "config": {..}?}. Returns the updated server view.
    """
    try:
        from mcp_host import registry
        updated = None
        if "config" in payload and isinstance(payload["config"], dict):
            updated = registry.set_config(server_id, payload["config"])
        if "enabled" in payload:
            updated = registry.set_enabled(server_id, bool(payload["enabled"]))
        if updated is None:
            # no recognized field, or unknown id
            from mcp_host.registry import _one, _BY_ID  # type: ignore
            if server_id not in _BY_ID:
                return {"success": False, "error": f"unknown server '{server_id}'"}
            updated = _one(server_id)
        return {"success": True, "server": updated}
    except Exception as e:
        logger.error(f"mcp_update_server error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ============ LANGGRAPH WORKFLOW ENDPOINTS (Phase 1) ============

@app.post("/workflow/chat")
async def workflow_chat(message: dict):
    """
    LangGraph workflow endpoint — Phase 1 testing.
    
    Request: {"message": "...", "session_id": "..."}
    Response: Streaming SSE events
      - {type: "thinking", step: "..."}
      - {type: "response", reply: "...", module: "..."}
    """
    user_message = message.get("message", "").strip()
    session_id = (message.get("session_id") or "").strip()
    
    if not user_message:
        return {"error": "message is required"}
    
    if not session_id:
        from graph.working_memory import start_session
        session_id = start_session()
    
    logger.info(f"[LangGraph] Starting workflow for session {session_id}")
    
    async def generate():
        """Stream SSE events from LangGraph workflow execution."""
        from orchestration.workflow import (
            register_progress_emitter, unregister_progress_emitter,
        )
        loop = asyncio.get_event_loop()
        step_queue: asyncio.Queue = asyncio.Queue()

        def _emit_step(step: str):
            # Called (synchronously) from inside workflow nodes — hand the step to
            # the SSE generator via the event loop.
            try:
                loop.call_soon_threadsafe(
                    step_queue.put_nowait, {"type": "thinking", "step": step}
                )
            except Exception:
                pass

        register_progress_emitter(session_id, _emit_step)
        task = None
        try:
            yield f'data: {json.dumps({"type": "thinking", "step": "Understanding your request…"})}\n\n'

            # Run the workflow while draining progress steps concurrently
            task = asyncio.ensure_future(run_workflow(user_message, session_id))
            while not (task.done() and step_queue.empty()):
                try:
                    ev = await asyncio.wait_for(step_queue.get(), timeout=0.2)
                    yield f'data: {json.dumps(ev)}\n\n'
                except asyncio.TimeoutError:
                    continue

            final_state = await task

            mo = final_state.get("module_output", {}) or {}
            reply = mo.get("formatted") or mo.get("response") or "No response"
            module = mo.get("module", "unknown")
            intent = final_state.get("intent")
            sid = final_state.get("session_id", session_id)

            payload = {
                "type": "response",
                "reply": reply,
                "module": module,
                "intent": intent,
                "session_id": sid,
            }
            cal = final_state.get("calendar_events") or []
            todos = final_state.get("todo_items") or []
            if cal:
                payload["calendar_events"] = cal
            if todos:
                payload["todo_items"] = todos
            emails = (mo.get("data") or {}).get("emails")
            if emails:
                payload["emails"] = emails
            elapsed = (mo.get("data") or {}).get("elapsed_seconds")
            if elapsed is not None:
                payload["elapsed_seconds"] = elapsed

            logger.info(f"[LangGraph] Workflow completed: module={module}, intent={intent}")
            yield f'data: {json.dumps(payload)}\n\n'

        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'
        finally:
            unregister_progress_emitter(session_id)
            # If the client disconnected (tab closed, request timed out) before the
            # workflow finished, `task` was left running orphaned in the background —
            # confirmed live: a client-side timeout on a slow multi-tool turn didn't
            # stop the server from continuing to call send_email/create_meet minutes
            # after nobody was waiting on the result. GeneratorExit (client gone)
            # still reaches this finally, so cancel anything still in flight.
            if task is not None and not task.done():
                task.cancel()

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/workflow/cleanup")
async def workflow_cleanup_now(background_tasks: BackgroundTasks):
    """
    Manually trigger the same drain+cleanup routine the scheduled maintenance loop
    runs every ORBIX_MAINTENANCE_INTERVAL_S seconds (useful for on-demand testing/ops).
    """
    background_tasks.add_task(_run_maintenance)
    return {"status": "maintenance queued"}



# ============ CALENDAR ENDPOINTS ============
@app.get("/calendar/events")
def list_calendar_events(year: int = None, month: int = None):
    return {"events": get_events(year, month)}


@app.post("/calendar/events")
def add_calendar_event(payload: dict):
    ev = create_event(**payload)
    return {"success": True, "event": ev}


@app.post("/calendar/events/bulk")
def add_calendar_events_bulk(payload: dict):
    events = bulk_create_events(payload.get("events", []))
    return {"success": True, "events": events, "count": len(events)}


@app.put("/calendar/events/{event_id}")
def modify_calendar_event(event_id: str, payload: dict):
    ev = update_event(event_id, **payload)
    if ev:
        return {"success": True, "event": ev}
    return {"success": False, "error": "Event not found"}


@app.delete("/calendar/events/{event_id}")
def remove_calendar_event(event_id: str):
    return {"success": delete_event(event_id)}

@app.post("/chat")
async def chat(message: dict):
    """
    Non-streaming chat — delegates to the LangGraph workflow (route → context →
    chat/travel module → background writes). Gmail/Meet/calendar actions are executed
    by the MCP agent inside the chat module, not by hardcoded intent branches.
    """
    try:
        user_text = (message.get("message") or "").strip()
        if not user_text:
            return {"error": "message is required", "reply": "Please type a message."}

        session_id = (message.get("session_id") or "").strip()
        if not session_id:
            from graph.working_memory import start_session
            session_id = start_session()

        final_state = await run_workflow(user_text, session_id)
        mo = final_state.get("module_output", {}) or {}
        reply = mo.get("formatted") or mo.get("response") or "No response"
        resp = {"reply": reply, "intent": final_state.get("intent"),
                "session_id": final_state.get("session_id", session_id)}
        emails = (mo.get("data") or {}).get("emails")
        if emails:
            resp["emails"] = emails
        return resp

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return {"error": str(e), "reply": "Sorry, I encountered an error. Is Ollama running?"}


@app.post("/chat/stream")
async def chat_stream(message: dict):
    """
    Streaming chat endpoint — emits SSE events so the frontend can show
    a live 'thinking' bubble with each processing step.

    Event types:
      thinking  → {type:"thinking", step:"..."}
      response  → {type:"response", reply:"...", intent:"...", action?:"...", parameters?:{}}
      error     → {type:"error", message:"..."}
    """
    user_text = message.get("message", "")
    loop = asyncio.get_event_loop()

    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    async def generate():
        plan_task = agent_task = None
        try:
            yield _sse({"type": "thinking", "step": "Understanding your request…"})
            await asyncio.sleep(0)

            # ── Intent classification ────────────────────────────────────────
            # Regex first (instant, no model call) — this only needs to decide
            # travel vs. everything-else (every other label routes the same way
            # below), and calling the separate fine-tuned classifier here on
            # every turn was evicting the agent loop's warm model context (see
            # PERFORMANCE_AUDIT.md item 3). Only escalates to that model when
            # explicitly opted in via ORBIX_USE_LLM_ROUTER=1.
            from orchestration.routing import _regex_route
            intent, _confidence = _regex_route(user_text)
            effective_query = user_text
            sid = (message.get("session_id") or "").strip() or None
            if intent == "general_chat" and sid and re.fullmatch(
                r"\s*(?:go ahead|continue|do it|yes|please do|proceed)[.!]?\s*",
                user_text, re.I,
            ):
                for prior in reversed(get_messages(sid, limit=12)):
                    if prior.get("role") != "user":
                        continue
                    prior_intent, _ = _regex_route(prior.get("text", ""))
                    if prior_intent == "flight_search":
                        intent, effective_query = prior_intent, prior["text"]
                        break
            if os.environ.get("ORBIX_USE_LLM_ROUTER") == "1":
                try:
                    classification = await loop.run_in_executor(
                        None, IntentClassifier.classify, user_text
                    )
                    intent = classification.get("intent") or intent
                except Exception as e:
                    logger.warning(f"LLM classifier failed ({e}); keeping regex result")

            intent_label = intent.replace("_", " ").title()
            yield _sse({"type": "thinking", "step": f"Intent detected: {intent_label}"})
            await asyncio.sleep(0)

            if intent == "flight_search":
                from orchestration.flight_fastpath import execute_flight_search

                yield _sse({"type": "thinking", "step": "Searching live economy fares…"})
                fare_result = await asyncio.to_thread(execute_flight_search, effective_query)
                response = (fare_result or {}).get("response") or (
                    "I couldn't parse that flight search. Please include origin, "
                    "destination, and departure date."
                )
                sid = sid or start_session()
                add_message(sid, "user", user_text)
                add_message(sid, "assistant", response, enqueue=False)
                yield _sse({"type": "response", "reply": response,
                            "intent": intent, "session_id": sid})
                return

            # ── travel_planner → dedicated MCP travel pipeline ───────────────
            # Travel keeps its own branch; everything else (general chat AND
            # Gmail/Meet/calendar actions) is handled by the memory- and action-aware
            # MCP agent below, which decides and executes tool calls itself.
            if intent == "travel_planner":
                # We'll stream individual travel steps through a queue
                step_queue = asyncio.Queue()
                travel_result = {}

                def _emit(step: str):
                    asyncio.run_coroutine_threadsafe(
                        step_queue.put(step), loop
                    )

                async def _run_plan():
                    # plan_trip is a coroutine (spawns an MCP travel server over stdio),
                    # so await it directly on the event loop instead of run_in_executor
                    # (which would return an un-awaited coroutine).
                    result = await plan_trip(user_text, emit=_emit)
                    await step_queue.put(None)  # sentinel
                    return result

                plan_task = asyncio.ensure_future(_run_plan())

                # Drain step queue while planning runs
                while True:
                    step = await step_queue.get()
                    if step is None:
                        break
                    yield _sse({"type": "thinking", "step": step})
                    await asyncio.sleep(0)

                travel_result = await plan_task

                if "error" in travel_result:
                    yield _sse({"type": "response",
                                "reply": travel_result["error"],
                                "intent": intent})
                    return

                # Build reply from result
                dest      = travel_result["entities"].get("to_city", "your destination")
                itinerary = travel_result.get("itinerary", "")
                flights   = travel_result.get("flights", [])
                hotels    = travel_result.get("hotels", [])
                attrs     = travel_result.get("attractions", [])

                parts = [f"## Travel Plan for {dest}\n"]
                if flights:
                    parts.append("**✈ Best Flight:** " +
                        f"{flights[0]['currency']} {flights[0]['price']} | "
                        f"{flights[0]['departure']}→{flights[0]['arrival']} | "
                        f"{flights[0]['duration']}")
                if hotels:
                    parts.append("**🏨 Top Hotel:** " +
                        f"{hotels[0]['name']} — {hotels[0]['currency']} {hotels[0]['price']}/night")
                if attrs:
                    top = ", ".join(a["name"] for a in attrs[:5])
                    parts.append(f"**📍 Top Attractions:** {top}")
                parts.append("\n### Itinerary\n" + itinerary)

                # Auto-create calendar events from travel itinerary
                cal_ev_defs = extract_travel_calendar_events(
                    itinerary,
                    dest,
                    travel_result["entities"].get("departure_date"),
                )
                saved_cal_events = bulk_create_events(cal_ev_defs)
                todo_items = [{"text": f"Plan trip to {dest}", "source": "ai_travel"}]

                yield _sse({
                    "type": "response",
                    "reply": "\n\n".join(parts),
                    "intent": intent,
                    "calendar_events": saved_cal_events,
                    "todo_items": todo_items,
                })
                return

            # ── everything else → memory- & action-aware MCP agent ───────────
            # General chat and all Gmail/Meet/calendar actions run here: the agent
            # recalls over Neo4j and executes tool calls (send_email, create_meet,
            # calendar CRUD) itself, chaining them as needed. Calendar side-effects are
            # surfaced from the tool trace so the frontend auto-sync keeps working.
            from mcp_host.agent import run_turn
            from orchestration.workflow import _calendar_events_from_trace

            step_q = asyncio.Queue()
            tool_scopes = {
                "send_email": {"send_email", "recall", "search", "search_transcript"},
                "create_meeting": {"create_meet", "recall", "search"},
                "meeting_and_email": {"create_meet", "send_email", "recall", "search"},
                "get_emails": {"list_emails"},
                "add_task": {"add_task", "list_tasks"},
                "complete_task": {"complete_task", "list_tasks"},
                "list_tasks": {"list_tasks"},
                "web_search": {"web_search", "fetch_url"},
                "browser_automation": {
                    "browser_navigate", "browser_snapshot", "browser_click",
                    "browser_type", "browser_fill_form", "browser_select_option",
                    "browser_press_key", "browser_wait_for", "browser_tabs",
                    "browser_navigate_back", "browser_close",
                },
                "file_read": {"read_file", "list_directory", "search_files"},
                "file_write": {"read_file", "write_file", "list_directory", "search_files"},
                "calendar_read": {"list_calendar_events"},
                "calendar_write": {"create_calendar_event", "list_calendar_events"},
            }
            server_scopes = {
                "send_email": {"orbix-memory", "orbix-gsuite"},
                "create_meeting": {"orbix-memory", "orbix-gsuite"},
                "meeting_and_email": {"orbix-memory", "orbix-gsuite"},
                "get_emails": {"orbix-gsuite"},
                "add_task": {"orbix-tasks"},
                "complete_task": {"orbix-tasks"},
                "list_tasks": {"orbix-tasks"},
                "web_search": {"orbix-web"},
                "browser_automation": {"orbix-playwright"},
                "file_read": {"orbix-files"},
                "file_write": {"orbix-files"},
                "calendar_read": {"orbix-gsuite"},
                "calendar_write": {"orbix-gsuite"},
            }
            agent_task = asyncio.ensure_future(
                run_turn(user_text, session_id=sid, on_event=step_q.put,
                         allowed_tool_names=tool_scopes.get(intent),
                         allowed_server_names=server_scopes.get(
                             intent, {"orbix-memory"}
                         )))
            # stream the agent's tool steps into the thinking bubble as they happen
            while not (agent_task.done() and step_q.empty()):
                try:
                    ev = await asyncio.wait_for(step_q.get(), timeout=0.2)
                    yield _sse(ev)
                    await asyncio.sleep(0)
                except asyncio.TimeoutError:
                    continue
            out = await agent_task

            cal_events, todos = _calendar_events_from_trace(out.get("trace", []))
            payload = {"type": "response", "reply": out["reply"],
                       "intent": intent, "session_id": out["session_id"]}
            if cal_events:
                payload["calendar_events"] = cal_events
            if todos:
                payload["todo_items"] = todos
            yield _sse(payload)

        except Exception as e:
            logger.error("Stream chat error: %s", e, exc_info=True)
            yield _sse({"type": "error", "message": str(e)})
        finally:
            # Same orphaned-task risk as /workflow/chat if the client disconnects
            # mid-turn — cancel whichever background task is still running.
            for _t in (plan_task, agent_task):
                if _t is not None and not _t.done():
                    _t.cancel()

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/agent/chat")
async def agent_chat(message: dict, background_tasks: BackgroundTasks):
    """
    Memory-aware agent endpoint: query -> READ-ONLY LLM/MCP recall on Neo4j -> reply.
    Durable writes happen off-path: a flush-barrier drains this session's pending
    turns into Neo4j before answering, and a post-reply background task extracts the
    new turn. Carries a session_id for multi-turn context. Legacy /chat is untouched.
    """
    try:
        from mcp_host.agent import run_turn
        user_text = message.get("message", "")
        if not user_text:
            return {"error": "message is required"}
        session_id = message.get("session_id")
        out = await run_turn(user_text, session_id=session_id)
        # Durable memory writes now go through the LangGraph background write path;
        # this legacy endpoint no longer drains to Neo4j itself.
        return {
            "reply": out["reply"],
            "session_id": out["session_id"],
            "tools_used": [{"tool": t["tool"], "readonly": t["readonly"]} for t in out["trace"]],
        }
    except Exception as e:
        logger.error("Agent chat error: %s", e, exc_info=True)
        return {"error": str(e), "reply": "Agent error — is Ollama running and a "
                "tool-calling model pulled? Is Neo4j up?"}


@app.post("/process-intent")
async def process_intent(payload: dict):
    """
    Intent processor — delegates to the LangGraph workflow, which routes the query to
    the MCP agent (Gmail/Meet/calendar actions as tool calls) or the travel pipeline.
    """
    try:
        user_query = (payload.get("query") or "").strip()
        if not user_query:
            return {"success": False, "error": "query is required"}

        logger.info(f"Processing intent for query: {user_query}")

        from graph.working_memory import start_session
        session_id = (payload.get("session_id") or "").strip() or start_session()

        final_state = await run_workflow(user_query, session_id)
        mo = final_state.get("module_output", {}) or {}
        reply = mo.get("formatted") or mo.get("response") or "No response"
        return {
            "success": not bool(final_state.get("errors")),
            "intent": final_state.get("intent"),
            "reply": reply,
            "session_id": final_state.get("session_id", session_id),
            "workflow_executed": True,
        }

    except Exception as e:
        logger.error(f"Error in process_intent: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Intent processing error: {str(e)}",
            "workflow_executed": False
        }

@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    tmp_path = None
    try:
        logger.info(f"Received voice file: {file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"Transcribing audio from {tmp_path}")
        model_instance = get_model()
        segments, info = model_instance.transcribe(tmp_path, language="en")
        user_text = "".join([segment.text for segment in segments]).strip()
        
        logger.info(f"Transcribed text: {user_text}")

        if not user_text:
            return {
                "text": "",
                "reply": "I didn't catch that. Please try speaking again.",
                "intent": None,
                "workflow_executed": False
            }

        # Run the transcript through the LangGraph workflow — same brain as text chat
        # (MCP agent executes Gmail/Meet/calendar actions; travel via its pipeline).
        from graph.working_memory import start_session
        session_id = start_session()
        final_state = await run_workflow(user_text, session_id)
        mo = final_state.get("module_output", {}) or {}
        reply = mo.get("formatted") or mo.get("response") or "Done."
        logger.info(f"Voice command handled: intent={final_state.get('intent')}")
        return {
            "text": user_text,
            "reply": reply,
            "intent": final_state.get("intent"),
            "session_id": final_state.get("session_id", session_id),
            "workflow_executed": True,
        }
    except Exception as e:
        logger.error(f"Voice processing error: {str(e)}", exc_info=True)
        return {
            "text": "",
            "reply": f"Error processing voice: {str(e)}",
            "error": str(e),
            "intent": None,
            "workflow_executed": False
        }
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except:
            pass

@app.get("/vpn_test")
def vpn_test():
    return {
        "status": "VPN OK",
        "source": "WireGuard mesh",
    }


@app.get("/emails/latest")
async def get_latest_emails(max_results: int = 10):
    """Get the latest emails from inbox and sent"""
    import asyncio
    try:
        client = get_gmail_client()
        if not client or not client.is_authenticated():
            return {"error": "needs_auth", "auth_url": "/auth/login", "emails": []}

        loop = asyncio.get_event_loop()
        emails = await asyncio.wait_for(
            loop.run_in_executor(None, client.get_all_recent_emails, max_results),
            timeout=30.0
        )
        return {"success": True, "emails": emails}
    except asyncio.TimeoutError:
        logger.error("Email fetch timed out after 30s")
        # Return stale cache rather than an error so the UI stays populated
        if client and hasattr(client, "_email_cache") and client._email_cache:
            logger.info("Serving stale email cache after timeout")
            return {"success": True, "emails": client._email_cache, "stale": True}
        return {"error": "Gmail request timed out", "emails": []}
    except Exception as e:
        err = str(e)
        logger.error(f"Error fetching emails: {err}")
        if "needs_auth" in err:
            return {"error": "needs_auth", "auth_url": "/auth/login", "emails": []}
        return {"success": False, "error": err, "emails": []}


@app.post("/emails/send")
def send_email(payload: dict):
    """Send an email with optional LLM-generated content"""
    try:
        # Extract parameters
        to_email = payload.get("to_email")
        subject = payload.get("subject")
        body = payload.get("body")
        use_llm = payload.get("use_llm", False)
        user_prompt = payload.get("user_prompt")
        recipient_name = payload.get("recipient_name")
        meeting_link = payload.get("meeting_link")
        
        if not to_email:
            return {"success": False, "error": "to_email is required"}
        
        # Generate content using LLM if requested
        if use_llm and user_prompt:
            mail_content = MailGenerator.generate_mail_content(
                user_prompt,
                recipient_name=recipient_name,
                meeting_link=meeting_link
            )
            subject = mail_content['subject']
            body = mail_content['body']
        
        if not subject or not body:
            return {"success": False, "error": "subject and body are required"}
        
        # Send email
        client = get_gmail_client()
        if not client:
            return {"success": False, "error": "Gmail client not initialized"}
        
        result = client.send_email(to_email, subject, body)
        return result
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return {"success": False, "error": str(e)}


@app.post("/meetings/create")
def create_google_meet(payload: dict):
    """Create a Google Meet and optionally send email with the link"""
    try:
        event_title = payload.get("event_title", "Meeting")
        event_description = payload.get("event_description", "")
        attendee_email = payload.get("attendee_email")
        send_email_flag = payload.get("send_email", True)
        user_prompt = payload.get("user_prompt")
        
        if not attendee_email:
            return {"success": False, "error": "attendee_email is required"}
        
        # Create Google Meet
        client = get_gmail_client()
        if not client:
            return {"success": False, "error": "Gmail client not initialized"}
        
        meet_result = client.create_google_meet(
            event_title,
            event_description,
            attendee_email
        )
        
        if not meet_result['success']:
            return meet_result
        
        # Send email with meet link if requested
        if send_email_flag and user_prompt:
            mail_content = MailGenerator.generate_mail_content(
                user_prompt,
                recipient_name=attendee_email.split('@')[0],
                meeting_link=meet_result['meet_link']
            )
            
            send_result = client.send_email(
                attendee_email,
                mail_content['subject'],
                mail_content['body']
            )
            
            meet_result['email_sent'] = send_result['success']
        
        return meet_result
    except Exception as e:
        logger.error(f"Error creating Google Meet: {str(e)}")
        return {"success": False, "error": str(e)}


@app.post("/mail/generate-content")
def generate_mail_content(payload: dict):
    """Generate email subject and body using LLM"""
    try:
        user_prompt = payload.get("user_prompt")
        recipient_name = payload.get("recipient_name")
        meeting_link = payload.get("meeting_link")
        
        if not user_prompt:
            return {"success": False, "error": "user_prompt is required"}
        
        content = MailGenerator.generate_mail_content(
            user_prompt,
            recipient_name=recipient_name,
            meeting_link=meeting_link
        )
        
        return {"success": True, **content}
    except Exception as e:
        logger.error(f"Error generating mail content: {str(e)}")
        return {"success": False, "error": str(e)}


@app.post("/sms_ingest")
def sms_ingest(payload: dict):
    try:
        message = payload.get("body", "")

        prompt = f"""
    You received this SMS:

    {message}

    Decide:
    - Is this OTP?
    - Is this spam?
    - Is this important?
    - Should user be notified urgently?
    """

        ai_reply = generate_response(prompt)

        return {
            "status": "processed",
            "analysis": ai_reply,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "analysis": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# ============ SERVE FRONTEND ============
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
