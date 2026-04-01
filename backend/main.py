from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pathlib import Path
from llm.ollama_client import generate_response
from llm.prompt import build_prompt
from llm.hf_client import orchestrate
from google_service.gmail_client import GmailClient
from google_service.mail_generator import MailGenerator
from google_service.travel_planner import plan_trip
from intent_workflow import IntentClassifier, WorkflowExecutor, WorkflowTask
from faster_whisper import WhisperModel
import asyncio
import tempfile
import json
from datetime import datetime
import logging
import os

# Force CPU-only mode to avoid CUDA library issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
logging.basicConfig(level=logging.DEBUG)

# Frontend directory (used for serving static files and index.html)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI()

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

# Initialize Workflow Executor
workflow_executor = WorkflowExecutor()

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


# ============ WORKFLOW TASKS ============
def task_create_google_meet(attendee_email: str, event_title: str, event_description: str = "", **kwargs) -> dict:
    """Task: Create Google Meet"""
    try:
        client = get_gmail_client()
        if not client:
            raise Exception("Gmail client not initialized")
        
        logger.info(f"Task: Creating Google Meet - Title: {event_title}, Attendee: {attendee_email}")
        meet_result = client.create_google_meet(event_title, event_description, attendee_email)
        
        if not meet_result['success']:
            raise Exception(f"Failed to create meet: {meet_result.get('error', 'Unknown error')}")
        
        logger.info(f"Google Meet created: {meet_result['meet_link']}")
        return {
            "meet_link": meet_result['meet_link'],
            "event_id": meet_result.get('event_id'),
            "meeting_created": True
        }
    except Exception as e:
        logger.error(f"Error in task_create_google_meet: {str(e)}")
        raise


def task_send_email(recipient_email: str, subject: str = "", body: str = "",
                    use_llm: bool = False, user_prompt: str = "",
                    recipient_name: str = "", meeting_link: str = None,
                    email_content: dict = None, **kwargs) -> dict:
    """Task: Send Email — uses orchestrator pre-generated content when available."""
    try:
        client = get_gmail_client()
        if not client:
            raise Exception("Gmail client not initialized")

        logger.info(f"Task: Sending email to {recipient_email}")

        # Priority: orchestrator content > explicit subject/body > LLM generation
        if email_content and email_content.get("subject") and email_content.get("body"):
            logger.info("Using orchestrator-generated email content")
            subject = email_content["subject"]
            body    = email_content["body"]
            if meeting_link and meeting_link not in body:
                body += f"\n\nGoogle Meet Link: {meeting_link}"
        elif not subject or not body:
            logger.info("Generating email content via MailGenerator")
            mail_content = MailGenerator.generate_mail_content(
                user_prompt or recipient_email,
                recipient_name=recipient_name,
                meeting_link=meeting_link,
                prefilled=email_content
            )
            subject = mail_content["subject"]
            body    = mail_content["body"]

        if not subject or not body:
            raise Exception("subject and body are required")

        logger.info(f"Sending email — subject: {subject}")
        result = client.send_email(recipient_email, subject, body)

        if not result["success"]:
            raise Exception(f"Failed to send email: {result.get('error', 'Unknown error')}")

        logger.info(f"Email sent successfully to {recipient_email}")
        return {
            "email_sent": True,
            "recipient": recipient_email,
            "subject": subject,
            "message_id": result.get("message_id")
        }
    except Exception as e:
        logger.error(f"Error in task_send_email: {str(e)}")
        raise


def task_plan_travel(destination: str, origin: str = None, departure_date: str = None,
                     return_date: str = None, travelers: int = None,
                     preferences: list = None, travel_plan: dict = None, **kwargs) -> dict:
    """Task: Return travel plan — uses orchestrator-generated plan when available."""
    if travel_plan and travel_plan.get("itinerary"):
        logger.info(f"Task: Using orchestrator travel plan for {destination}")
        return {
            "destination": destination,
            "origin": origin,
            "travel_plan": travel_plan
        }
    # Fallback: ask Ollama to generate a basic plan
    logger.info(f"Task: Generating travel plan for {destination} via Ollama")
    prompt = f"Create a concise travel itinerary for a trip to {destination}"
    if origin:
        prompt += f" from {origin}"
    if departure_date:
        prompt += f" departing {departure_date}"
    if return_date:
        prompt += f" returning {return_date}"
    if travelers:
        prompt += f" for {travelers} traveler(s)"
    if preferences:
        prompt += f". Preferences: {', '.join(preferences)}"
    plan_text = generate_response(prompt)
    return {
        "destination": destination,
        "origin": origin,
        "travel_plan": {"itinerary": plan_text}
    }


def task_get_emails(max_results: int = 5, **kwargs) -> dict:
    """Task: Get latest emails"""
    try:
        client = get_gmail_client()
        if not client:
            raise Exception("Gmail client not initialized")
        
        logger.info(f"Task: Fetching latest {max_results} emails")
        emails = client.get_latest_emails(max_results)
        
        return {
            "emails": emails,
            "count": len(emails)
        }
    except Exception as e:
        logger.error(f"Error in task_get_emails: {str(e)}")
        raise


# ============ INITIALIZE WORKFLOWS ============
def register_workflows():
    """Register all workflows for different intents"""
    
    # Workflow 1: Send email only
    workflow_executor.register_workflow("send_email", [
        WorkflowTask(
            name="send_email",
            function=task_send_email,
            required_params=["recipient_email"],
            on_error="stop"
        )
    ])
    
    # Workflow 2: Create meeting only
    workflow_executor.register_workflow("create_meeting", [
        WorkflowTask(
            name="create_google_meet",
            function=task_create_google_meet,
            required_params=["attendee_email", "event_title"],
            on_error="stop"
        )
    ])
    
    # Workflow 3: Create meeting AND send email with invite
    workflow_executor.register_workflow("meeting_and_email", [
        WorkflowTask(
            name="create_google_meet",
            function=task_create_google_meet,
            required_params=["attendee_email", "event_title"],
            on_error="stop"
        ),
        WorkflowTask(
            name="send_email_with_invite",
            function=task_send_email,
            required_params=["recipient_email"],
            on_error="continue"  # Continue even if email fails
        )
    ])
    
    # Workflow 4: Schedule meeting (same as meeting_and_email for now)
    workflow_executor.register_workflow("schedule_meeting", [
        WorkflowTask(
            name="create_google_meet",
            function=task_create_google_meet,
            required_params=["attendee_email", "event_title"],
            on_error="stop"
        ),
        WorkflowTask(
            name="send_email_with_invite",
            function=task_send_email,
            required_params=["recipient_email"],
            on_error="continue"
        )
    ])
    
    # Workflow 5: Get emails
    workflow_executor.register_workflow("get_emails", [
        WorkflowTask(
            name="fetch_emails",
            function=task_get_emails,
            required_params=[],
            on_error="stop"
        )
    ])
    
    # Workflow 6: Travel planner
    workflow_executor.register_workflow("travel_planner", [
        WorkflowTask(
            name="plan_travel",
            function=task_plan_travel,
            required_params=["destination"],
            on_error="stop"
        )
    ])

    # Workflow 7: General chat (no special workflow)
    workflow_executor.register_workflow("general_chat", [])

    logger.info("All workflows registered successfully")


# Initialize workflows on startup
register_workflows()

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
    return {"message": "Orbii Backend API is running", "status": "ok"}

@app.post("/chat")
def chat(message: dict):
    try:
        user_text = message["message"]

        # ── Single-shot orchestration via gpraneeth555/llama-3-13k ──────────
        classification  = IntentClassifier.classify(user_text)
        intent          = classification["intent"]
        parameters      = classification.get("parameters", {})
        email_content   = classification.get("email_content", {})
        travel_plan     = classification.get("travel_plan", {})

        # Attach orchestrator outputs into parameters so workflow tasks receive them
        if email_content and (email_content.get("subject") or email_content.get("body")):
            parameters["email_content"] = email_content
        if travel_plan and travel_plan.get("itinerary"):
            parameters["travel_plan"] = travel_plan

        # Cross-fill email ↔ attendee
        if "attendee_email" in parameters and "recipient_email" not in parameters:
            parameters["recipient_email"] = parameters["attendee_email"]
        elif "recipient_email" in parameters and "attendee_email" not in parameters:
            parameters["attendee_email"] = parameters["recipient_email"]

        # ── general_chat ─────────────────────────────────────────────────────
        if intent == "general_chat":
            prompt = build_prompt(user_text)
            reply  = generate_response(prompt)
            return {"reply": reply, "intent": intent}

        # ── travel_planner ────────────────────────────────────────────────────
        if intent == "travel_planner":
            result = task_plan_travel(**parameters)
            plan   = result.get("travel_plan", {})
            dest   = result.get("destination", "your destination")
            reply_parts = [f"Here's your travel plan for **{dest}**:"]
            if plan.get("itinerary"):
                reply_parts.append(plan["itinerary"])
            if plan.get("recommendations"):
                reply_parts.append("**Recommendations:** " + ", ".join(plan["recommendations"]))
            if plan.get("budget_estimate"):
                reply_parts.append(f"**Budget:** {plan['budget_estimate']}")
            if plan.get("tips"):
                reply_parts.append(f"**Tips:** {plan['tips']}")
            return {"reply": "\n\n".join(reply_parts), "intent": intent}

        # ── get_emails ────────────────────────────────────────────────────────
        if intent == "get_emails":
            result = task_get_emails(max_results=parameters.get("max_results", 10))
            return {
                "reply": f"Fetched {result['count']} emails.",
                "intent": intent,
                "emails": result["emails"]
            }

        # Auto-upgrade create_meeting → meeting_and_email when email is present
        if intent == "create_meeting" and (
            parameters.get("attendee_email") or parameters.get("recipient_email")
        ):
            intent = "meeting_and_email"

        # ── email / meeting intents — execute directly ────────────────────────
        if intent in ("send_email", "meeting_and_email", "create_meeting", "schedule_meeting"):
            recipient = parameters.get("recipient_email") or parameters.get("attendee_email", "")
            if not recipient:
                return {"reply": "Please mention the recipient's email address.", "intent": intent}

            meet_link  = None
            parts      = []

            if intent in ("meeting_and_email", "create_meeting", "schedule_meeting"):
                try:
                    mr = task_create_google_meet(
                        attendee_email=recipient,
                        event_title=parameters.get("event_title", "Meeting"),
                        event_description=parameters.get("event_description", ""),
                    )
                    meet_link = mr.get("meet_link")
                    if meet_link:
                        parts.append(f"**Google Meet created:** {meet_link}")
                except Exception as e:
                    parts.append(f"Meet creation failed: {e}")

            if intent != "create_meeting":
                try:
                    er = task_send_email(
                        recipient_email=recipient,
                        email_content=email_content if email_content else None,
                        meeting_link=meet_link,
                        user_prompt=parameters.get("event_description", ""),
                        recipient_name=parameters.get("recipient_name", recipient.split("@")[0]),
                    )
                    subj = (email_content or {}).get("subject") or er.get("subject", "")
                    if er.get("email_sent"):
                        parts.append(f"**Email sent** to {recipient}"
                                     + (f"\n**Subject:** {subj}" if subj else ""))
                    else:
                        parts.append(f"Email failed: {er.get('error','unknown')}")
                except Exception as e:
                    parts.append(f"Email failed: {e}")

            return {"reply": "\n\n".join(parts) or "Done.", "intent": intent}

        # ── fallback ──────────────────────────────────────────────────────────
        prompt = build_prompt(user_text)
        reply  = generate_response(prompt)
        return {"reply": reply, "intent": intent}

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
        try:
            yield _sse({"type": "thinking", "step": "Understanding your request…"})
            await asyncio.sleep(0)

            # ── Intent classification (blocking → executor) ─────────────────
            classification = await loop.run_in_executor(
                None, IntentClassifier.classify, user_text
            )
            intent       = classification["intent"]
            parameters   = classification.get("parameters", {})
            email_content = classification.get("email_content", {})
            travel_plan_pre = classification.get("travel_plan", {})

            intent_label = intent.replace("_", " ").title()
            yield _sse({"type": "thinking", "step": f"Intent detected: {intent_label}"})
            await asyncio.sleep(0)

            # Cross-fill email ↔ attendee
            if "attendee_email" in parameters and "recipient_email" not in parameters:
                parameters["recipient_email"] = parameters["attendee_email"]
            elif "recipient_email" in parameters and "attendee_email" not in parameters:
                parameters["attendee_email"] = parameters["recipient_email"]

            # Upgrade create_meeting → meeting_and_email when an email address is present
            # (user said "meet with X@..." which implies sending an invite)
            if intent == "create_meeting" and (
                parameters.get("attendee_email") or parameters.get("recipient_email")
            ):
                intent = "meeting_and_email"
                yield _sse({"type": "thinking",
                            "step": "Attendee detected — will send invite email too"})
                await asyncio.sleep(0)

            if email_content and (email_content.get("subject") or email_content.get("body")):
                parameters["email_content"] = email_content
            if travel_plan_pre and travel_plan_pre.get("itinerary"):
                parameters["travel_plan"] = travel_plan_pre

            # ── general_chat ────────────────────────────────────────────────
            if intent == "general_chat":
                yield _sse({"type": "thinking", "step": "Generating response…"})
                await asyncio.sleep(0)
                prompt = build_prompt(user_text)
                reply  = await loop.run_in_executor(None, generate_response, prompt)
                yield _sse({"type": "response", "reply": reply, "intent": intent})
                return

            # ── travel_planner ──────────────────────────────────────────────
            if intent == "travel_planner":
                # We'll stream individual travel steps through a queue
                step_queue = asyncio.Queue()
                travel_result = {}

                def _emit(step: str):
                    asyncio.run_coroutine_threadsafe(
                        step_queue.put(step), loop
                    )

                async def _run_plan():
                    result = await loop.run_in_executor(
                        None, lambda: plan_trip(user_text, emit=_emit)
                    )
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

                yield _sse({"type": "response",
                            "reply": "\n\n".join(parts),
                            "intent": intent})
                return

            # ── get_emails ──────────────────────────────────────────────────
            if intent == "get_emails":
                yield _sse({"type": "thinking", "step": "Fetching your emails…"})
                await asyncio.sleep(0)
                result = await loop.run_in_executor(
                    None, lambda: task_get_emails(max_results=parameters.get("max_results", 10))
                )
                yield _sse({"type": "response",
                            "reply": f"Fetched {result['count']} emails.",
                            "intent": intent,
                            "emails": result["emails"]})
                return

            # ── email / meeting intents — execute directly, no modal ──────────
            if intent in ("send_email", "meeting_and_email",
                          "create_meeting", "schedule_meeting"):

                recipient = (parameters.get("recipient_email")
                             or parameters.get("attendee_email", ""))

                if not recipient:
                    yield _sse({"type": "response",
                                "reply": "I couldn't find a recipient email address in your request. "
                                         "Please mention the email address you'd like to contact.",
                                "intent": intent})
                    return

                meet_link = None

                # ── Step A: create Google Meet if needed ─────────────────────
                if intent in ("meeting_and_email", "create_meeting", "schedule_meeting"):
                    yield _sse({"type": "thinking", "step": "Creating Google Meet…"})
                    await asyncio.sleep(0)
                    try:
                        meet_result = await loop.run_in_executor(
                            None,
                            lambda: task_create_google_meet(
                                attendee_email=recipient,
                                event_title=parameters.get("event_title", "Meeting"),
                                event_description=parameters.get("event_description", ""),
                                **{k: v for k, v in parameters.items()
                                   if k not in ("attendee_email","event_title","event_description")}
                            )
                        )
                        meet_link = meet_result.get("meet_link")
                        yield _sse({"type": "thinking",
                                    "step": f"Meet created: {meet_link}"})
                        await asyncio.sleep(0)
                    except Exception as e:
                        yield _sse({"type": "thinking",
                                    "step": f"Meet creation failed: {e}"})
                        await asyncio.sleep(0)

                # ── Step B: send email (skip for create_meeting only) ─────────
                if intent != "create_meeting":
                    yield _sse({"type": "thinking", "step": f"Sending email to {recipient}…"})
                    await asyncio.sleep(0)
                    try:
                        email_result = await loop.run_in_executor(
                            None,
                            lambda: task_send_email(
                                recipient_email=recipient,
                                email_content=email_content if email_content else None,
                                meeting_link=meet_link,
                                user_prompt=parameters.get("event_description", "")
                                            or parameters.get("body", ""),
                                recipient_name=parameters.get("recipient_name",
                                                              recipient.split("@")[0]),
                                **{k: v for k, v in parameters.items()
                                   if k not in ("recipient_email","email_content",
                                                "meeting_link","user_prompt","recipient_name")}
                            )
                        )
                        yield _sse({"type": "thinking", "step": "Email sent ✓"})
                        await asyncio.sleep(0)
                    except Exception as e:
                        yield _sse({"type": "thinking",
                                    "step": f"Email failed: {e}"})
                        await asyncio.sleep(0)
                        email_result = {"email_sent": False, "error": str(e)}

                # ── Build reply ───────────────────────────────────────────────
                parts = []
                if meet_link:
                    parts.append(f"**Google Meet created:** {meet_link}")
                if intent != "create_meeting":
                    subj = (email_content or {}).get("subject") or email_result.get("subject", "")
                    sent_ok = email_result.get("email_sent", False)
                    if sent_ok:
                        parts.append(f"**Email sent** to {recipient}"
                                     + (f"\n**Subject:** {subj}" if subj else ""))
                    else:
                        parts.append(f"Email to {recipient} failed: "
                                     + email_result.get("error", "unknown error"))

                yield _sse({"type": "response",
                            "reply": "\n\n".join(parts) if parts else "Done.",
                            "intent": intent})
                return

            # ── fallback ────────────────────────────────────────────────────
            yield _sse({"type": "thinking", "step": "Generating response…"})
            await asyncio.sleep(0)
            prompt = build_prompt(user_text)
            reply  = await loop.run_in_executor(None, generate_response, prompt)
            yield _sse({"type": "response", "reply": reply, "intent": intent})

        except Exception as e:
            logger.error("Stream chat error: %s", e, exc_info=True)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/process-intent")
def process_intent(payload: dict):
    """
    Intelligent intent processor
    - Classifies user intent (send email, create meeting, etc.)
    - Executes appropriate workflow
    - Returns results and status
    """
    try:
        user_query = payload.get("query", "")
        
        if not user_query:
            return {"success": False, "error": "query is required"}
        
        logger.info(f"Processing intent for query: {user_query}")
        
        # Step 1: Classify intent
        classification = IntentClassifier.classify(user_query)
        logger.info(f"Classification result: intent={classification['intent']}, confidence={classification.get('confidence', 0)}")
        
        intent = classification['intent']
        parameters = classification['parameters']
        
        # For recipient_email, also support attendee_email
        if 'attendee_email' in parameters and 'recipient_email' not in parameters:
            parameters['recipient_email'] = parameters['attendee_email']
        
        # Step 2: Validate parameters
        is_valid, error_msg = IntentClassifier.validate_parameters(intent, parameters)
        if not is_valid and intent != "general_chat":
            logger.warning(f"Parameter validation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "intent": intent,
                "confidence": classification.get('confidence', 0),
                "missing_fields": [f for f in IntentClassifier.get_required_fields(intent) 
                                  if f not in parameters]
            }
        
        # Step 3: Execute workflow
        if intent == "general_chat":
            # For general chat, generate response directly
            prompt = build_prompt(user_query)
            reply = generate_response(prompt)
            return {
                "success": True,
                "intent": intent,
                "confidence": classification.get('confidence', 0),
                "reply": reply,
                "workflow_executed": False
            }
        else:
            # Execute the workflow
            execution_result = workflow_executor.execute(intent, parameters)
            
            return {
                "success": execution_result['success'],
                "intent": intent,
                "confidence": classification.get('confidence', 0),
                "tasks_executed": execution_result['tasks_executed'],
                "tasks_failed": execution_result['tasks_failed'],
                "results": execution_result['results'],
                "errors": execution_result['errors'],
                "explanation": classification.get('explanation', ''),
                "workflow_executed": True
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

        # Classify intent and execute workflow
        classification = IntentClassifier.classify(user_text)
        intent = classification['intent']
        parameters = classification['parameters']
        
        # For recipient_email, also support attendee_email
        if 'attendee_email' in parameters and 'recipient_email' not in parameters:
            parameters['recipient_email'] = parameters['attendee_email']
        
        logger.info(f"Voice command classified as: {intent}")
        
        if intent == "general_chat":
            # General chat response
            reply = generate_response(user_text)
            logger.info(f"Generated reply: {reply[:100]}...")
            
            return {
                "text": user_text,
                "reply": reply,
                "intent": intent,
                "workflow_executed": False
            }
        else:
            # Execute workflow for specific intents
            is_valid, error_msg = IntentClassifier.validate_parameters(intent, parameters)
            if not is_valid:
                logger.warning(f"Parameter validation failed: {error_msg}")
                return {
                    "text": user_text,
                    "reply": f"I need more information. {error_msg}",
                    "intent": intent,
                    "workflow_executed": False,
                    "error": error_msg
                }
            
            execution_result = workflow_executor.execute(intent, parameters)
            
            # Generate a human-friendly response
            if execution_result['success']:
                reply = f"Successfully completed: {', '.join(execution_result['tasks_executed'])}"
                if 'meet_link' in str(execution_result.get('results', {})):
                    reply += ". Meeting link has been created and sent."
            else:
                reply = f"Error: {', '.join(execution_result['tasks_failed'])}"
                if execution_result['errors']:
                    reply += f" - {list(execution_result['errors'].values())[0]}"
            
            return {
                "text": user_text,
                "reply": reply,
                "intent": intent,
                "workflow_executed": True,
                "success": execution_result['success'],
                "tasks_executed": execution_result['tasks_executed'],
                "tasks_failed": execution_result['tasks_failed'],
                "results": execution_result['results']
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
            timeout=20.0
        )
        return {"success": True, "emails": emails}
    except asyncio.TimeoutError:
        logger.error("Email fetch timed out after 20s")
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
