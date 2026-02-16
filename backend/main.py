from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from llm.ollama_client import generate_response
from llm.prompt import build_prompt
from google_service.gmail_client import GmailClient
from google_service.mail_generator import MailGenerator
from intent_workflow import IntentClassifier, WorkflowExecutor, WorkflowTask
from faster_whisper import WhisperModel
import tempfile
from datetime import datetime
import logging
import os

# Force CPU-only mode to avoid CUDA library issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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
    global gmail_client
    if gmail_client is None:
        try:
            gmail_client = GmailClient()
            logger.info("Gmail client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gmail client: {str(e)}")
            return None
    return gmail_client


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


def task_send_email(recipient_email: str, subject: str = "", body: str = "", use_llm: bool = False, 
                    user_prompt: str = "", recipient_name: str = "", meeting_link: str = None, **kwargs) -> dict:
    """Task: Send Email"""
    try:
        client = get_gmail_client()
        if not client:
            raise Exception("Gmail client not initialized")
        
        logger.info(f"Task: Sending email to {recipient_email}")
        
        # Generate content using LLM if requested
        if use_llm and user_prompt:
            logger.info("Generating email content with LLM")
            mail_content = MailGenerator.generate_mail_content(
                user_prompt,
                recipient_name=recipient_name,
                meeting_link=meeting_link
            )
            subject = mail_content['subject']
            body = mail_content['body']
        
        if not subject or not body:
            raise Exception("subject and body are required")
        
        logger.info(f"Sending email with subject: {subject}")
        result = client.send_email(recipient_email, subject, body)
        
        if not result['success']:
            raise Exception(f"Failed to send email: {result.get('error', 'Unknown error')}")
        
        logger.info(f"Email sent successfully to {recipient_email}")
        return {
            "email_sent": True,
            "recipient": recipient_email,
            "message_id": result.get('message_id')
        }
    except Exception as e:
        logger.error(f"Error in task_send_email: {str(e)}")
        raise


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
    
    # Workflow 6: General chat (no special workflow)
    workflow_executor.register_workflow("general_chat", [])
    
    logger.info("All workflows registered successfully")


# Initialize workflows on startup
register_workflows()

@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/health")
def health_check():
    return {"message": "Orbii Backend API is running", "status": "ok"}

@app.post("/chat")
def chat(message: dict):
    try:
        user_text = message["message"]

        classification = IntentClassifier.classify(user_text)
        intent = classification["intent"]
        parameters = classification.get("parameters", {})

        if intent == "general_chat":
            prompt = build_prompt(user_text)
            reply = generate_response(prompt)
            return {"reply": reply, "intent": intent}

        if intent in ("send_email", "meeting_and_email", "create_meeting", "schedule_meeting"):
            return {
                "reply": "Opening the email form for you.",
                "intent": intent,
                "parameters": parameters,
                "action": "open_mail_modal"
            }

        if intent == "get_emails":
            result = task_get_emails(max_results=parameters.get("max_results", 10))
            return {
                "reply": f"Fetched {result['count']} emails.",
                "intent": intent,
                "emails": result["emails"]
            }

        prompt = build_prompt(user_text)
        reply = generate_response(prompt)
        return {"reply": reply, "intent": intent}
    except Exception as e:
        return {"error": str(e), "reply": "Sorry, I encountered an error. Is Ollama running?"}


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
        if not client:
            return {"error": "Gmail client not initialized", "emails": []}
        
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
        logger.error(f"Error fetching emails: {str(e)}")
        return {"success": False, "error": str(e), "emails": []}


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
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
