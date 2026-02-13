from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.llm.ollama_client import generate_response
from backend.llm.prompt import build_prompt
import whisper
import tempfile
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load whisper once
model = whisper.load_model("base")

@app.post("/chat")
def chat(message: dict):
    prompt = build_prompt(message["message"])
    reply = generate_response(prompt)
    return {"reply": reply}

@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = model.transcribe(tmp_path)
    user_text = result["text"].strip()

    if not user_text:
        return {
            "text": "",
            "reply": "I didn’t catch that. Please try speaking again."
        }

    reply = generate_response(user_text)
    return {
        "text": user_text,
        "reply": reply
    }

@app.get("/vpn_test")
def vpn_test():
    return {
        "status": "VPN OK",
        "source": "WireGuard mesh",
    }


@app.post("/sms_ingest")
def sms_ingest(payload: dict):
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
