import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.wake_word import listen_for_wake_word
from voice.stt import listen_and_transcribe
from voice.tts import speak
from backend.llm.ollama_client import generate_response

def voice_loop(send_to_ui):
    print("🎙️ Orbii Voice Service Running")

    while True:
        listen_for_wake_word()

        user_text = listen_and_transcribe()
        if not user_text.strip():
            continue

        send_to_ui({
            "role": "user",
            "content": user_text
        })

        response = generate_response(user_text)

        send_to_ui({
            "role": "assistant",
            "content": response
        })

        speak(response)
