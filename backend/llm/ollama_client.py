import logging
from llm.hf_client import chat_response, ORCHESTRATOR_MODEL, FALLBACK_MODEL

logger = logging.getLogger(__name__)


def generate_response(prompt: str) -> str:
    """Generate a conversational reply via the local Ollama fine-tuned model."""
    return chat_response(prompt)


def is_available() -> bool:
    """Check if Ollama is reachable."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False
