import ollama
import logging

logger = logging.getLogger(__name__)

def generate_response(prompt: str) -> str:
    """Generate a response from Ollama using llama3.1:8b model."""
    try:
        logger.debug(f"Sending prompt to llama3.1:8b model")
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        result = response["message"]["content"].strip()
        logger.debug(f"Received response: {result[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise
