import unittest
import sys
import os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from backend.llm.ollama_client import generate_response


class TestGenerateResponse(unittest.TestCase):

    @patch("backend.llm.ollama_client.ollama")
    def test_returns_string(self, mock_ollama):
        mock_ollama.chat.return_value = {"message": {"content": "Hello there"}}
        result = generate_response("test prompt")
        self.assertIsInstance(result, str)

    @patch("backend.llm.ollama_client.ollama")
    def test_returns_stripped_content(self, mock_ollama):
        mock_ollama.chat.return_value = {"message": {"content": "  response text  "}}
        result = generate_response("prompt")
        self.assertEqual(result, "response text")

    @patch("backend.llm.ollama_client.ollama")
    def test_calls_ollama_chat(self, mock_ollama):
        mock_ollama.chat.return_value = {"message": {"content": "ok"}}
        generate_response("test")
        mock_ollama.chat.assert_called_once()

    @patch("backend.llm.ollama_client.ollama")
    def test_uses_correct_model(self, mock_ollama):
        mock_ollama.chat.return_value = {"message": {"content": "ok"}}
        generate_response("test")
        call_args = mock_ollama.chat.call_args
        self.assertEqual(call_args.kwargs.get("model", call_args[1].get("model", None)) or call_args[0][0] if call_args[0] else None, None)
        actual_call = mock_ollama.chat.call_args
        self.assertIn("llama3.1:8b", str(actual_call))

    @patch("backend.llm.ollama_client.ollama")
    def test_passes_prompt_as_user_message(self, mock_ollama):
        mock_ollama.chat.return_value = {"message": {"content": "response"}}
        generate_response("my prompt text")
        call_args = mock_ollama.chat.call_args
        messages = call_args[1].get("messages") if call_args[1] else None
        if messages is None and len(call_args[0]) > 1:
            messages = call_args[0][1]
        self.assertIsNotNone(messages)

    @patch("backend.llm.ollama_client.ollama")
    def test_raises_on_connection_error(self, mock_ollama):
        mock_ollama.chat.side_effect = ConnectionError("Ollama not running")
        with self.assertRaises(ConnectionError):
            generate_response("test")

    @patch("backend.llm.ollama_client.ollama")
    def test_raises_on_generic_exception(self, mock_ollama):
        mock_ollama.chat.side_effect = Exception("unexpected error")
        with self.assertRaises(Exception):
            generate_response("test")

    @patch("backend.llm.ollama_client.ollama")
    def test_handles_empty_response(self, mock_ollama):
        mock_ollama.chat.return_value = {"message": {"content": ""}}
        result = generate_response("test")
        self.assertEqual(result, "")

    @patch("backend.llm.ollama_client.ollama")
    def test_handles_multiline_response(self, mock_ollama):
        mock_ollama.chat.return_value = {"message": {"content": "line1\nline2\nline3"}}
        result = generate_response("test")
        self.assertIn("\n", result)


if __name__ == "__main__":
    unittest.main()
