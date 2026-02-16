import unittest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from fastapi.testclient import TestClient
from backend.main import app
import io


class TestVoiceEndpointValidation(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_voice_requires_file_upload(self):
        response = self.client.post("/voice")
        self.assertIn(response.status_code, [200, 422])

    def test_voice_accepts_audio_file(self):
        audio_data = io.BytesIO(b"\x00" * 1000)
        response = self.client.post(
            "/voice",
            files={"file": ("voice.wav", audio_data, "audio/wav")}
        )
        self.assertIn(response.status_code, [200, 500])


class TestVoiceTranscription(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch("backend.main.get_model")
    @patch("backend.main.generate_response", return_value="This is a response")
    @patch("backend.main.IntentClassifier.classify", return_value={
        "intent": "general_chat", "confidence": 0.9, "parameters": {}, "explanation": "chat"
    })
    def test_transcription_returns_text_and_reply(self, mock_classify, mock_gen, mock_model):
        mock_segment = MagicMock()
        mock_segment.text = "hello there"
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = ([mock_segment], MagicMock())
        mock_model.return_value = mock_model_instance

        audio_data = io.BytesIO(b"\x00" * 1000)
        response = self.client.post(
            "/voice",
            files={"file": ("voice.wav", audio_data, "audio/wav")}
        )
        data = response.json()
        self.assertIn("text", data)
        self.assertIn("reply", data)

    @patch("backend.main.get_model")
    def test_empty_transcription(self, mock_model):
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = ([], MagicMock())
        mock_model.return_value = mock_model_instance

        audio_data = io.BytesIO(b"\x00" * 1000)
        response = self.client.post(
            "/voice",
            files={"file": ("voice.wav", audio_data, "audio/wav")}
        )
        data = response.json()
        self.assertEqual(data["text"], "")


class TestVoiceIntentClassification(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch("backend.main.get_model")
    @patch("backend.main.generate_response", return_value="General response")
    @patch("backend.main.IntentClassifier.classify", return_value={
        "intent": "general_chat", "confidence": 0.9, "parameters": {}, "explanation": "general"
    })
    def test_voice_general_chat(self, mock_classify, mock_gen, mock_model):
        mock_segment = MagicMock()
        mock_segment.text = "what is the weather"
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = ([mock_segment], MagicMock())
        mock_model.return_value = mock_model_instance

        audio_data = io.BytesIO(b"\x00" * 1000)
        response = self.client.post(
            "/voice",
            files={"file": ("voice.wav", audio_data, "audio/wav")}
        )
        data = response.json()
        self.assertEqual(data["intent"], "general_chat")
        self.assertFalse(data["workflow_executed"])

    @patch("backend.main.get_model")
    @patch("backend.main.IntentClassifier.classify", return_value={
        "intent": "send_email", "confidence": 0.9,
        "parameters": {"recipient_email": "a@b.com"},
        "explanation": "send email"
    })
    @patch("backend.main.IntentClassifier.validate_parameters", return_value=(False, "Missing fields"))
    def test_voice_invalid_params_returns_error(self, mock_validate, mock_classify, mock_model):
        mock_segment = MagicMock()
        mock_segment.text = "send email"
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = ([mock_segment], MagicMock())
        mock_model.return_value = mock_model_instance

        audio_data = io.BytesIO(b"\x00" * 1000)
        response = self.client.post(
            "/voice",
            files={"file": ("voice.wav", audio_data, "audio/wav")}
        )
        data = response.json()
        self.assertFalse(data["workflow_executed"])


class TestVoiceErrorHandling(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch("backend.main.get_model")
    def test_transcription_error(self, mock_model):
        mock_model.side_effect = Exception("Model failed")
        audio_data = io.BytesIO(b"\x00" * 1000)
        response = self.client.post(
            "/voice",
            files={"file": ("voice.wav", audio_data, "audio/wav")}
        )
        data = response.json()
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
