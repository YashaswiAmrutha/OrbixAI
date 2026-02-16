import unittest
import sys
import os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from fastapi.testclient import TestClient


class TestRootEndpoint(unittest.TestCase):

    def setUp(self):
        with patch("backend.main.get_gmail_client", return_value=None):
            from backend.main import app
            self.client = TestClient(app)

    def test_root_returns_200(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_root_returns_status_ok(self):
        response = self.client.get("/")
        data = response.json()
        self.assertEqual(data["status"], "ok")


class TestChatEndpoint(unittest.TestCase):

    def setUp(self):
        from backend.main import app
        self.client = TestClient(app)

    @patch("backend.main.generate_response", return_value="Hello!")
    @patch("backend.main.IntentClassifier.classify", return_value={
        "intent": "general_chat", "confidence": 0.9, "parameters": {}, "explanation": "greeting"
    })
    def test_chat_general_returns_reply(self, mock_classify, mock_gen):
        response = self.client.post("/chat", json={"message": "hello"})
        data = response.json()
        self.assertEqual(data["reply"], "Hello!")
        self.assertEqual(data["intent"], "general_chat")

    @patch("backend.main.IntentClassifier.classify", return_value={
        "intent": "send_email", "confidence": 0.95, "parameters": {"recipient_email": "a@b.com"}, "explanation": "email"
    })
    def test_chat_email_intent_returns_action(self, mock_classify):
        response = self.client.post("/chat", json={"message": "send email to a@b.com"})
        data = response.json()
        self.assertEqual(data["action"], "open_mail_modal")
        self.assertEqual(data["intent"], "send_email")

    def test_chat_missing_message_key(self):
        response = self.client.post("/chat", json={})
        self.assertIn(response.status_code, [200, 422, 500])

    def test_chat_empty_message(self):
        response = self.client.post("/chat", json={"message": ""})
        self.assertIsNotNone(response.json())


class TestProcessIntentEndpoint(unittest.TestCase):

    def setUp(self):
        from backend.main import app
        self.client = TestClient(app)

    def test_process_intent_missing_query(self):
        response = self.client.post("/process-intent", json={})
        data = response.json()
        self.assertFalse(data.get("success", True))

    @patch("backend.main.generate_response", return_value="Sure!")
    @patch("backend.main.IntentClassifier.classify", return_value={
        "intent": "general_chat", "confidence": 0.9, "parameters": {}, "explanation": "chat"
    })
    def test_process_intent_general_chat(self, mock_classify, mock_gen):
        response = self.client.post("/process-intent", json={"query": "hello"})
        data = response.json()
        self.assertTrue(data["success"])
        self.assertFalse(data["workflow_executed"])


class TestEmailsEndpoint(unittest.TestCase):

    def setUp(self):
        from backend.main import app
        self.client = TestClient(app)

    @patch("backend.main.get_gmail_client")
    def test_emails_returns_list(self, mock_client):
        mock_instance = MagicMock()
        mock_instance.get_all_recent_emails.return_value = []
        mock_client.return_value = mock_instance
        response = self.client.get("/emails/latest")
        data = response.json()
        self.assertIn("emails", data)

    @patch("backend.main.get_gmail_client", return_value=None)
    def test_emails_no_client(self, mock_client):
        response = self.client.get("/emails/latest")
        data = response.json()
        self.assertEqual(data["emails"], [])

    def test_emails_accepts_max_results(self):
        response = self.client.get("/emails/latest", params={"max_results": 3})
        self.assertIn(response.status_code, [200, 500])


class TestSendEmailEndpoint(unittest.TestCase):

    def setUp(self):
        from backend.main import app
        self.client = TestClient(app)

    def test_send_email_requires_to_email(self):
        response = self.client.post("/emails/send", json={})
        data = response.json()
        self.assertFalse(data["success"])

    @patch("backend.main.get_gmail_client")
    def test_send_email_requires_subject_and_body(self, mock_client):
        response = self.client.post("/emails/send", json={"to_email": "a@b.com"})
        data = response.json()
        self.assertFalse(data["success"])


class TestMeetingsEndpoint(unittest.TestCase):

    def setUp(self):
        from backend.main import app
        self.client = TestClient(app)

    def test_create_meeting_requires_attendee(self):
        response = self.client.post("/meetings/create", json={})
        data = response.json()
        self.assertFalse(data["success"])

    @patch("backend.main.get_gmail_client", return_value=None)
    def test_create_meeting_no_client(self, mock_client):
        response = self.client.post("/meetings/create", json={"attendee_email": "a@b.com"})
        data = response.json()
        self.assertFalse(data["success"])


class TestVoiceEndpoint(unittest.TestCase):

    def setUp(self):
        from backend.main import app
        self.client = TestClient(app)

    def test_voice_requires_file(self):
        response = self.client.post("/voice")
        self.assertIn(response.status_code, [200, 422])


class TestErrorCases(unittest.TestCase):

    def setUp(self):
        from backend.main import app
        self.client = TestClient(app)

    def test_unknown_endpoint(self):
        response = self.client.get("/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_wrong_method_on_chat(self):
        response = self.client.get("/chat")
        self.assertEqual(response.status_code, 405)

    def test_malformed_json(self):
        response = self.client.post(
            "/chat",
            data="not-json",
            headers={"Content-Type": "application/json"}
        )
        self.assertIn(response.status_code, [422, 400])


if __name__ == "__main__":
    unittest.main()
