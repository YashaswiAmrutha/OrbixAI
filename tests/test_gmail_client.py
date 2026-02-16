import unittest
import sys
import os
from unittest.mock import patch, MagicMock, PropertyMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))


class TestGmailClientInit(unittest.TestCase):

    @patch("backend.google_service.gmail_client.build")
    @patch("backend.google_service.gmail_client.UserCredentials.from_authorized_user_file")
    @patch("os.path.exists", return_value=True)
    def test_initializes_with_token(self, mock_exists, mock_creds, mock_build):
        mock_cred_instance = MagicMock()
        mock_cred_instance.valid = True
        mock_creds.return_value = mock_cred_instance
        mock_build.return_value = MagicMock()

        from backend.google_service.gmail_client import GmailClient
        client = GmailClient.__new__(GmailClient)
        client.credentials_file = "credentials.json"
        client.token_file = "token.json"
        client.service = None
        client.calendar_service = None
        client.credentials = None
        client._initialize_service()

        self.assertIsNotNone(client.service)

    @patch("os.path.exists", return_value=False)
    def test_raises_without_credentials(self, mock_exists):
        from backend.google_service.gmail_client import GmailClient
        client = GmailClient.__new__(GmailClient)
        client.credentials_file = "missing.json"
        client.token_file = "token.json"
        client.service = None
        client.calendar_service = None
        client.credentials = None

        with self.assertRaises(Exception):
            client._initialize_service()


class TestGetLatestEmails(unittest.TestCase):

    def _make_client(self):
        from backend.google_service.gmail_client import GmailClient
        client = GmailClient.__new__(GmailClient)
        client.service = MagicMock()
        client.calendar_service = MagicMock()
        client.credentials = MagicMock()
        return client

    def test_returns_list(self):
        client = self._make_client()
        client.service.users().messages().list().execute.return_value = {"messages": []}
        result = client.get_latest_emails(5)
        self.assertIsInstance(result, list)

    def test_empty_inbox(self):
        client = self._make_client()
        client.service.users().messages().list().execute.return_value = {"messages": []}
        result = client.get_latest_emails(5)
        self.assertEqual(len(result), 0)

    def test_parses_email_data(self):
        client = self._make_client()
        client.service.users().messages().list().execute.return_value = {
            "messages": [{"id": "msg1"}]
        }
        client.service.users().messages().get().execute.return_value = {
            "payload": {
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "Date", "value": "Mon, 1 Jan 2024 00:00:00 +0000"}
                ]
            },
            "snippet": "Preview text"
        }
        result = client.get_latest_emails(5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["from"], "sender@example.com")
        self.assertEqual(result[0]["subject"], "Test Subject")
        self.assertEqual(result[0]["type"], "received")


class TestSendEmail(unittest.TestCase):

    def _make_client(self):
        from backend.google_service.gmail_client import GmailClient
        client = GmailClient.__new__(GmailClient)
        client.service = MagicMock()
        client.calendar_service = MagicMock()
        client.credentials = MagicMock()
        return client

    def test_send_returns_success(self):
        client = self._make_client()
        client.service.users().messages().send().execute.return_value = {"id": "sent123"}
        result = client.send_email("to@example.com", "Subject", "Body")
        self.assertTrue(result["success"])
        self.assertEqual(result["message_id"], "sent123")

    def test_send_returns_recipient(self):
        client = self._make_client()
        client.service.users().messages().send().execute.return_value = {"id": "s1"}
        result = client.send_email("recipient@test.com", "Sub", "Bod")
        self.assertEqual(result["to"], "recipient@test.com")

    def test_send_html_email(self):
        client = self._make_client()
        client.service.users().messages().send().execute.return_value = {"id": "h1"}
        result = client.send_email("to@a.com", "Sub", "plain", html_body="<b>bold</b>")
        self.assertTrue(result["success"])


class TestCreateGoogleMeet(unittest.TestCase):

    def _make_client(self):
        from backend.google_service.gmail_client import GmailClient
        client = GmailClient.__new__(GmailClient)
        client.service = MagicMock()
        client.calendar_service = MagicMock()
        client.credentials = MagicMock()
        return client

    def test_create_meet_returns_success(self):
        client = self._make_client()
        client.calendar_service.events().insert().execute.return_value = {
            "id": "event1",
            "conferenceData": {
                "entryPoints": [{"uri": "https://meet.google.com/abc-def-ghi"}]
            }
        }
        result = client.create_google_meet("Standup", "Daily standup", "attendee@test.com")
        self.assertTrue(result["success"])
        self.assertEqual(result["meet_link"], "https://meet.google.com/abc-def-ghi")

    def test_create_meet_returns_event_id(self):
        client = self._make_client()
        client.calendar_service.events().insert().execute.return_value = {
            "id": "ev1",
            "conferenceData": {"entryPoints": [{"uri": "https://meet.google.com/x"}]}
        }
        result = client.create_google_meet("Meeting", "", "a@b.com")
        self.assertEqual(result["event_id"], "ev1")


class TestGetEmailForContact(unittest.TestCase):

    def test_returns_none_for_unknown(self):
        from backend.google_service.gmail_client import GmailClient
        client = GmailClient.__new__(GmailClient)
        client.service = MagicMock()
        client.calendar_service = MagicMock()
        client.credentials = MagicMock()
        result = client.get_email_for_contact("Unknown Person")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
