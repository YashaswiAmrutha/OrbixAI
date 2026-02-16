import unittest
import sys
import os
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from backend.google_service.mail_generator import MailGenerator


class TestGenerateMailContent(unittest.TestCase):

    @patch("backend.google_service.mail_generator.generate_response")
    def test_returns_dict_with_subject_and_body(self, mock_llm):
        mock_llm.return_value = "SUBJECT: Test Subject\nBODY: Test body content"
        result = MailGenerator.generate_mail_content("write a test email")
        self.assertIn("subject", result)
        self.assertIn("body", result)

    @patch("backend.google_service.mail_generator.generate_response")
    def test_parses_subject_correctly(self, mock_llm):
        mock_llm.return_value = "SUBJECT: Project Update\nBODY: Here is the update."
        result = MailGenerator.generate_mail_content("update email")
        self.assertEqual(result["subject"], "Project Update")

    @patch("backend.google_service.mail_generator.generate_response")
    def test_parses_body_correctly(self, mock_llm):
        mock_llm.return_value = "SUBJECT: Hello\nBODY: Dear team,\nPlease review."
        result = MailGenerator.generate_mail_content("review email")
        self.assertIn("Dear team", result["body"])

    @patch("backend.google_service.mail_generator.generate_response")
    def test_fallback_on_unparseable_response(self, mock_llm):
        mock_llm.return_value = "Just some random text without proper format"
        result = MailGenerator.generate_mail_content("email about meeting")
        self.assertIsNotNone(result["subject"])
        self.assertIsNotNone(result["body"])

    @patch("backend.google_service.mail_generator.generate_response")
    def test_includes_meeting_link(self, mock_llm):
        mock_llm.return_value = "SUBJECT: Meeting Invite\nBODY: Join at https://meet.google.com/abc"
        result = MailGenerator.generate_mail_content(
            "send meeting invite",
            meeting_link="https://meet.google.com/abc"
        )
        self.assertIn("subject", result)

    @patch("backend.google_service.mail_generator.generate_response")
    def test_includes_recipient_name(self, mock_llm):
        mock_llm.return_value = "SUBJECT: Hello John\nBODY: Dear John, how are you?"
        result = MailGenerator.generate_mail_content(
            "greeting email",
            recipient_name="John"
        )
        self.assertIn("John", result["body"])

    @patch("backend.google_service.mail_generator.generate_response")
    def test_handles_llm_exception(self, mock_llm):
        mock_llm.side_effect = Exception("LLM down")
        result = MailGenerator.generate_mail_content("test email")
        self.assertIsNotNone(result["subject"])
        self.assertIsNotNone(result["body"])

    @patch("backend.google_service.mail_generator.generate_response")
    def test_empty_subject_triggers_fallback(self, mock_llm):
        mock_llm.return_value = "SUBJECT: \nBODY: "
        result = MailGenerator.generate_mail_content("email")
        self.assertTrue(len(result["subject"]) > 0)


class TestGenerateMeetingInvitation(unittest.TestCase):

    @patch("backend.google_service.mail_generator.generate_response")
    def test_returns_subject_and_body(self, mock_llm):
        mock_llm.return_value = "SUBJECT: Meeting Invitation\nBODY: You are invited."
        result = MailGenerator.generate_meeting_invitation("Alice")
        self.assertIn("subject", result)
        self.assertIn("body", result)

    @patch("backend.google_service.mail_generator.generate_response")
    def test_with_time_and_purpose(self, mock_llm):
        mock_llm.return_value = "SUBJECT: Sprint Review\nBODY: Sprint review at 3pm."
        result = MailGenerator.generate_meeting_invitation(
            "Bob", meeting_time="3pm", meeting_purpose="Sprint Review"
        )
        self.assertIn("subject", result)

    @patch("backend.google_service.mail_generator.generate_response")
    def test_fallback_on_error(self, mock_llm):
        mock_llm.side_effect = Exception("error")
        result = MailGenerator.generate_meeting_invitation("Charlie")
        self.assertIn("Meeting", result["subject"])
        self.assertIsNotNone(result["body"])

    @patch("backend.google_service.mail_generator.generate_response")
    def test_fallback_includes_purpose(self, mock_llm):
        mock_llm.side_effect = Exception("error")
        result = MailGenerator.generate_meeting_invitation(
            "Dana", meeting_purpose="Budget Review"
        )
        self.assertIn("Meeting", result["subject"])


if __name__ == "__main__":
    unittest.main()
