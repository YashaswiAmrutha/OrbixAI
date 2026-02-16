import unittest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from backend.intent_workflow.intent_classifier import IntentClassifier


class TestIntentDefinitions(unittest.TestCase):

    def test_intents_dict_exists(self):
        self.assertIsInstance(IntentClassifier.INTENTS, dict)

    def test_intents_contains_send_email(self):
        self.assertIn("send_email", IntentClassifier.INTENTS)

    def test_intents_contains_create_meeting(self):
        self.assertIn("create_meeting", IntentClassifier.INTENTS)

    def test_intents_contains_meeting_and_email(self):
        self.assertIn("meeting_and_email", IntentClassifier.INTENTS)

    def test_intents_contains_schedule_meeting(self):
        self.assertIn("schedule_meeting", IntentClassifier.INTENTS)

    def test_intents_contains_get_emails(self):
        self.assertIn("get_emails", IntentClassifier.INTENTS)

    def test_intents_contains_general_chat(self):
        self.assertIn("general_chat", IntentClassifier.INTENTS)

    def test_each_intent_has_required_fields(self):
        for intent_name, intent_data in IntentClassifier.INTENTS.items():
            self.assertIn("required_fields", intent_data)
            self.assertIsInstance(intent_data["required_fields"], list)

    def test_each_intent_has_description(self):
        for intent_name, intent_data in IntentClassifier.INTENTS.items():
            self.assertIn("description", intent_data)
            self.assertIsInstance(intent_data["description"], str)


class TestGetRequiredFields(unittest.TestCase):

    def test_send_email_requires_recipient(self):
        fields = IntentClassifier.get_required_fields("send_email")
        self.assertIn("recipient_email", fields)

    def test_create_meeting_requires_title(self):
        fields = IntentClassifier.get_required_fields("create_meeting")
        self.assertIn("event_title", fields)

    def test_meeting_and_email_requires_attendee_and_title(self):
        fields = IntentClassifier.get_required_fields("meeting_and_email")
        self.assertIn("attendee_email", fields)
        self.assertIn("event_title", fields)

    def test_general_chat_requires_nothing(self):
        fields = IntentClassifier.get_required_fields("general_chat")
        self.assertEqual(fields, [])

    def test_get_emails_requires_nothing(self):
        fields = IntentClassifier.get_required_fields("get_emails")
        self.assertEqual(fields, [])

    def test_unknown_intent_returns_empty(self):
        fields = IntentClassifier.get_required_fields("nonexistent")
        self.assertEqual(fields, [])


class TestValidateParameters(unittest.TestCase):

    def test_valid_send_email_parameters(self):
        is_valid, msg = IntentClassifier.validate_parameters(
            "send_email", {"recipient_email": "test@example.com"}
        )
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_missing_send_email_parameters(self):
        is_valid, msg = IntentClassifier.validate_parameters("send_email", {})
        self.assertFalse(is_valid)
        self.assertIn("recipient_email", msg)

    def test_valid_meeting_and_email_parameters(self):
        is_valid, msg = IntentClassifier.validate_parameters(
            "meeting_and_email",
            {"attendee_email": "test@example.com", "event_title": "Standup"}
        )
        self.assertTrue(is_valid)

    def test_partial_meeting_and_email_parameters(self):
        is_valid, msg = IntentClassifier.validate_parameters(
            "meeting_and_email", {"attendee_email": "test@example.com"}
        )
        self.assertFalse(is_valid)

    def test_general_chat_always_valid(self):
        is_valid, msg = IntentClassifier.validate_parameters("general_chat", {})
        self.assertTrue(is_valid)

    def test_empty_string_treated_as_missing(self):
        is_valid, msg = IntentClassifier.validate_parameters(
            "send_email", {"recipient_email": ""}
        )
        self.assertFalse(is_valid)


class TestClassifyWithMock(unittest.TestCase):

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_returns_dict(self, mock_llm):
        mock_llm.return_value = '{"intent": "general_chat", "confidence": 0.9, "parameters": {}, "explanation": "greeting"}'
        result = IntentClassifier.classify("hello")
        self.assertIsInstance(result, dict)

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_has_intent_key(self, mock_llm):
        mock_llm.return_value = '{"intent": "general_chat", "confidence": 0.9, "parameters": {}, "explanation": "greeting"}'
        result = IntentClassifier.classify("hello")
        self.assertIn("intent", result)

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_has_parameters_key(self, mock_llm):
        mock_llm.return_value = '{"intent": "send_email", "confidence": 0.95, "parameters": {"recipient_email": "test@example.com"}, "explanation": "send email"}'
        result = IntentClassifier.classify("send email to test@example.com")
        self.assertIn("parameters", result)

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_email_intent(self, mock_llm):
        mock_llm.return_value = '{"intent": "send_email", "confidence": 0.95, "parameters": {"recipient_email": "user@example.com"}, "explanation": "user wants to send email"}'
        result = IntentClassifier.classify("send email to user@example.com")
        self.assertEqual(result["intent"], "send_email")

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_meeting_intent(self, mock_llm):
        mock_llm.return_value = '{"intent": "create_meeting", "confidence": 0.9, "parameters": {"event_title": "standup"}, "explanation": "create meeting"}'
        result = IntentClassifier.classify("create a meeting")
        self.assertEqual(result["intent"], "create_meeting")

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_filters_empty_parameters(self, mock_llm):
        mock_llm.return_value = '{"intent": "send_email", "confidence": 0.9, "parameters": {"recipient_email": "a@b.com", "subject": "", "body": null}, "explanation": "send"}'
        result = IntentClassifier.classify("send email to a@b.com")
        self.assertNotIn("subject", result["parameters"])

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_invalid_intent_defaults_general_chat(self, mock_llm):
        mock_llm.return_value = '{"intent": "unknown_intent", "confidence": 0.5, "parameters": {}, "explanation": "unknown"}'
        result = IntentClassifier.classify("do something weird")
        self.assertEqual(result["intent"], "general_chat")

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_handles_markdown_wrapped_json(self, mock_llm):
        mock_llm.return_value = '```json\n{"intent": "general_chat", "confidence": 0.8, "parameters": {}, "explanation": "chat"}\n```'
        result = IntentClassifier.classify("hello")
        self.assertEqual(result["intent"], "general_chat")

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_handles_malformed_json(self, mock_llm):
        mock_llm.return_value = "this is not json at all"
        result = IntentClassifier.classify("hello")
        self.assertEqual(result["intent"], "general_chat")
        self.assertLessEqual(result["confidence"], 0.5)

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_handles_llm_exception(self, mock_llm):
        mock_llm.side_effect = Exception("connection refused")
        result = IntentClassifier.classify("hello")
        self.assertEqual(result["intent"], "general_chat")

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_confidence_in_range(self, mock_llm):
        mock_llm.return_value = '{"intent": "send_email", "confidence": 0.95, "parameters": {"recipient_email": "a@b.com"}, "explanation": "send"}'
        result = IntentClassifier.classify("send email")
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)

    @patch("backend.intent_workflow.intent_classifier.generate_response")
    def test_classify_get_emails_intent(self, mock_llm):
        mock_llm.return_value = '{"intent": "get_emails", "confidence": 0.9, "parameters": {"max_results": 5}, "explanation": "fetch emails"}'
        result = IntentClassifier.classify("show my emails")
        self.assertEqual(result["intent"], "get_emails")


if __name__ == "__main__":
    unittest.main()
