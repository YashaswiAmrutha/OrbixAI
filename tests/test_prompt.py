import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from backend.llm.prompt import build_prompt


class TestBuildPrompt(unittest.TestCase):

    def test_returns_string(self):
        result = build_prompt("hello")
        self.assertIsInstance(result, str)

    def test_contains_user_message(self):
        result = build_prompt("what is Python")
        self.assertIn("what is Python", result)

    def test_contains_system_identity(self):
        result = build_prompt("hello")
        self.assertIn("OrbixAI", result)

    def test_contains_personality_section(self):
        result = build_prompt("hello")
        self.assertIn("PERSONALITY", result)

    def test_contains_capabilities_section(self):
        result = build_prompt("hello")
        self.assertIn("CAPABILITIES", result)

    def test_contains_guidelines_section(self):
        result = build_prompt("hello")
        self.assertIn("GUIDELINES", result)

    def test_ends_with_orbixai_label(self):
        result = build_prompt("test")
        self.assertTrue(result.strip().endswith("OrbixAI:"))

    def test_different_messages_produce_different_prompts(self):
        p1 = build_prompt("hello")
        p2 = build_prompt("goodbye")
        self.assertNotEqual(p1, p2)

    def test_empty_message(self):
        result = build_prompt("")
        self.assertIsInstance(result, str)
        self.assertIn("OrbixAI", result)

    def test_special_characters(self):
        result = build_prompt("what's 2+2? <script>alert('x')</script>")
        self.assertIn("what's 2+2?", result)

    def test_multiline_message(self):
        result = build_prompt("line1\nline2\nline3")
        self.assertIn("line1", result)
        self.assertIn("line3", result)


if __name__ == "__main__":
    unittest.main()
