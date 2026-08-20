import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.google_service.travel_services import (
    _fallback_day_sections,
    generate_itinerary,
    _itinerary_day_numbers,
)


def test_detects_markdown_and_plain_day_headings():
    text = "# Day 1: Arrival\n\n**Day 2 - Museums**\n\nDay 3: Departure"
    assert _itinerary_day_numbers(text) == {1, 2, 3}


def test_fallback_can_fill_only_the_missing_days():
    sections = _fallback_day_sections(
        "Paris",
        [{"name": "Louvre"}, {"name": "Eiffel Tower"}],
        4,
        {3, 4},
    )
    assert _itinerary_day_numbers(sections) == {3, 4}
    assert "Day 1" not in sections
    assert "Day 2" not in sections


def test_truncated_model_output_is_completed_before_returning():
    chat = Mock(side_effect=[
        {"message": {"content": "Day 1: Arrival\n- Explore downtown"}},
        {"message": {"content": "Day 2: Museums\n- Visit the main museum"}},
    ])
    fake_ollama = SimpleNamespace(chat=chat)
    attractions = [
        {"name": "Louvre", "category": "museum"},
        {"name": "Eiffel Tower", "category": "landmark"},
    ]
    with patch.dict(sys.modules, {"ollama": fake_ollama}), patch(
        "llm.model_registry.get_instruction_model", return_value="mistral:latest"
    ):
        result = generate_itinerary(
            "Paris", attractions, 4, "2026-08-27"
        )

    assert _itinerary_day_numbers(result) == {1, 2, 3, 4}
    assert chat.call_count == 2
