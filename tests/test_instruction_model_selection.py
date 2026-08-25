from unittest.mock import patch

from backend.llm.model_registry import get_instruction_model


def test_mistral_selection_is_used_for_itinerary_generation():
    with patch("backend.llm.model_registry.get_active", return_value="mistral:latest"), \
         patch("backend.llm.model_registry.tool_calling_mode", return_value="native"):
        assert get_instruction_model() == "mistral:latest"


def test_llama_selection_is_used_for_itinerary_generation():
    with patch("backend.llm.model_registry.get_active", return_value="llama3.1:8b"), \
         patch("backend.llm.model_registry.tool_calling_mode", return_value="native"):
        assert get_instruction_model() == "llama3.1:8b"


def test_legacy_finetune_uses_native_instruction_fallback():
    with patch("backend.llm.model_registry.get_active", return_value="fine-tune"), \
         patch("backend.llm.model_registry.tool_calling_mode", return_value="legacy"), \
         patch.dict("backend.llm.model_registry.os.environ", {}, clear=True):
        assert get_instruction_model() == "mistral:latest"
