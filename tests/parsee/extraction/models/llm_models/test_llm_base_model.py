from parsee import ollama_config
from parsee.chat.custom_dataclasses import Message
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.models.model_loader import get_llm_base_model


def test_make_prompt_request__with_cache():
    """The prompt should be cached after the first call."""
    model = get_llm_base_model(ollama_config("llama3"))
    message = Message("What is the capital of France?", [])
    prompt = Prompt(None, f"{message}", available_data="123", history=[])
    model.make_prompt_request(prompt)
    prompt = Prompt(None, f"{message}", available_data="123", history=[])
    model.make_prompt_request(prompt)
    assert model.make_prompt_request.cache_info().hits == 1
