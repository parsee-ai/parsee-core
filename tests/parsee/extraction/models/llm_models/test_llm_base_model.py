from parsee import ollama_config
from parsee.chat.custom_dataclasses import Message
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.models.model_loader import get_llm_base_model


class MockOllamaClient:
    def __init__(self, host):
        pass

    def chat(self, model, messages):
        return {"message": {"content": "123"}}


def test_make_prompt_request__with_cache(monkeypatch):
    """The prompt should be cached after the first call."""
    monkeypatch.setattr("parsee.extraction.models.llm_models.model_collection.ollama_model.Client",
                        MockOllamaClient)

    model = get_llm_base_model(ollama_config("llama3"))
    model.make_prompt_request.cache_clear()
    message = Message("What is the capital of France?", [])
    prompt = Prompt(None, f"{message}", available_data="123", history=[])
    model.make_prompt_request(prompt)
    prompt = Prompt(None, f"{message}", available_data="123", history=[])
    model.make_prompt_request(prompt)
    assert model.make_prompt_request.cache_info().hits == 1


def test_make_prompt_request__with_cache_list_argument(monkeypatch):
    """The prompt with a list argument should be cached after the first call."""
    monkeypatch.setattr("parsee.extraction.models.llm_models.model_collection.ollama_model.Client",
                        MockOllamaClient)
    model = get_llm_base_model(ollama_config(model_name="llama3"))
    model.make_prompt_request.cache_clear()
    message = Message("What is the capital of France?", [])
    prompt = Prompt(None, f"{message}", available_data=[Base64Image("jpeg", "abc")], history=[])
    model.make_prompt_request(prompt)
    prompt = Prompt(None, f"{message}", available_data=[Base64Image("jpeg", "abc")], history=[])
    model.make_prompt_request(prompt)
    assert model.make_prompt_request.cache_info().hits == 1
