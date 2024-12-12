import httpx
from anthropic import RateLimitError

from parsee import anthropic_config
from parsee.chat.custom_dataclasses import Message
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.models.model_loader import get_llm_base_model


class MockAnthropic:
    def __init__(self, api_key):
        self.messages = self

    def create(self, model, max_tokens, temperature, system, messages):
        raise RateLimitError("Rate limit exceeded",
                             response=httpx.Response(request=httpx.Request(method="get",
                                                                           url="https://example.com"),
                                                     status_code=429),
                             body="123")




def test__call_api_retry(monkeypatch):
    """The call to the API should be retried a specified in the settings number of times
    and at the end throw the original RateLimitError."""
    monkeypatch.setattr("parsee.extraction.models.llm_models.model_collection.anthropic_model.anthropic.Anthropic",
                        MockAnthropic)
    spec = anthropic_config(anthropic_api_key="123", model_name="123")
    model = get_llm_base_model(spec)
    message = Message("What is the capital of France?", [])
    prompt = Prompt(None, f"{message}", available_data="123", history=[])
    try:
        model.make_prompt_request(prompt)
    except RateLimitError:
        pass
    assert model._call_api.statistics["attempt_number"] == 2
