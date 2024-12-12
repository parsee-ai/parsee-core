import httpx
from mistralai import SDKError

from parsee import mistral_api_config
from parsee.chat.custom_dataclasses import Message
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.models.model_loader import get_llm_base_model


class MockMistral:
    def __init__(self, api_key):
        self.chat = self

    def complete(self, model, messages, temperature):
        raise SDKError("Rate limit exceeded",
                       status_code=429,
                             raw_response=httpx.Response(request=httpx.Request(method="get",
                                                                           url="https://example.com"),
                                                     status_code=429),
                             body="123")




def test__call_api_retry(monkeypatch):
    """The call to the API should be retried a specified in the settings number of times
    and at the end throw the original SDKError."""
    monkeypatch.setattr("parsee.extraction.models.llm_models.model_collection.mistral_model.Mistral",
                        MockMistral)
    spec = mistral_api_config(mistral_api_key="123", model_name="123")
    model = get_llm_base_model(spec)
    message = Message("What is the capital of France?", [])
    prompt = Prompt(None, f"{message}", available_data="123", history=[])
    try:
        model.make_prompt_request(prompt)
    except SDKError:
        pass
    assert model._call_api.statistics["attempt_number"] == 2
