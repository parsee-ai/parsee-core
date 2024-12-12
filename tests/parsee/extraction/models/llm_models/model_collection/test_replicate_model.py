import logging
import sys

from replicate.exceptions import ReplicateError

from parsee import replicate_config
from parsee.chat.custom_dataclasses import Message
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.models.model_loader import get_llm_base_model


def mock_run(self, input):
    raise ReplicateError("Request was throttled. Expected available in 1 second.",)


def test__call_api_retry(monkeypatch, caplog):
    """The call to the API should be retried a specified in the settings number of times
    and at the end throw the original SDKError."""
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr("parsee.extraction.models.llm_models.model_collection.replicate_model.replicate.run",
                        mock_run)
    spec = replicate_config(replicate_api_key="123", model_name="123")
    model = get_llm_base_model(spec)
    message = Message("What is the capital of France?", [])
    prompt = Prompt(None, f"{message}", available_data="123", history=[])
    try:
        model.make_prompt_request(prompt)
    except ReplicateError:
        pass
    assert model._call_api.statistics["attempt_number"] == 2
