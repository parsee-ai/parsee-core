from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from tenacity import RetryError, Future, stop_after_attempt, wait_none

from parsee.chat.custom_dataclasses import Message, SingleImageProcessingSettings
from parsee.chat.main import RunSpecification, _run_receiver_loop, run_chat, run_chat_with_fallback
from parsee.converters.image_creation import DiskImageCreator
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.storage.interfaces import DocumentContent, Modality
from parsee.storage.in_memory_storage import InMemoryStorageManager
from parsee.storage.local_file_manager import LocalFileManager


@dataclass
class MockMlModelSpecification:
    model_id: Any

    def __hash__(self):
        return hash(self.model_id)

    def __eq__(self, other):
        if isinstance(other, MockMlModelSpecification):
            return self.model_id == other.model_id
        return NotImplemented


class MockPromptRequest:
    def __init__(self, answers):
        self.answers = answers
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        answer = self.answers[len(self.prompts) - 1]
        return answer, Decimal("1")

    def cache_info(self):
        return None


class MockModel:
    def __init__(self, spec, answers=None):
        self.spec = spec
        self.make_prompt_request = MockPromptRequest(answers or ["answer"])


class MockDocumentManager:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def load_documents(self, references, modality, search_term, max_images, show_chunk_index):
        self.calls.append((references, modality, search_term, max_images, show_chunk_index))
        return self.content


def make_run_spec(modality=Modality.TEXT, single_image_processing=None):
    return RunSpecification(
        modality=modality,
        max_load_images=None,
        single_image_processing=single_image_processing,
        most_recent_references_only=True,
        show_chunk_index=False,
    )


def test_run_chat_with_fallback(monkeypatch):
    """Should return the first successful model response, which in this case is the second model."""
    spec_successes = {MockMlModelSpecification(0): False,
                      MockMlModelSpecification(1): True,
                      MockMlModelSpecification(2): False}

    def mock_run_chat(message, message_history, document_manager, spec, run_spec):
        if spec_successes[spec]:
            return Message(text=f"Success {spec.model_id}", references=[], author=None, cost=None)
        raise RetryError(Future(0))
    monkeypatch.setattr("parsee.chat.main.run_chat", mock_run_chat)
    storage = InMemoryStorageManager(None, DiskImageCreator())
    file_manager = LocalFileManager(storage, [])

    result = run_chat_with_fallback(Message(text="Test", references=[], author=None, cost=None), [], file_manager,
                                    list(spec_successes.keys()), make_run_spec())
    assert result is not None
    assert result.text == "Success 1"


def test_run_chat_with_fallback_retries_receiver_loop(monkeypatch):
    """Should retry the full receiver loop after all models fail."""
    specs = [MockMlModelSpecification(0), MockMlModelSpecification(1)]
    calls = []

    def mock_run_chat(message, message_history, document_manager, spec, run_spec):
        calls.append(spec.model_id)
        if calls == [0, 1, 0]:
            return Message(text="Recovered", references=[], author=None, cost=None)
        raise RetryError(Future(0))

    monkeypatch.setattr("parsee.chat.main.run_chat", mock_run_chat)
    monkeypatch.setattr("parsee.chat.main._run_receiver_loop",
                        _run_receiver_loop.retry_with(stop=stop_after_attempt(3), wait=wait_none()))
    storage = InMemoryStorageManager(None, DiskImageCreator())
    file_manager = LocalFileManager(storage, [])

    result = run_chat_with_fallback(Message(text="Test", references=[], author=None, cost=None), [], file_manager,
                                    specs, make_run_spec())

    assert result is not None
    assert result.text == "Recovered"
    assert calls == [0, 1, 0]


def test_run_chat_with_fallback_does_not_retry_receiver_loop_by_default(monkeypatch):
    """Should preserve the existing single-pass default."""
    specs = [MockMlModelSpecification(0), MockMlModelSpecification(1)]
    calls = []

    def mock_run_chat(message, message_history, document_manager, spec, run_spec):
        calls.append(spec.model_id)
        raise RetryError(Future(0))

    monkeypatch.setattr("parsee.chat.main.run_chat", mock_run_chat)
    monkeypatch.setattr("parsee.chat.main._run_receiver_loop",
                        _run_receiver_loop.retry_with(stop=stop_after_attempt(1), wait=wait_none()))
    storage = InMemoryStorageManager(None, DiskImageCreator())
    file_manager = LocalFileManager(storage, [])

    result = run_chat_with_fallback(Message(text="Test", references=[], author=None, cost=None), [], file_manager,
                                    specs, make_run_spec())

    assert result is None
    assert calls == [0, 1]


def test_run_chat_flattens_combined_content_for_single_prompt(monkeypatch):
    spec = MockMlModelSpecification("model")
    model = MockModel(spec)
    monkeypatch.setattr("parsee.chat.main.get_llm_base_model", lambda _: model)
    first_image = Base64Image("image/png", "first")
    second_image = Base64Image("image/png", "second")
    document_manager = MockDocumentManager(
        DocumentContent(
            images={"doc-a": [first_image], "doc-b": [second_image]},
            texts={"doc-a": ["first text"], "doc-b": ["second text"]},
        )
    )

    result = run_chat(
        Message(text="Test", references=[], author=None, cost=None),
        [],
        document_manager,
        spec,
        make_run_spec(Modality.IMAGES_AND_TEXT),
    )

    assert result.text == "answer"
    assert model.make_prompt_request.prompts[0].images == [first_image, second_image]
    assert model.make_prompt_request.prompts[0].text == "first text\nsecond text"


def test_run_chat_passes_page_texts_to_pagewise_prompts(monkeypatch):
    spec = MockMlModelSpecification("model")
    model = MockModel(spec, ["first answer", "second answer"])
    monkeypatch.setattr("parsee.chat.main.get_llm_base_model", lambda _: model)
    first_image = Base64Image("image/png", "first")
    second_image = Base64Image("image/png", "second")
    document_manager = MockDocumentManager(
        DocumentContent(
            images={"doc-a": [first_image], "doc-b": [second_image]},
            texts={"doc-a": ["first text"], "doc-b": ["second text"]},
        )
    )
    run_spec = make_run_spec(
        Modality.IMAGES_AND_TEXT,
        SingleImageProcessingSettings(min_images_trigger=2, merge_strategy=lambda answers: "|".join(answers)),
    )

    result = run_chat(
        Message(text="Test", references=[], author=None, cost=None),
        [],
        document_manager,
        spec,
        run_spec,
    )

    assert result.text == "first answer|second answer"
    assert [prompt.images for prompt in model.make_prompt_request.prompts] == [[first_image], [second_image]]
    assert [prompt.text for prompt in model.make_prompt_request.prompts] == ["first text", "second text"]
