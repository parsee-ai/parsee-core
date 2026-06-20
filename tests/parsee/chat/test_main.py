from dataclasses import dataclass
from typing import Any

from tenacity import RetryError, Future, stop_after_attempt

from parsee.chat.custom_dataclasses import Message
from parsee.chat.main import _run_receiver_loop, run_chat_with_fallback
from parsee.converters.image_creation import DiskImageCreator
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

def test_run_chat_with_fallback(monkeypatch):
    """Should return the first successful model response, which in this case is the second model."""
    spec_successes = {MockMlModelSpecification(0): False,
                      MockMlModelSpecification(1): True,
                      MockMlModelSpecification(2): False}
    def mock_run_chat(message, message_history, document_manager, spec, most_recent_references_only, show_chunk_index,
                      single_page_processing_max_images_trigger):
        if spec_successes[spec]:
            return [Message(text=f"Success {spec.model_id}", references=[], author=None, cost=None)]
        raise RetryError(Future(0))
    monkeypatch.setattr("parsee.chat.main.run_chat", mock_run_chat)
    storage = InMemoryStorageManager(None, DiskImageCreator())
    file_manager = LocalFileManager(storage, [])

    result = run_chat_with_fallback(Message(text="Test", references=[], author=None, cost=None), [], file_manager,
                           spec_successes, False)
    assert result[0].text == "Success 1"


def test_run_chat_with_fallback_retries_receiver_loop(monkeypatch):
    """Should retry the full receiver loop after all models fail."""
    specs = [MockMlModelSpecification(0), MockMlModelSpecification(1)]
    calls = []

    def mock_run_chat(message, message_history, document_manager, spec, most_recent_references_only, show_chunk_index,
                      single_page_processing_max_images_trigger):
        calls.append(spec.model_id)
        if calls == [0, 1, 0]:
            return [Message(text="Recovered", references=[], author=None, cost=None)]
        raise RetryError(Future(0))

    monkeypatch.setattr("parsee.chat.main.run_chat", mock_run_chat)
    monkeypatch.setattr("parsee.chat.main._run_receiver_loop",
                        _run_receiver_loop.retry_with(stop=stop_after_attempt(3)))
    storage = InMemoryStorageManager(None, DiskImageCreator())
    file_manager = LocalFileManager(storage, [])

    result = run_chat_with_fallback(Message(text="Test", references=[], author=None, cost=None), [], file_manager,
                                    specs, False)

    assert result[0].text == "Recovered"
    assert calls == [0, 1, 0]


def test_run_chat_with_fallback_does_not_retry_receiver_loop_by_default(monkeypatch):
    """Should preserve the existing single-pass default."""
    specs = [MockMlModelSpecification(0), MockMlModelSpecification(1)]
    calls = []

    def mock_run_chat(message, message_history, document_manager, spec, most_recent_references_only, show_chunk_index,
                      single_page_processing_max_images_trigger):
        calls.append(spec.model_id)
        raise RetryError(Future(0))

    monkeypatch.setattr("parsee.chat.main.run_chat", mock_run_chat)
    monkeypatch.setattr("parsee.chat.main._run_receiver_loop",
                        _run_receiver_loop.retry_with(stop=stop_after_attempt(1)))
    storage = InMemoryStorageManager(None, DiskImageCreator())
    file_manager = LocalFileManager(storage, [])

    result = run_chat_with_fallback(Message(text="Test", references=[], author=None, cost=None), [], file_manager,
                                    specs, False)

    assert result == []
    assert calls == [0, 1]
