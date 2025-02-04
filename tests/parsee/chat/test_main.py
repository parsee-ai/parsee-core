from dataclasses import dataclass
from typing import Any

from tenacity import RetryError, Future

from parsee.chat.custom_dataclasses import Message
from parsee.chat.main import run_chat_with_fallback
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

