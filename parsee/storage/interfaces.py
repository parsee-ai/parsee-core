from decimal import Decimal
from typing import *

from parsee.templates.job_template import JobTemplate
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.storage.vector_stores.interfaces import VectorStore
from parsee.extraction.extractor_elements import FileReference, ExtractedEl
from parsee.converters.image_creation import ImageCreator
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.utils.enums import SearchStrategy
from parsee.chat.custom_dataclasses import ChatSettings


class StorageManager:

    vector_store: VectorStore
    image_creator: ImageCreator

    def __init__(self, vector_store: VectorStore, image_creator: ImageCreator):
        self.vector_store = vector_store
        self.image_creator = image_creator

    def db_values_template(self, job_template: JobTemplate, strict: bool) -> JobTemplate:
        raise NotImplementedError

    def log_expense(self, service: str, amount: Decimal, class_id: str):
        raise NotImplementedError

    def get_available_models(self) -> List[MlModelSpecification]:
        raise NotImplementedError


class DocumentManager:

    storage: StorageManager

    def __init__(self, storage: StorageManager, settings: ChatSettings):
        self.storage = storage
        self.settings = settings

    def load_documents(self, references: List[FileReference], multimodal: bool, search_term: Optional[str], max_images: Optional[int], max_tokens: Optional[int]) -> Union[str, List[Base64Image]]:
        raise NotImplementedError
