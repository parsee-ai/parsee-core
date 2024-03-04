from decimal import Decimal
from typing import *

from parsee.extraction.extractor_elements import StandardDocumentFormat, FinalOutputTableColumn, ParseeAnswer
from parsee.utils.enums import DocumentType
from parsee.templates.job_template import JobTemplate
from parsee.extraction.extractor_dataclasses import ParseeBucket
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.datasets.readers.interfaces import ModelReader
from parsee.storage.vector_stores.interfaces import VectorStore


class StorageManager:

    vector_store: VectorStore

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def db_values_template(self, job_template: JobTemplate, strict: bool) -> JobTemplate:
        raise NotImplemented

    def log_expense(self, service: str, amount: Decimal, class_id: str):
        raise NotImplemented

    def get_available_models(self) -> List[MlModelSpecification]:
        raise NotImplemented
