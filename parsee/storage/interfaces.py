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

    def get_raw_document(self, source_type: DocumentType, source_identifier: str) -> str:
        raise NotImplemented

    def convert_doc(self, source_type: DocumentType, source_identifier: str, entry_id: int):
        raise NotImplemented

    def get_doc(self, source_type: DocumentType, source_identifier: str) -> StandardDocumentFormat:
        raise NotImplemented

    def db_values_template(self, job_template: JobTemplate, strict: bool) -> JobTemplate:
        raise NotImplemented

    def store_mappings(self, source_type: DocumentType, source_identifier: str, mappings: List[ParseeBucket]):
        raise NotImplemented

    def store_final_output(self, source_type: DocumentType, source_identifier: str, template: JobTemplate,
                           output_tabular: List[FinalOutputTableColumn], output_text: List[ParseeAnswer]):
        raise NotImplemented

    def start_job(self):
        raise NotImplemented

    def finish_job(self):
        raise NotImplemented

    def get_job_id(self) -> int:
        raise NotImplemented

    def stop_job_with_error(self, error_msg: str):
        raise NotImplemented

    def log_callback(self, url: str, data: any, response: any, status_code: int):
        raise NotImplemented

    def log_expense(self, service: str, amount: Decimal, class_id: str):
        raise NotImplemented

    def get_available_models(self) -> List[MlModelSpecification]:
        raise NotImplemented

    def get_model_reader(self, model_spec: MlModelSpecification) -> ModelReader:
        raise NotImplemented
