from typing import *
from decimal import Decimal
from copy import deepcopy

from parsee.storage.interfaces import StorageManager
from parsee.storage.vector_stores.simple_faiss import SimpleFaissStore
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.templates.job_template import JobTemplate
from parsee.extraction.extractor_dataclasses import AssignedMeta, AssignedLocation, AssignedAnswer, AssignedBucket


class InMemoryStorageManager(StorageManager):
    
    truth_questions: List[AssignedAnswer]
    truth_locations: List[AssignedLocation]
    truth_meta: List[AssignedMeta]
    truth_mappings: List[AssignedBucket]

    def __init__(self, available_models: Optional[List[MlModelSpecification]]):
        super().__init__(SimpleFaissStore())
        self.models = available_models if available_models is not None else []
        self.truth_questions = []
        self.truth_locations = []

    def get_available_models(self) -> List[MlModelSpecification]:
        return self.models if self.models is not None else []

    def log_expense(self, service: str, amount: Decimal, class_id: str):
        print("expense", service, amount, class_id)
        
    def assign_truth_values(self, general_questions: Optional[List[AssignedAnswer]], locations: Optional[List[AssignedLocation]]):
        if general_questions is not None:
            self.truth_questions = general_questions
        if locations is not None:
            self.truth_locations = locations
        
    def db_values_template(self, job_template: JobTemplate, rerun_if_no_input: bool) -> JobTemplate:

        # adjust template
        template_new = deepcopy(job_template)

        # add assigned values
        template_new.detection.settings = {**template_new.detection.settings, "assigned": self.truth_locations}
        template_new.questions.settings = {**template_new.questions.settings, "assigned": self.truth_questions}

        # adjust models
        for item in template_new.detection.items:
            item.model = "assigned"
            item.mappingModel = "assigned"
        for item in template_new.meta:
            item.model = "assigned"
        for item in template_new.questions.items:
            item.model = "assigned"

        return template_new
