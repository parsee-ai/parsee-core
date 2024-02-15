from typing import *
from decimal import Decimal

from parsee.extraction.tasks.mappings.mapping_classifier import MappingClassifier, ElementSchema, MappingSchema, ParseeBucket
from parsee.storage.interfaces import StorageManager
from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel
from parsee.extraction.tasks.mappings.features import LLMMappingFeatureBuilder
from parsee.extraction.extractor_elements import FinalOutputTable
from parsee.extraction.extractor_dataclasses import ID_NOT_AVAILABLE


class MappingClassifierLLM(MappingClassifier):

    def __init__(self, items: List[ElementSchema], storage: StorageManager, llm: LLMBaseModel, **kwargs):
        super().__init__(items)
        self.classifier_name = llm.classifier_name
        self.llm = llm
        self.storage = storage
        self.prob = 0.8
        self.feature_builder: LLMMappingFeatureBuilder = LLMMappingFeatureBuilder()

    def parse_answer(self, answer: str, schema: MappingSchema, kv_idx: int, li_identifier: str) -> Union[None, ParseeBucket]:
        for bucket in schema.buckets:
            if self.feature_builder.item_id_string(bucket).lower() in answer.lower() or bucket.id.lower() in answer.lower():
                return ParseeBucket(schema.id, bucket.id, li_identifier, self.classifier_name, self.prob, kv_idx, 0, Decimal(1))
        return ParseeBucket(schema.id, ID_NOT_AVAILABLE, li_identifier, self.classifier_name, self.prob, kv_idx, 0, Decimal(1))

    def classify_with_schema(self, table: FinalOutputTable, schema: MappingSchema) -> List[ParseeBucket]:

        output = []
        for idx, _ in enumerate(table.line_items):
            prompt = self.feature_builder.make_prompt(table, schema, idx)
            answer, amount = self.llm.make_prompt_request(str(prompt))
            self.storage.log_expense(self.llm.classifier_name, amount, f"mapping:{table.detected_class}")
            output.append(self.parse_answer(answer, schema, idx, table.li_identifier))
        return output
