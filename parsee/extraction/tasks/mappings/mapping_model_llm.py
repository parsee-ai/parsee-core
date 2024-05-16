from typing import *
from decimal import Decimal

from parsee.extraction.tasks.mappings.mapping_model import MappingModel, ElementSchema, MappingSchema, ParseeBucket
from parsee.storage.interfaces import StorageManager
from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel
from parsee.extraction.tasks.mappings.features import LLMMappingFeatureBuilder
from parsee.extraction.extractor_elements import FinalOutputTable
from parsee.utils.constants import ID_NOT_AVAILABLE
from parsee.utils.helper import parse_json_dict, parse_int_simple


class MappingModelLLM(MappingModel):

    def __init__(self, items: List[ElementSchema], storage: StorageManager, llm: LLMBaseModel, **kwargs):
        super().__init__(items)
        self.model_name = llm.spec.model_id
        self.llm = llm
        self.storage = storage
        self.prob = 0.8
        self.feature_builder: LLMMappingFeatureBuilder = LLMMappingFeatureBuilder()

    def parse_answer(self, table: FinalOutputTable, answer: str, schema: MappingSchema, li_identifier: str) -> List[ParseeBucket]:
        answer_dict = parse_json_dict(answer)
        if answer_dict is None or not isinstance(answer_dict, Dict):
            return []
        output = []
        handled = set()
        for bucket in schema.buckets:
            if bucket.id in answer_dict and isinstance(answer_dict[bucket.id], List):
                for answer in answer_dict[bucket.id]:
                    line_item_idx = parse_int_simple(answer)
                    if line_item_idx is not None:
                        item = ParseeBucket(schema.id, bucket.id, li_identifier, self.model_name, self.prob, line_item_idx, 0, Decimal(1))
                        output.append(item)
                        handled.add(line_item_idx)
        # handle items that have not been classified
        for line_item_idx, _ in enumerate(table.line_items):
            if line_item_idx not in handled:
                output.append(ParseeBucket(schema.id, ID_NOT_AVAILABLE, li_identifier, self.model_name, self.prob, line_item_idx, 0, Decimal(1)))
        return output

    def classify_with_schema(self, table: FinalOutputTable, schema: MappingSchema) -> List[ParseeBucket]:

        prompt = self.feature_builder.make_prompt(table, schema)
        answer, amount = self.llm.make_prompt_request(prompt)
        self.storage.log_expense(self.llm.spec.model_id, amount, f"mapping:{table.detected_class}")
        return self.parse_answer(table, answer, schema, table.li_identifier)
