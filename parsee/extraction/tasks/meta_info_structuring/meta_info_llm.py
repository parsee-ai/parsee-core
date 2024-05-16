from typing import *
import re

from parsee.extraction.extractor_elements import ExtractedEl, FinalOutputTableColumn
from parsee.extraction.tasks.meta_info_structuring.meta_info import MetaInfoModel
from parsee.extraction.models.llm_models.structuring_schema import get_prompt_schema_item
from parsee.templates.general_structuring_schema import StructuringItemSchema
from parsee.extraction.extractor_dataclasses import ParseeMeta
from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel
from parsee.storage.interfaces import StorageManager
from parsee.extraction.tasks.meta_info_structuring.features import LLMMetaFeatureBuilder


class MetaLLMModel(MetaInfoModel):

    def __init__(self, items: List[StructuringItemSchema], llm: LLMBaseModel, storage: StorageManager, **kwargs):
        super().__init__(items)
        self.storage = storage
        self.model_name = llm.spec.model_id if llm is not None else "llm"
        self.default_prob_answer = 0.8
        self.elements = []
        self.llm = llm
        self.feature_builder: LLMMetaFeatureBuilder = LLMMetaFeatureBuilder()

    def parse_prompt_answer(self, prompt_answer: str) -> Dict[str, Tuple[str, bool]]:
        answers = prompt_answer.splitlines()
        output = {}
        for k, answer in enumerate(answers):
            result = re.search(r'((\d+)\) *)(.+)', answer)
            if result is not None and len(result.groups()) > 2:
                number = result.group(2)
                value_predicted = result.group(3)
                if number.isdigit() and 0 <= int(number)-1 < len(self.items):
                    item_idx = int(number)-1
                    output[self.items[item_idx].id] = get_prompt_schema_item(self.items[item_idx]).get_value(value_predicted)
        return output

    def predict_meta(self, columns: List[FinalOutputTableColumn], elements: List[ExtractedEl]) -> List[List[ParseeMeta]]:

        all_output = []
        for column in columns:
            prompt = self.feature_builder.make_prompt(column, elements, self.items)

            prompt_answer, amount = self.llm.make_prompt_request(prompt)

            self.storage.log_expense(self.llm.spec.model_id, amount, "meta LLM")

            prediction_dict = self.parse_prompt_answer(prompt_answer)

            output: List[ParseeMeta] = []
            for key, values in prediction_dict.items():
                value, parse_success = values
                output.append(ParseeMeta(self.model_name, column.col_idx, column.sources, key, value, self.default_prob_answer if parse_success else 0))
            all_output.append(output)

        return all_output
