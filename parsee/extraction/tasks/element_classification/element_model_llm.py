from typing import *
import re

from parsee.extraction.tasks.element_classification.element_model import ElementModel, ElementSchema, StandardDocumentFormat, ParseeLocation
from parsee.storage.interfaces import StorageManager
from parsee.utils.helper import is_number_cell, clean_numeric_value, parse_json_array
from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel
from parsee.extraction.tasks.element_classification.features import LLMLocationFeatureBuilder


class ElementModelLLM(ElementModel):

    def __init__(self, items: List[ElementSchema], storage: StorageManager, llm: LLMBaseModel, **kwargs):
        super().__init__(items)
        self.model_name = llm.spec.model_id
        self.llm = llm
        self.max_search_items = 10
        self.storage = storage
        self.prob = 0.8
        self.feature_builder: LLMLocationFeatureBuilder = LLMLocationFeatureBuilder()

    def parse_prompt_answer(self, prompt_answer: str) -> List[int]:
        answers = prompt_answer.splitlines()
        output = []
        for k, answer in enumerate(answers):
            values_predicted = parse_json_array(answer)
            if values_predicted is not None:
                for val in values_predicted:
                    if isinstance(val, int):
                        output.append(val)
        return output

    def classify_elements(self, document: StandardDocumentFormat) -> List[ParseeLocation]:

        output: List[ParseeLocation] = []

        for item in self.items:

            prompt = self.feature_builder.make_prompt(item, document, self.storage)

            answer, amount = self.llm.make_prompt_request(prompt)
            self.storage.log_expense(self.llm.spec.model_id, amount, item.id)

            best_indexes = self.parse_prompt_answer(answer)

            partial_prob = 0.0 if len(best_indexes) <= 1 else 0.6

            for best_idx in best_indexes:
                if 0 <= best_idx < len(document.elements):
                    el = document.elements[best_idx]
                    location = ParseeLocation(self.model_name, partial_prob, item.id, self.prob, el.source, [])
                    output.append(location)
        return output
