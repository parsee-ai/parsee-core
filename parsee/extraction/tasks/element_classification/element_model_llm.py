from typing import *
import re

from parsee.extraction.tasks.element_classification.element_model import ElementModel, ElementSchema, StandardDocumentFormat, ParseeLocation
from parsee.storage.interfaces import StorageManager
from parsee.utils.helper import is_number_cell, clean_numeric_value
from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel
from parsee.extraction.tasks.element_classification.features import LLMLocationFeatureBuilder


class ElementModelLLM(ElementModel):

    def __init__(self, items: List[ElementSchema], storage: StorageManager, llm: LLMBaseModel, **kwargs):
        super().__init__(items)
        self.model_name = llm.model_name
        self.llm = llm
        self.max_search_items = 10
        self.storage = storage
        self.prob = 0.8
        self.feature_builder: LLMLocationFeatureBuilder = LLMLocationFeatureBuilder()

    def parse_prompt_answer(self, prompt_answer: str) -> Union[int, None]:
        answers = prompt_answer.splitlines()
        for k, answer in enumerate(answers):
            result = re.search(r'(\[|)(\d+)(\]|)', answer)
            if result is not None and len(result.groups()) > 2:
                value_predicted = result.group(2)
                clean_value = int(clean_numeric_value(value_predicted))
                if is_number_cell(value_predicted) and clean_value >= 0:
                    return clean_value
        return None

    def classify_elements(self, document: StandardDocumentFormat) -> List[ParseeLocation]:

        output: List[ParseeLocation] = []
        PARTIAL_PROB = 0

        for item in self.items:

            prompt = self.feature_builder.make_prompt(item, document, self.storage)

            answer, amount = self.llm.make_prompt_request(str(prompt))
            self.storage.log_expense(self.llm.model_name, amount, item.id)

            best_idx = self.parse_prompt_answer(answer)

            if best_idx is not None and 0 <= best_idx < len(document.elements):
                el = document.elements[best_idx]
                location = ParseeLocation(self.model_name, PARTIAL_PROB, item.id, self.prob, el.source, [])
                output.append(location)
        return output
