from typing import *
import re

from src.extraction.ml.tasks.element_classification.element_classifier import ElementClassifier, ElementSchema, StandardDocumentFormat, ParseeLocation
from src.storage.interfaces import StorageManager
from src.utils.helper import is_number_cell, clean_numeric_value
from src.extraction.ml.models.llm_models.llm_base_model import LLMBaseModel
from src.extraction.ml.tasks.element_classification.features import LLMLocationFeatureBuilder


class ElementClassifierLLM(ElementClassifier):

    def __init__(self, items: List[ElementSchema], storage: StorageManager, llm: LLMBaseModel, **kwargs):
        super().__init__(items)
        self.classifier_name = llm.classifier_name
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

            answer, amount = self.llm.make_prompt_request(prompt)
            self.storage.log_expense(self.llm.classifier_name, amount, item.id)

            best_idx = self.parse_prompt_answer(answer)

            if best_idx is not None and 0 <= best_idx < len(document.elements):
                el = document.elements[best_idx]
                location = ParseeLocation(self.classifier_name, PARTIAL_PROB, item.id, self.prob, el.source, [])
                output.append(location)
        return output
