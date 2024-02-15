from typing import *

from parsee.extraction.extractor_elements import StandardDocumentFormat
from parsee.templates.general_structuring_schema import StructuringItemSchema, GeneralQueryItemSchema
from parsee.extraction.extractor_dataclasses import ParseeAnswer, AssignedAnswer, ParseeMeta
from parsee.extraction.tasks.questions.utils import build_raw_value
from parsee.datasets.dataset_dataclasses import DatasetRow


class QuestionModel:

    classifier_name = ""

    def __init__(self, items: List[GeneralQueryItemSchema], meta_items: List[StructuringItemSchema]):
        self.items = items
        self.meta = meta_items

    def predict_answers(self, document: StandardDocumentFormat) -> List[ParseeAnswer]:
        raise NotImplemented


class AssignedQuestionModel(QuestionModel):

    classifier_name = "manual"

    def __init__(self, items: List[GeneralQueryItemSchema], meta_items: List[StructuringItemSchema], truth_questions: List[AssignedAnswer], **kwargs):
        super().__init__(items, meta_items)
        self.answers = truth_questions

    def predict_answers(self, document: StandardDocumentFormat) -> List[ParseeAnswer]:
        output: List[ParseeAnswer] = []

        for answer in self.answers:
            meta_values = [ParseeMeta(self.classifier_name, meta.column_index if meta.column_index is not None else 0, answer.sources, meta.class_id, meta.class_value, 1) for meta in answer.meta]
            output.append(ParseeAnswer(self.classifier_name, answer.sources, answer.class_id, answer.class_value, build_raw_value(answer.class_value, meta_values, answer.sources), True, meta_values))

        return output