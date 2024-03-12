from typing import *

from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl
from parsee.extraction.tasks.questions.question_model import QuestionModel
from parsee.extraction.tasks.questions.features import GeneralQueriesPromptBuilder
from parsee.extraction.extractor_dataclasses import ParseeAnswer, ParseeMeta
from parsee.storage.interfaces import StorageManager
from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel
from parsee.extraction.tasks.questions.utils import parse_sources, parse_answer_blocks, parse_main_and_meta, MAIN_QUESTION_STR
from parsee.templates.general_structuring_schema import StructuringItemSchema, GeneralQueryItemSchema
from parsee.extraction.models.llm_models.structuring_schema import get_prompt_schema_item
from parsee.datasets.dataset_dataclasses import DatasetRow


class LLMQuestionModel(QuestionModel):

    def __init__(self, items: List[GeneralQueryItemSchema], meta_items: List[StructuringItemSchema], storage: StorageManager, llm: LLMBaseModel, **kwargs):
        super().__init__(items, meta_items)
        self.storage = storage
        self.llm = llm
        self.model_name = llm.model_name
        self.prompt_builder = GeneralQueriesPromptBuilder(storage)

    def parse_prompt_answer(self, item: GeneralQueryItemSchema, prompt_answer: str, total_elements: Optional[int], document: Optional[StandardDocumentFormat]) -> List[ParseeAnswer]:

        multi_block = item.metaInfoIds is not None and len(item.metaInfoIds) > 0

        answer_blocks = parse_answer_blocks(prompt_answer) if multi_block else [prompt_answer]

        output = []
        for answer_block in answer_blocks:

            final_answer, sources = parse_sources(answer_block, total_elements)

            main_and_meta = parse_main_and_meta(final_answer) if multi_block else {MAIN_QUESTION_STR: final_answer}

            main_answer = None
            parse_successful = False
            detected_meta = []
            sources_full = [document.elements[el_idx].source for el_idx in sources] if document is not None else []
            for key, val in main_and_meta.items():

                if key is None:
                    continue

                if MAIN_QUESTION_STR in key:
                    prompt_item = get_prompt_schema_item(item)
                    main_answer, parse_successful = prompt_item.get_value(val)
                else:
                    meta_items_filtered = [x for x in self.meta if f"({x.id})" in key]
                    if len(meta_items_filtered) > 0:
                        meta_item = meta_items_filtered[0]
                        prompt_item = get_prompt_schema_item(meta_item)
                        parsed, parse_successful = prompt_item.get_value(val)

                        detected_meta.append(ParseeMeta(self.model_name, 0, sources_full, meta_item.id, parsed, 0.8 if parse_successful else 0))

            if main_answer is not None:

                output.append(ParseeAnswer(self.model_name, sources_full, item.id, main_answer, answer_block, parse_successful, detected_meta))

        return output

    def predict_for_prompt(self, prompt: str, schema_item: GeneralQueryItemSchema, max_element_index: Optional[int], document: Optional[StandardDocumentFormat]) -> List[ParseeAnswer]:
        prompt_answer, amount = self.llm.make_prompt_request(str(prompt))
        self.storage.log_expense(self.llm.model_name, amount, schema_item.id)
        return self.parse_prompt_answer(schema_item, prompt_answer, max_element_index, document)

    def predict_answers(self, document: StandardDocumentFormat) -> List[ParseeAnswer]:

        answers: List[ParseeAnswer] = []
        for schema_item in self.items:
            prompt = self.prompt_builder.build_prompt(schema_item, self.meta, document)
            answers += self.predict_for_prompt(str(prompt), schema_item, len(document.elements), document)

        return answers
