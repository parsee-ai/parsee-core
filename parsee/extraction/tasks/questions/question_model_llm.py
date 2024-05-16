from typing import *

from parsee.extraction.extractor_elements import StandardDocumentFormat
from parsee.extraction.tasks.questions.question_model import QuestionModel
from parsee.extraction.tasks.questions.features import GeneralQueriesPromptBuilder, MAIN_QUESTION_STR
from parsee.extraction.extractor_dataclasses import ParseeAnswer, ParseeMeta
from parsee.storage.interfaces import StorageManager
from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel
from parsee.templates.general_structuring_schema import StructuringItemSchema, GeneralQueryItemSchema
from parsee.extraction.models.llm_models.structuring_schema import get_prompt_schema_item
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.utils.helper import parse_json_dict


class LLMQuestionModel(QuestionModel):

    def __init__(self, items: List[GeneralQueryItemSchema], meta_items: List[StructuringItemSchema], storage: StorageManager, llm: LLMBaseModel, **kwargs):
        super().__init__(items, meta_items)
        self.storage = storage
        self.llm = llm
        self.model_name = llm.spec.model_id
        self.prompt_builder = GeneralQueriesPromptBuilder(storage)

    def parse_prompt_answer(self, item: GeneralQueryItemSchema, prompt_answer: str, total_elements: Optional[int], document: Optional[StandardDocumentFormat]) -> List[ParseeAnswer]:

        json_data = parse_json_dict(prompt_answer)

        if json_data is None:
            return []

        answer_blocks = []
        if "answers" in json_data and isinstance(json_data["answers"], list):
            answer_blocks = json_data["answers"]
        elif MAIN_QUESTION_STR in json_data:
            answer_blocks = [json_data]

        sources = []
        if "sources" in json_data and isinstance(json_data["sources"], list):
            sources = [int(x) for x in json_data["sources"] if str(x).isdigit() and (total_elements is None or 0 <= int(x) <= total_elements)]

        sources_full = [document.elements[el_idx].source for el_idx in sources] if document is not None else []

        output = []
        for main_and_meta in answer_blocks:

            if not isinstance(main_and_meta, dict):
                continue

            main_answer = None
            parse_successful = False
            detected_meta = []

            for key, val in main_and_meta.items():

                if key is None:
                    continue

                val = str(val)

                if MAIN_QUESTION_STR in key:
                    prompt_item = get_prompt_schema_item(item)
                    main_answer, parse_successful = prompt_item.get_value(val)
                else:
                    meta_items_filtered = [x for x in self.meta if x.id == key]
                    if len(meta_items_filtered) > 0:
                        meta_item = meta_items_filtered[0]
                        prompt_item = get_prompt_schema_item(meta_item)
                        parsed, parse_successful = prompt_item.get_value(val)

                        detected_meta.append(ParseeMeta(self.model_name, 0, sources_full, meta_item.id, parsed, 0.8 if parse_successful else 0))

            if main_answer is not None:

                output.append(ParseeAnswer(self.model_name, sources_full, item.id, main_answer, prompt_answer, parse_successful, detected_meta))

        return output

    def predict_for_prompt(self, prompt: Prompt, schema_item: GeneralQueryItemSchema, max_element_index: Optional[int], document: Optional[StandardDocumentFormat]) -> List[ParseeAnswer]:
        prompt_answer, amount = self.llm.make_prompt_request(prompt)
        self.storage.log_expense(self.llm.spec.model_id, amount, schema_item.id)
        return self.parse_prompt_answer(schema_item, prompt_answer, max_element_index, document)

    def predict_answers(self, document: StandardDocumentFormat) -> List[ParseeAnswer]:

        answers: List[ParseeAnswer] = []
        for schema_item in self.items:
            relevant_elements = self.prompt_builder.get_relevant_elements(schema_item, document)
            prompt = self.prompt_builder.build_prompt(schema_item, self.meta, document, relevant_elements, self.llm.spec.multimodal, self.llm.spec.max_images, self.llm.spec.max_image_pixels)
            answers += self.predict_for_prompt(prompt, schema_item, len(document.elements), document)

        return answers
