from typing import *
from functools import reduce
import json

from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.models.llm_models.structuring_schema import get_prompt_schema_item
from parsee.templates.general_structuring_schema import GeneralQueryItemSchema, StructuringItemSchema
from parsee.extraction.extractor_dataclasses import ParseeMeta, ExtractedSource, ParseeAnswer
from parsee.utils.enums import DocumentType, SearchStrategy
from parsee.storage.interfaces import StorageManager
from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl, StructuredTable


MAIN_QUESTION_STR = "main_question"


class GeneralQueriesPromptBuilder:

    storage: StorageManager

    def __init__(self, storage: StorageManager):
        self.storage = storage

    def format_single_item(self, answer: ParseeAnswer, item: GeneralQueryItemSchema, meta_items: List[StructuringItemSchema]) -> Dict:
        schema_item = get_prompt_schema_item(item)
        output = {MAIN_QUESTION_STR: schema_item.parsed_to_raw(answer.class_value)}
        for meta_item in meta_items:
            meta_schema_item = get_prompt_schema_item(meta_item)
            meta_answers = [x for x in answer.meta if x.class_id == meta_item.id]
            answer_chosen = meta_answers[0].class_value if len(meta_answers) > 0 else "n/a"
            output[meta_item.id] = meta_schema_item.parsed_to_raw(answer_chosen)
        return output

    def build_raw_value(self, answers: List[ParseeAnswer], item: GeneralQueryItemSchema, meta_items: List[StructuringItemSchema], add_sources: bool = True) -> str:
        if len(meta_items) > 0:
            output = {"answers": [self.format_single_item(x, item, meta_items) for x in answers]}
        else:
            # if no meta items, there can be only one answer
            output = self.format_single_item(answers[0], item, meta_items)
        if add_sources:
            output["sources"] = [x.element_index for x in answers[0].sources]
        return json.dumps(output)

    def get_relevant_elements(self, schema_item: GeneralQueryItemSchema, document: StandardDocumentFormat) -> List[ExtractedEl]:

        if schema_item.searchStrategy == SearchStrategy.VECTOR:
            return self.storage.vector_store.find_closest_elements(document, schema_item.title, schema_item.keywords, False)
        elif schema_item.searchStrategy == SearchStrategy.START:
            return document.elements
        else:
            raise NotImplementedError

    def get_elements_text(self, elements: List[ExtractedEl], document: StandardDocumentFormat):
        llm_text = "This is the available data to answer the question (in the following, if tables have empty cells, they are omitted, if a cell spans several columns, values might be repeated for each cell):\n"
        for el in elements:
            llm_text += f"[{el.source.element_index}]: {el.get_text_and_surrounding_elements_text(document.elements) if isinstance(el, StructuredTable) else el.get_text_llm(True)} [/{el.source.element_index}]\n"
        return llm_text

    def build_prompt(self, structuring_item: GeneralQueryItemSchema, meta_items: List[StructuringItemSchema], document: StandardDocumentFormat, relevant_elements: List[ExtractedEl], multimodal: bool = False, max_images: Optional[int] = None, max_image_size: Optional[int] = None) -> Prompt:

        if not multimodal:
            general_info = "You are supposed to answer a question based on text fragments that are provided. " \
                           "The fragments start with a number in square brackets and then the actual text. The end of the fragment is shown by the same number in square brackets, only that the number is preceded by a slash. E.g. [22] Some Text [/22]. The lower the number of the fragment, " \
                           "the earlier the fragment appeared in the document (reading from left to right, top to bottom). "
        else:
            general_info = "You are supposed to answer a question based on one or several images that are provided below."

        prompt_schema_item = get_prompt_schema_item(structuring_item)

        additional_info_str = f" Additional info: {structuring_item.additionalInfo}" if structuring_item.additionalInfo.strip() != "" else ""
        main_question = f'The question is: {structuring_item.title} {additional_info_str} {prompt_schema_item.get_possible_values_str()}'

        relevant_meta_items = [x for x in meta_items if structuring_item.metaInfoIds is not None and x.id in structuring_item.metaInfoIds]

        # build full example
        source_examples = [ExtractedSource(DocumentType.PDF, None, None, 241, None), ExtractedSource(DocumentType.PDF, None, None, 423, None)]
        meta_examples = [ParseeMeta("test", 0, source_examples, x.id, get_prompt_schema_item(x).get_example(True), 0.8) for x in relevant_meta_items]
        sample_answer = ParseeAnswer("sample", source_examples, structuring_item.id, prompt_schema_item.get_example(), "", True, meta_examples)
        example_output = self.build_raw_value([sample_answer], structuring_item, relevant_meta_items, not multimodal)
        sources_format = "Under the key 'sources', you should also provide the numbers of the text fragments you used to answer the question." if not multimodal else ""

        full_example = f"Your answer should be formatted as a JSON object, under the key '{MAIN_QUESTION_STR}' you can put as value the answer to the question in the designated format. {sources_format} For example your answer could look like this: {example_output}" \
            if len(relevant_meta_items) == 0 else f"Your answer should be formatted as a JSON object. Under the key 'answers' can be one or more JSON object(s), with key value pairs for the main question and the meta items. For the main question use the key '{MAIN_QUESTION_STR}', for the meta items use their ID as key. {sources_format} For example your answer could look like this: {example_output}"

        # add prompting for meta
        if len(relevant_meta_items) > 0:
            main_question += "\n We also want to retrieve some meta information. In the following we will present the meta item ID and then the additional question to be answered, format: (META_ID): QUESTION."
            for meta_item in relevant_meta_items:
                meta_prompt_item = get_prompt_schema_item(meta_item)
                main_question += f"\n({meta_item.id}): {meta_item.title} {meta_item.additionalInfo} {meta_prompt_item.get_possible_values_str()}"

        prompt = Prompt(general_info, main_question, None,
                        full_example,
                        self.get_elements_text(relevant_elements, document) if not multimodal else self.storage.image_creator.get_images(document, relevant_elements, max_images, max_image_size))
        return prompt
