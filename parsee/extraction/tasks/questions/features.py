from typing import *
from functools import reduce

from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.models.llm_models.structuring_schema import get_prompt_schema_item
from parsee.templates.general_structuring_schema import GeneralQueryItemSchema, StructuringItemSchema
from parsee.extraction.tasks.questions.utils import MAIN_QUESTION_STR
from parsee.extraction.extractor_dataclasses import ParseeMeta, ExtractedSource
from parsee.utils.enums import DocumentType, SearchStrategy
from parsee.storage.interfaces import StorageManager
from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl


class GeneralQueriesPromptBuilder:

    storage: StorageManager

    def __init__(self, storage: StorageManager):
        self.storage = storage

    def build_raw_value(self, main_value: any, meta: List[ParseeMeta], sources: List[ExtractedSource], item: GeneralQueryItemSchema, meta_items: List[StructuringItemSchema]):
        schema_item = get_prompt_schema_item(item)
        if len(meta_items) > 0:
            output = f"{MAIN_QUESTION_STR}: {main_value}"
            for meta_item in meta_items:
                meta_schema_item = get_prompt_schema_item(meta_item)
                meta_answers = [x for x in meta if x.class_id == meta_item.id]
                answer_chosen = meta_answers[0].class_value if len(meta_answers) > 0 else "n/a"
                output += f"\n({meta_item.id}): {meta_schema_item.parsed_to_raw(answer_chosen)}"
        else:
            output = f"{schema_item.parsed_to_raw(main_value)}"
        output += "\nSources: " + ",".join(f"[{x.element_index}]" for x in sources)
        return output

    def get_relevant_elements(self, schema_item: GeneralQueryItemSchema, document: StandardDocumentFormat) -> List[ExtractedEl]:

        if schema_item.searchStrategy == SearchStrategy.VECTOR:
            return self.storage.vector_store.find_closest_elements(document, schema_item.title, schema_item.keywords, False)
        elif schema_item.searchStrategy == SearchStrategy.START:
            return document.elements
        else:
            raise NotImplemented

    def get_elements_text(self, elements: List[ExtractedEl]):
        llm_text = " (For the following data, if tables have empty cells, they are omitted)\n"
        for el in elements:
            llm_text += f"[{el.source.element_index}]: {el.get_text_llm(True)}\n"
        return llm_text

    def build_prompt(self, structuring_item: GeneralQueryItemSchema, meta_items: List[StructuringItemSchema], document: StandardDocumentFormat, relevant_elements_custom: Optional[List[ExtractedEl]] = None) -> Prompt:

        elements = self.get_relevant_elements(structuring_item, document) if relevant_elements_custom is None else relevant_elements_custom

        general_info = "You are supposed to answer a question based on text fragments that are provided. " \
                       "The fragments start with a number and then the actual text. The lower the number of the fragment, " \
                       "the earlier the fragment appeared in the document (reading from left to right, top to bottom). " \
                       "Please respond as concise as possible."

        prompt_schema_item = get_prompt_schema_item(structuring_item)

        additional_info_str = f" Additional info: {structuring_item.additionalInfo}" if structuring_item.additionalInfo.strip() != "" else ""
        main_question = f'The question is: {structuring_item.title} {additional_info_str} {prompt_schema_item.get_possible_values_str()}'

        relevant_meta_items = [x for x in meta_items if structuring_item.metaInfoIds is not None and x.id in structuring_item.metaInfoIds]

        # build full example
        source_examples = [ExtractedSource(DocumentType.PDF, None, None, 241, None), ExtractedSource(DocumentType.PDF, None, None, 423, None)]
        meta_examples = [ParseeMeta("test", 0, source_examples, x.id, get_prompt_schema_item(x).get_example(True), 0.8) for x in relevant_meta_items]
        example_output = self.build_raw_value(prompt_schema_item.get_example(), meta_examples, source_examples, structuring_item, meta_items)

        full_example = f"Your answer could look like this: {example_output}" if len(relevant_meta_items) == 0 else f"One possible answer block could be: {example_output}"

        # add prompting for meta
        if len(relevant_meta_items) > 0:
            main_question += "\n We also want to retrieve some meta information. In the following we will present the meta item ID and then the additional question to be answered, format: (META_ID): QUESTION. In the answer please first provide the meta ID (for the main question use 'main question' instead) in brackets and then the answer. If there are several correct answers to the question that are differentiated by their meta values, structure your answer into multiple answer blocks of the same format, one after the other separated by new lines."
            for meta_item in relevant_meta_items:
                meta_prompt_item = get_prompt_schema_item(meta_item)
                main_question += f"\n({meta_item.id}): {meta_item.title} {meta_item.additionalInfo} {meta_prompt_item.get_possible_values_str()}"

        prompt = Prompt(general_info, main_question,
                        'Please at the end of each answer also provide the numbers of the text fragments you used to answer in square brackets.',
                        full_example,
                        self.get_elements_text(elements))

        return prompt