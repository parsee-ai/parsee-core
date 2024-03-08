from typing import *

from parsee.extraction.extractor_elements import ExtractedEl, get_text_distance, StandardDocumentFormat
from parsee.extraction.extractor_dataclasses import ParseeLocation
from parsee.templates.element_schema import ElementSchema
from parsee.utils.helper import words_contained, clean_text_for_word_vectors2, composition_percentages
from parsee.utils.enums import ElementType, SearchStrategy
from parsee.datasets.dataset_dataclasses import DatasetRow
from parsee.utils.constants import ELEMENTS_WORDS_TO_INCLUDE, ELEMENTS_TABLES_TO_INCLUDE
from parsee.storage.interfaces import StorageManager
from parsee.extraction.models.llm_models.prompts import Prompt


class LocationFeatureBuilder:

    memory: Dict[int, Dict[str, any]]

    def __init__(self):
        self.memory = {}
        self.number_replacement = "xnumberx"

    def _get_base_features(self, element_index: int, element: ExtractedEl) -> Dict[str, any]:

        if element_index in self.memory:
            return self.memory[element_index]

        el_dict = {}
        el_text = element.get_text()
        num_words = len(words_contained(el_text))
        el_dict['type'] = element.el_type.value
        el_dict['text'] = element.get_text_llm(False)
        el_dict['text_clean'] = clean_text_for_word_vectors2(el_text, remove_special_chars=True, remove_all_numbers=True, number_token=self.number_replacement)
        el_dict['num_words'] = num_words
        el_dict["percent_numbers"] = composition_percentages(el_text)['numbers']

        self.memory[element_index] = el_dict
        return el_dict

    def make_features(self, source_identifier: Optional[str], template_id: Optional[str], element_indices: List[int], elements: List[ExtractedEl], include_tables: bool = True) -> List[DatasetRow]:

        output = []

        for el_idx in element_indices:

            base_features = self._get_base_features(el_idx, elements[el_idx])
            text_before = self._get_text_features(el_idx, True, elements, ELEMENTS_WORDS_TO_INCLUDE)
            text_after = self._get_text_features(el_idx, False, elements, ELEMENTS_WORDS_TO_INCLUDE)
            tables_before = self._get_table_features(el_idx, True, elements, ELEMENTS_TABLES_TO_INCLUDE if include_tables else 0)
            tables_after = self._get_table_features(el_idx, False, elements, ELEMENTS_TABLES_TO_INCLUDE if include_tables else 0)

            full_features = {
                **base_features,
                "text_before": text_before,
                "text_after": text_after,
                **tables_before,
                **tables_after
            }

            row = DatasetRow(source_identifier, template_id, el_idx, full_features)

            output.append(row)

        return output

    # returns forward/backward looking text features
    def _get_text_features(self, idx: int, backward: bool, elements: List[ExtractedEl], word_limit: int) -> str:

        text_entries_to_take = []
        word_counter = 0

        range_generator = range(idx + 1, len(elements)) if not backward else range(idx - 1, -1, -1)

        for kk in range_generator:

            element = elements[kk]

            if element.el_type == ElementType.TEXT:

                el_features = self._get_base_features(kk, elements[kk])

                word_counter += el_features['num_words']
                if backward:
                    text_entries_to_take.append(el_features['text_clean'])
                else:
                    text_entries_to_take.insert(0, el_features["text_clean"])

                if word_counter >= word_limit:
                    break

        # take zeros entry
        if len(text_entries_to_take) == 0:
            return ""
        else:
            return (" ".join(text_entries_to_take)).strip()

    # returns features for tables before/after actual table
    def _get_table_features(self, idx: int, backward: bool, elements: List[ExtractedEl], table_limit: int) -> Dict[str, any]:
    
        tables_added = 0
        temp_features = {}
    
        range_generator = range(idx + 1, len(elements)) if not backward else range(idx - 1, -1, -1)
    
        for kk in range_generator:

            element = elements[kk]

            if element.el_type == ElementType.TABLE:

                table_features = self._get_base_features(kk, element)

                tables_added += 1
    
                # determine text distance between tables
                text_distance = get_text_distance(element.source.element_index, elements[idx].source.element_index, elements)
    
                # add all features
                key_append = f"_before_{tables_added}" if backward else f"_after_{tables_added}"
                temp_features["text_distance"+key_append] = text_distance
                temp_features["percent_numbers" + key_append] = table_features["percent_numbers"]
                temp_features["num_words" + key_append] = table_features["num_words"]
                temp_features["text_clean" + key_append] = table_features["text_clean"]
                temp_features["text_simple" + key_append] = table_features["text"]
    
                if tables_added >= table_limit:
                    break
    
        # if not enough tables added, add empty features
        if tables_added < table_limit:
            for a in range(0, table_limit - tables_added):
                idx_offset = abs(a+1+tables_added)
                key_append = f"_before_{idx_offset}" if backward else f"_after_{idx_offset}"
                temp_features["text_distance" + key_append] = 0
                temp_features["percent_numbers" + key_append] = 0
                temp_features["num_words" + key_append] = 0
                temp_features["text_clean" + key_append] = ""
                temp_features["text_simple" + key_append] = ""
    
        return temp_features


class LLMLocationFeatureBuilder(LocationFeatureBuilder):

    def __init__(self):
        super().__init__()
        self.number_replacement = "[number]"

    def build_raw_answer(self, locations: List[ParseeLocation]) -> str:
        sorted_ids = list(sorted([x.source.element_index for x in locations]))
        return f"[{','.join([str(x) for x in sorted_ids])}]"

    def get_elements_text(self, features: List[DatasetRow]):
        llm_text = ""
        for feature_entry in features:
            llm_text += f"[{feature_entry.element_identifier}] text before table: {feature_entry.get_feature('text_before')}; line items in table: {feature_entry.get_feature('text')} [end of item] \n"
        return llm_text

    def make_prompt(self, item: ElementSchema, document: StandardDocumentFormat, storage: StorageManager) -> Prompt:

        if item.searchStrategy == SearchStrategy.VECTOR:
            closest_elements = storage.vector_store.find_closest_elements(document, item.title, item.keywords, True)
        elif item.searchStrategy == SearchStrategy.START:
            closest_elements = [el for el in document.elements if el.el_type == ElementType.TABLE]
        else:
            raise NotImplemented

        additional_info_str = f" Additional info: {item.additionalInfo}" if item.additionalInfo.strip() != "" else ""
        all_element_indices = [x.source.element_index for x in closest_elements]

        features = self.make_features(None, None, all_element_indices, document.elements, False)

        prompt = Prompt("", f'we want to find an item labeled "{item.title}".{additional_info_str}', f"""We are providing data in the following with the element_index and the text, such as [ELEMENT_INDEX] "TEXT" [end of item].
                Using the text, identify "{item.title}" and return the elements which you think are most likely representing "{item.title}" (it can be one item or several together).
                It is very important that you only return the element_index and nothing else.""", "Your response could be for example: [10] or [241],[1204] ", self.get_elements_text(features))

        return prompt
