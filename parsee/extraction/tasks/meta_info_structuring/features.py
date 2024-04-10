from typing import *

from parsee.extraction.extractor_elements import FinalOutputTableColumn, ExtractedEl, StructuredTable
from parsee.templates.general_structuring_schema import StructuringItemSchema
from parsee.extraction.extractor_dataclasses import ParseeMeta
from parsee.utils.helper import is_number_cell, is_year_cell, clean_text_for_word_vectors2
from parsee.datasets.dataset_dataclasses import DatasetRow, MetaUniqueIdentifier
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.models.llm_models.structuring_schema import get_prompt_schema_item


class MetaFeatureBuilder:

    memory: Dict[MetaUniqueIdentifier, DatasetRow]

    def __init__(self):
        self.memory = {}
        self.cell_start_delimiter = "xcellstartx"
        self.row_start_delimiter = "xrowstartx"
        self.cell_self_append = "xselfx"
        self.number_replacement = "xnumberx"

    # returns features for meta info ml of table using word embeddings
    def get_meta_features_table(self, structured_table: StructuredTable, col_index_chosen: int, text_before: int, base_year: Union[int, None]) -> Dict[str, any]:

        def get_words_from_cells(cell_obj_list, repeat=False):
            objects_used = []
            all_fragments = []
            for c_obj in cell_obj_list:
                if type(c_obj) is str:
                    all_fragments.append(c_obj)
                else:
                    if (c_obj not in objects_used or repeat) and c_obj.val is not None:
                        all_fragments.append(c_obj.val)
                        objects_used.append(c_obj)

            all_text = " ".join(all_fragments)
            return all_text

        # collect text from various places
        words_list = {"before": text_before, "li_and_values": "", "top_li": "", "top_data": "", "top_column_all": "", "top_column_pos_0": "", "top_column_pos_1": "", "top_column_pos_2": "",
                      "other_li": "", "other_values": "", "neighbor_col_top_left": "", "neighbor_col_top_right": "", "total_header_values": "", "top_li_reworked": ""}

        # determine words of line items and values
        cell_list = []
        for row_index in structured_table.numeric_rows_indices:
            for cell_obj in structured_table.rows[row_index].final_values:
                # exclude numbers
                if is_number_cell(cell_obj.val) and not is_year_cell(cell_obj.val):
                    continue
                cell_list.append(cell_obj)
        words_list['li_and_values'] = get_words_from_cells(cell_list)

        # determine words on top of line items
        cell_list = []
        for row in structured_table.header_rows:
            for col_index, cell_obj in enumerate(row.final_values):
                if col_index < min(structured_table.numeric_cols_indices):
                    cell_list.append(cell_obj)
        words_list['top_li'] = get_words_from_cells(cell_list)

        # determine words on top of data
        cell_list = []
        for row in structured_table.header_rows:
            for col_index, cell_obj in enumerate(row.final_values):
                if col_index >= min(structured_table.numeric_cols_indices):
                    cell_list.append(cell_obj)
        words_list['top_data'] = get_words_from_cells(cell_list)

        # top line items reworked
        cell_list = []
        for row in structured_table.header_rows:
            if row.has_values(max_col_index=min(structured_table.numeric_cols_indices) - 1):
                for col_index, cell_obj in enumerate(row.final_values):
                    if col_index < min(structured_table.numeric_cols_indices):
                        # check that cell obj is not contained in first value column
                        if cell_obj != row.final_values[structured_table.numeric_cols_indices[0]]:
                            cell_list.append(cell_obj)
        words_list['top_li_reworked'] = get_words_from_cells(cell_list)

        # total header data
        cell_list = []
        for row in structured_table.header_rows:
            if row.has_values():
                cell_list.append(self.row_start_delimiter)
                for col_index, cell_obj in enumerate(row.final_values):
                    if col_index in structured_table.numeric_cols_indices:
                        delim = self.cell_start_delimiter if col_index != col_index_chosen else self.cell_start_delimiter + self.cell_self_append
                        cell_list.append(delim)
                        cell_list.append(cell_obj)
        words_list['total_header_values'] = get_words_from_cells(cell_list, True)

        # determine words on top IN COLUMN
        cell_list = []
        for row in structured_table.header_rows:
            cell_obj = row.final_values[col_index_chosen]
            cell_list.append(cell_obj)
        words_list['top_column_all'] = get_words_from_cells(cell_list)

        # determine words on top IN COLUMN, pos 0 only
        cell_list = []
        start_index = min(structured_table.numeric_rows_indices) - 1 if len(structured_table.numeric_rows_indices) > 0 else 0
        to_skip = 0
        to_skip_counter = -1
        for row_index in range(start_index, -1, -1):
            cell = structured_table.rows[row_index].final_values[col_index_chosen]
            if cell.val is not None and cell.val.strip() != "":
                to_skip_counter += 1
                if to_skip <= to_skip_counter:
                    cell_list.append(cell)
                    break
        words_list['top_column_pos_0'] = get_words_from_cells(cell_list)

        # determine words on top IN COLUMN, pos 1 only
        cell_list = []
        start_index = min(structured_table.numeric_rows_indices) - 1 if len(structured_table.numeric_rows_indices) > 0 else 0
        to_skip = 1
        to_skip_counter = -1
        for row_index in range(start_index, -1, -1):
            cell = structured_table.rows[row_index].final_values[col_index_chosen]
            if cell.val is not None and cell.val.strip() != "":
                to_skip_counter += 1
                if to_skip <= to_skip_counter:
                    cell_list.append(cell)
                    break
        words_list['top_column_pos_1'] = get_words_from_cells(cell_list)

        # determine words on top IN COLUMN, pos 2 only
        cell_list = []
        start_index = min(structured_table.numeric_rows_indices) - 1 if len(structured_table.numeric_rows_indices) > 0 else 0
        to_skip = 2
        to_skip_counter = -1
        for row_index in range(start_index, -1, -1):
            cell = structured_table.rows[row_index].final_values[col_index_chosen]
            if cell.val is not None and cell.val.strip() != "":
                to_skip_counter += 1
                if to_skip <= to_skip_counter:
                    cell_list.append(cell)
                    break
        words_list['top_column_pos_2'] = get_words_from_cells(cell_list)

        # other locations, li
        cell_list = []
        for row in structured_table.other_rows:
            for col_index, cell_obj in enumerate(row.final_values):
                if col_index < min(structured_table.numeric_cols_indices):
                    cell_list.append(cell_obj)
        words_list['other_li'] = get_words_from_cells(cell_list)

        # other locations, values
        cell_list = []
        for row in structured_table.other_rows:
            for col_index, cell_obj in enumerate(row.final_values):
                if col_index >= min(structured_table.numeric_cols_indices):
                    cell_list.append(cell_obj)
        words_list['other_values'] = get_words_from_cells(cell_list)

        # neighbor col top left
        col_indices_available = list(sorted(structured_table.numeric_cols_indices))
        idx_self = col_indices_available.index(col_index_chosen)
        if idx_self > 0:
            cell_list = []
            for row in structured_table.header_rows:
                cell_obj = row.final_values[col_indices_available[idx_self - 1]]
                cell_list.append(cell_obj)
            words_list['neighbor_col_top_left'] = get_words_from_cells(cell_list)

        # neighbor col top right
        col_indices_available = list(sorted(structured_table.numeric_cols_indices))
        idx_self = col_indices_available.index(col_index_chosen)
        if idx_self < len(col_indices_available) - 1:
            cell_list = []
            for row in structured_table.header_rows:
                cell_obj = row.final_values[col_indices_available[idx_self + 1]]
                cell_list.append(cell_obj)
            words_list['neighbor_col_top_right'] = get_words_from_cells(cell_list)

        # make final values
        for key, values in words_list.items():
            words_list[key] = clean_text_for_word_vectors2(values, base_year, number_token=self.number_replacement)

        return words_list
    
    def make_features(self, source_identifier: Optional[str], template_id: Optional[str], column: FinalOutputTableColumn, elements: List[ExtractedEl], base_year: Optional[int], custom_features: Optional[Dict[str, any]]) -> DatasetRow:

        unique_identifier = MetaUniqueIdentifier(column.detected_class, column.col_idx, column.kv_identifier)

        if unique_identifier in self.memory:
            return self.memory[unique_identifier]

        text_words_to_include_before = 200
        main_el: StructuredTable = elements[column.locations[0].source.element_index]

        text_before_words = main_el.get_text_before(elements, text_words_to_include_before)
        org_idx = column.org_col_indices[0]
        features = self.get_meta_features_table(main_el, org_idx, text_before_words, base_year)

        features = {"col_idx_values": column.col_idx, "col_idx_all": org_idx, **features}

        if custom_features is not None:
            features = {**features, **custom_features}

        self.memory[unique_identifier] = DatasetRow(source_identifier, template_id, unique_identifier, features)

        return self.memory[unique_identifier]


class LLMMetaFeatureBuilder(MetaFeatureBuilder):

    def __init__(self):
        super().__init__()
        self.cell_start_delimiter = "(cell start)"
        self.row_start_delimiter = "(row start)"
        self.cell_self_append = "(cell main)"
        self.number_replacement = "[number]"

    def build_raw_answer(self, items: List[StructuringItemSchema], results: List[ParseeMeta]) -> str:
        output = ""
        for k, item in enumerate(items):
            if k > 0:
                output += "\n"
            schema_item = get_prompt_schema_item(item)
            meta_items_filtered = [x for x in results if x.class_id == item.id]
            val = "n/a" if len(meta_items_filtered) == 0 else schema_item.parsed_to_raw(meta_items_filtered[0].class_value)
            output += f"{k+1}) {val}"
        return output

    def text_main(self, items: List[StructuringItemSchema]) -> str:
        output = ""

        for k, item in enumerate(items):
            schema_item = get_prompt_schema_item(item)
            additional_info_str = f" Additional info: {item.additionalInfo}." if item.additionalInfo.strip() != "" else ""
            output += f"{k + 1}) {item.title}.{additional_info_str} {schema_item.get_possible_values_str()}\n"

        return output

    def example(self, items: List[StructuringItemSchema]) -> str:
        output = "Your answer could be for example: \n"
        for k, item in enumerate(items):
            output += f"{k + 1}) {get_prompt_schema_item(item).get_example()}\n"

        return output

    def build_prompt(self, features: DatasetRow, col_idx: int, items: List[StructuringItemSchema]) -> Prompt:

        available_data = f'''
                This is the information that is available:\n
                The current column index is: {col_idx}. \n
                On top of the column (in the header part), there is this text: "{features.get_feature('top_column_all')}"\n
                The content of the table is the following (excluding any numbers): {features.get_feature('li_and_values')}\n
                On top of the line items, there is this text: "{features.get_feature('top_li_reworked')}\n"

                Before the table starts, there is this text: "{features.get_feature('before')}"\n
                '''

        prompt = Prompt("We want to recognize certain information in a table, more specifically in a single column.", f'Please identify the following information based on the provided text: {self.text_main(items)}',
                        'It is very important that you only answer with the number of the question and then one of the valid values for that question afterwards.', self.example(items), available_data)

        return prompt

    def make_prompt(self, column: FinalOutputTableColumn, elements: List[ExtractedEl], items: List[StructuringItemSchema]) -> Prompt:

        features = self.make_features(None, None, column, elements, None, None)

        return self.build_prompt(features, column.col_idx, items)