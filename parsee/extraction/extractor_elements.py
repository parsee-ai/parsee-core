from __future__ import annotations

import numpy as np
import re
from typing import List, Union, Tuple, Dict, Optional, Set
from decimal import Decimal
from dataclasses import dataclass
from hashlib import sha256

from pandas import DataFrame

from parsee.extraction.extractor_dataclasses import ExtractedSource, ParseeLocation, ParseeMeta, ParseeAnswer, ParseeBucket
from parsee.utils.enums import ElementType, DocumentType
from parsee.extraction.tasks.mappings.utils import calc_buckets
from parsee.templates.mappings import MappingSchema
from parsee.extraction.tasks.mappings.utils import get_table_signature
from parsee.utils.helper import get_mean_for_column, clean_spaces, is_number_cell, clean_numeric_value, composition_percentages, delete_trailing_zeros, is_year_cell, \
    words_contained, clean_number_for_matching


# returns the number of text chars between 2 elements
def get_text_distance(el_idx1: int, el_idx2: int, elements: List[ExtractedEl], exclude_indices: Optional[List[int]] = None, include_tables: bool = False):
    max_index = max(el_idx1, el_idx2)
    min_index = min(el_idx1, el_idx2)
    total_chars = 0
    for k in range(min_index + 1, max_index):
        if (elements[k].el_type == ElementType.TEXT or include_tables) and (exclude_indices is None or k not in exclude_indices):
            total_chars += len(elements[k].get_text())
    return total_chars


class FinalOutputTableColumn:
    detected_class: str
    col_idx: int
    li_identifier: Union[str, None]
    kv_identifier: str
    key_value_pairs: List[Tuple[str, any]]
    sources: List[ExtractedSource]
    meta: List[ParseeMeta]
    model: str
    locations: List[ParseeLocation]

    def __init__(self, location: ParseeLocation, element: StructuredTable, col_idx: int, col_idx_local: int, col_idx_org: int):
        self.col_idx = col_idx
        self.detected_class = location.detected_class
        self.key_value_pairs = []
        self.sources = []
        self.meta = []
        self.model = location.model
        self.li_identifier = None
        self.set_identifiers()
        self.locations = [location]
        self._elements = [element]
        self._local_col_indices = [col_idx_local]
        self.org_col_indices = [col_idx_org]
        self.build()

    def __str__(self):
        return str(self.dict_json())

    def __repr__(self):
        return str(self)

    def key_value_pairs_json(self):
        return [{str(key): (str(value) if value is not None else None)} for key, value in self.key_value_pairs]

    def set_identifiers(self):
        self.kv_identifier = sha256(str(self.key_value_pairs_json()).encode('utf-8')).hexdigest()
        self.li_identifier = sha256(str(get_table_signature(self.key_value_pairs)).encode('utf-8')).hexdigest()

    def dict_json(self):
        return {"class_id": self.detected_class, "li_identifier": self.li_identifier, "col_idx": self.col_idx, "meta": [x.to_json_dict() for x in self.meta], "values": self.key_value_pairs_json(), "sources": [x.to_json_dict() for x in self.sources]}

    def build(self):
        # build values
        kv_pairs = []
        sources = []
        for k, loc in enumerate(self.locations):
            el = self._elements[k]
            kv_pairs += [(x.clean_caption(), x.value_elements[self._local_col_indices[k]].numeric_value_cleaned) for li_idx, x in enumerate(el.line_items)]
            sources.append(loc.source)
        self.key_value_pairs = kv_pairs
        self.sources = sources
        self.set_identifiers()

    def add_empty_line_items(self, insert_idx: int, line_items: List[str]):
        new_kv = self.key_value_pairs[0:insert_idx]+[(x, None) for x in line_items]+self.key_value_pairs[insert_idx:]
        self.key_value_pairs = new_kv
        self.set_identifiers()

    def add_location(self, location: ParseeLocation, element: StructuredTable, local_col_idx: int, org_col_idx: int):
        self.locations.append(location)
        self._elements.append(element)
        self._local_col_indices.append(local_col_idx)
        self.org_col_indices.append(org_col_idx)
        self.build()

    def can_be_merged(self, col: FinalOutputTableColumn) -> bool:
        if len(col.key_value_pairs) != len(self.key_value_pairs) or col.li_identifier != self.li_identifier:
            return False
        for k, (li, val) in enumerate(self.key_value_pairs):
            check_li, check_val = col.key_value_pairs[k]
            if val is not None and check_val is not None:
                return False
        return True

    def merge_kv(self, col: FinalOutputTableColumn):
        # merges key value pairs with key value pairs of other column
        for k, (li, val) in enumerate(self.key_value_pairs):
            if val is None:
                self.key_value_pairs[k] = (li, col.key_value_pairs[k][1])
        self.set_identifiers()


class FinalOutputTable:
    detected_class: str
    columns: List[FinalOutputTableColumn]
    li_identifier: str
    line_items: List[str]

    def __init__(self, detected_class: str, columns: List[FinalOutputTableColumn], li_identifier: str):
        self.detected_class = detected_class
        self.columns = columns
        self.li_identifier = li_identifier
        self.line_items = [x[0] for x in columns[0].key_value_pairs]

    def li_number_matching(self) -> List[Tuple[str, Set[int]]]:
        output = []
        for k, li in enumerate(self.line_items):
            values = [clean_number_for_matching(col.key_value_pairs[k][1]) for col in self.columns]
            output.append((li, set(values)))
        return output

    def to_pandas(self) -> DataFrame:
        if len(self.columns) == 0:
            raise Exception("no columns found")
        header = ["line_item"]
        for col_idx, col in enumerate(self.columns):
            col_name = ("_".join([f"{x.class_id}:{x.class_value}" for x in col.meta])) if len(col.meta) > 0 else f"col{col_idx}"
            header.append(col_name)
        data = []
        for li_idx, li in enumerate(self.line_items):
            row = {header[0]: li}
            for k, col in enumerate(self.columns):
                row[header[k+1]] = col.key_value_pairs[li_idx][1]
            data.append(row)
        return DataFrame(data)


class ElementGroup:
    detected_class: str
    components: List[ParseeLocation]

    def __init__(self, detected_class: str, base_component: ParseeLocation, collapse_columns: bool = False):
        self.detected_class = detected_class
        self.components = [base_component]
        self.collapse_columns = collapse_columns

    def prob_combined(self) -> float:
        return float(np.mean([x.prob for x in self.components])) if len(self.components) > 0 else 0

    def closest_el(self, other: ElementGroup) -> ParseeLocation:
        el_sorted = list(sorted(self.components, key=lambda x: abs(x.source.element_index - other.components[0].source.element_index)))
        return el_sorted[0]

    def base_el(self) -> ParseeLocation:
        return self.components[0]

    def merge_with(self, other: ElementGroup):
        # merge elements
        self.components += other.components

    def structured_values(self, elements: List[ExtractedEl]) -> List[FinalOutputTableColumn]:

        # sort by element index
        components_sorted = list(sorted(self.components, key=lambda x: x.source.element_index))

        # collect all relevant elements
        elements_by_idx = {}
        for comp in components_sorted:
            el = elements[comp.source.element_index]
            elements_by_idx[comp.source.element_index] = el

        # build / update reference line items
        li_reference_list: Dict[int, List[str]] = {}
        li_placed_by_col: Dict[int, Set[int]] = {}
        for loc_k, location in enumerate(components_sorted):
            current_element: StructuredTable = elements_by_idx[location.source.element_index]
            li_reference_list[loc_k] = [x.clean_caption() for x in current_element.line_items]

        submissions_by_col = {}
        for loc_k, location in enumerate(components_sorted):
            # check that element is really a table
            if elements_by_idx[location.source.element_index].el_type == ElementType.TEXT:
                continue
            # index of last placed column from the current table
            placed_previous_index: Union[None, int] = None
            current_element: StructuredTable = elements_by_idx[location.source.element_index]
            current_numeric_col_indices = current_element.numeric_cols_indices
            for col_index_local, col_index_org in enumerate(current_numeric_col_indices):
                col_index_final = None
                # place one right of previously placed from same structured table if possible
                if col_index_final is None and placed_previous_index is not None:
                    col_index_final = placed_previous_index + 1
                # if column number is different for current structured table and submission data, see if left or right align of tables works better using the mean of the values of the columns
                if col_index_final is None and loc_k > 0 and len(current_numeric_col_indices) != len(submissions_by_col.keys()):
                    # calculate mean of cols for current table
                    current_table_col_values = {}
                    for idx_local, idx_org in enumerate(current_numeric_col_indices):
                        current_table_col_values[idx_local] = get_mean_for_column(current_element.get_values_for_column(idx_org))
                    # calculate mean for cols for existing final table
                    existing_values_by_col = {key: get_mean_for_column([x[1] for x in values_col.key_value_pairs]) for
                                              key, values_col in
                                              submissions_by_col.items()}
                    # calculate scores for left align
                    left_align_score = 0
                    right_align_score = 0
                    val_current, val_existing = list(current_table_col_values.values()), list(existing_values_by_col.values())
                    for c in range(0, min([len(current_table_col_values.keys()), len(existing_values_by_col.keys())])):
                        score_left = abs(val_current[c] - val_existing[c])
                        score_right = abs(val_current[-(1 + c)] - val_existing[-(1 + c)])
                        left_align_score += score_left
                        right_align_score += score_right

                    # align left
                    if left_align_score < right_align_score:
                        col_index_final = 0
                    # align right
                    else:
                        if len(current_element.numeric_cols_indices) > len(submissions_by_col.keys()):
                            # current table has more than final combined table, insert at position 0 and change all the keys of the final table
                            col_index_final = 0
                            modified_keys_dict = {key + 1: dict_values for key, dict_values in submissions_by_col.items()}
                            submissions_by_col = modified_keys_dict
                        else:
                            # current table has less keys than the combined table, insert so that it aligns on the right
                            diff = len(submissions_by_col.keys()) - len(current_element.numeric_cols_indices)
                            col_index_final = col_index_local + diff
                # take local index if no better matching possible
                if col_index_final is None:
                    col_index_final = col_index_local
                # update placed previous
                placed_previous_index = col_index_final
                if col_index_final not in submissions_by_col:
                    submissions_by_col[col_index_final] = FinalOutputTableColumn(location, current_element, col_index_final, col_index_local, col_index_org)
                else:
                    submissions_by_col[col_index_final].add_location(location, current_element, col_index_local, col_index_org)
                # update li placed dict
                if col_index_final not in li_placed_by_col:
                    li_placed_by_col[col_index_final] = set()
                li_placed_by_col[col_index_final].add(loc_k)

        # add missing line items to make sure that all columns have the same "length"
        for col_idx, col in submissions_by_col.items():
            if len(li_placed_by_col[col_idx]) < len(components_sorted):
                # add missing items
                for el_idx, li_values in li_reference_list.items():
                    if el_idx not in li_placed_by_col[col_idx]:
                        start_idx = sum([len(li_reference_list[x]) for x in range(0, el_idx)]) if el_idx > 0 else 0
                        col.add_empty_line_items(start_idx, li_values)

        # collapse columns from right to left if required and possible
        to_del = set()
        if self.collapse_columns:
            for key_idx in range(len(submissions_by_col.keys())-1,-1,-1):
                key = list(submissions_by_col.keys())[key_idx]
                # check if columns can be merged
                if key_idx > 0 and submissions_by_col[key].can_be_merged(submissions_by_col[list(submissions_by_col.keys())[key_idx-1]]):
                    submissions_by_col[list(submissions_by_col.keys())[key_idx - 1]].merge_kv(submissions_by_col[key])
                    to_del.add(key)
        return [x for k, x in submissions_by_col.items() if k not in to_del]


class StructuredLineItem:
    caption = None
    caption_elements = None
    value_elements = None
    row_index = None

    def __init__(self, from_elements, value_elements, row_index):

        self.value_elements = value_elements
        self.caption_elements = from_elements
        self.row_index = row_index

        self.caption = ""
        elements_used = []
        # go element by element to not double use one
        for el in from_elements:
            if el not in elements_used and el.val is not None:
                elements_used.append(el)
                if self.caption != "":
                    self.caption += " "
                self.caption += el.val.strip()

        # caption can't be empty string
        if self.caption.strip() == "":
            self.caption = "n/a"

    def __str__(self):
        return str(self.caption) + ": " + str([x.numeric_value_cleaned for x in self.value_elements])

    def __repr__(self):
        return str(self)

    def clean_caption(self):
        return clean_spaces(self.caption)


class ExtractedEl:
    el_type: ElementType
    source: ExtractedSource
    source_id: str

    def __init__(self, el_type: ElementType, source: ExtractedSource, text: Union[str, None] = None):
        self.el_type = el_type
        self.source = source
        self.text = text
        self.source_id = source.to_location_id()

    def __str__(self):
        return str(self.el_type) + ": " + repr(self.get_text())

    def __repr__(self):
        return self.__str__()

    def to_json_dict(self):
        return {"el_type": self.el_type.value, "source": self.source.to_json_dict(), "text": self.text, "source_id": self.source_id}

    def get_text(self, *args):
        return self.text

    def get_text_llm(self, contain_numbers: bool) -> str:
        return self.get_text()

    def get_data_simplified(self):
        return self.get_text()

    def get_text_before(self, all_elements: List[ExtractedEl], char_limit: int):
        text_before_words = ""
        # compile text before element to be included for features
        for k in range(self.source.element_index - 1, -1, -1):
            if all_elements[k].el_type == ElementType.TEXT:
                text_before_words = all_elements[k].get_text() + " " + text_before_words
                if len(text_before_words) >= char_limit:
                    break
            else:
                return text_before_words


class StructuredTableCell:
    val = None
    numeric_value_cleaned = None
    text_value_cleaned = None
    cell_type = None
    colspan = 1
    source = None

    # object is only valid if it is visible etc.
    valid = True

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return self.__str__()

    def __init__(self, val: Union[str, Decimal, int], colspan: int = 1):

        self.val = val
        self.colspan = colspan
        self.clean_cell()

    def to_json_dict(self):
        return {"val": self.val, "colspan": self.colspan}

    def contains_value(self):
        if self.val is None:
            return False
        if self.val.strip() == "":
            return False
        return True

    # cleans numeric values and sets cell type
    def clean_cell(self):

        # replace spaces
        self.val = self.val.replace(u'\xa0', u' ')

        # insert spaces if brackets too close
        self.val = re.sub(r'([^\s\(\)]+)(\()(.+\))', r'\g<1> (\g<3>', self.val)

        if is_number_cell(self.val):
            self.numeric_value_cleaned = clean_numeric_value(self.val)
            self.cell_type = "numeric"
        else:
            composition = composition_percentages(self.val)
            if composition['text'] > 0:
                self.cell_type = "text"
            elif composition['special'] > 0:
                self.cell_type = "special"
            else:
                self.cell_type = "null"
            self.text_value_cleaned = self.val.strip()

    def clean_value(self):
        return self.numeric_value_cleaned if self.cell_type == "numeric" else self.text_value_cleaned


class StructuredRow:
    row_type = None
    values = None
    # this is a list of values with all cells having colspan=1
    values_normalized = None
    # this is the final list of values (no duplicate columns etc)
    final_values = None

    def __str__(self):
        return "row: " + str(self.final_values)

    def __repr__(self):
        return self.__str__()

    def __init__(self, row_type: str, values: List[StructuredTableCell]):

        self.row_type = row_type
        self.values = values
        self.values_normalized = []
        self.final_values = []

        for val_obj in self.values:
            for a in range(0, val_obj.colspan):
                self.values_normalized.append(val_obj)

    def to_json_dict(self):
        return {"row_type": self.row_type, "values": [x.to_json_dict() for x in self.values]}

    def has_values(self, max_col_index=None):
        for col_index, v in enumerate(self.final_values):
            if max_col_index is not None and col_index > max_col_index:
                return False
            if v.contains_value():
                return True
        return False

    def add_value(self, cell_obj):
        if cell_obj.valid:
            self.values.append(cell_obj)
            for a in range(0, cell_obj.colspan):
                self.values_normalized.append(cell_obj)

    def finalise_values(self, ignored_col_indices):
        for k in range(0, len(self.values_normalized)):
            if k not in ignored_col_indices:
                self.final_values.append(self.values_normalized[k])

    def merge_columns(self, to_merge):
        # first column is kept only
        to_delete = [x['org'] for k, x in enumerate(to_merge) if k > 0]
        to_keep_index = to_merge[0]['org']
        for del_idx in to_delete:
            # transfer value if not set yet
            if not self.final_values[to_keep_index].contains_value():
                self.final_values[to_keep_index].val = self.final_values[del_idx].val
        self.final_values[to_keep_index].clean_cell()
        # delete columns
        self.final_values = [x for k, x in enumerate(self.final_values) if k not in to_delete]


class StructuredTable(ExtractedEl):
    rows = None
    numeric_values = None

    # for structuring of table
    header_rows = None
    line_items = None
    # not line items nor header rows
    other_rows = None
    numeric_cols_indices = None
    numeric_rows_indices = None
    empty_columns: List[int]

    def __str__(self):
        return str(self.rows)

    def __repr__(self):
        return self.__str__()

    def __init__(self, source: ExtractedSource, rows: List[StructuredRow]):
        super().__init__(ElementType.TABLE, source)
        self.rows = rows

        self.finalise_table()

        # structure table
        self.structure_table()

    def to_json_dict(self):
        return {**super().to_json_dict(), "rows": [x.to_json_dict() for x in self.rows]}

    def get_text(self, insert_li_break_text=None):
        text_pieces = []

        for row in self.rows:
            if insert_li_break_text is not None:
                text_pieces.append(insert_li_break_text)
            for val_obj in row.final_values:
                if val_obj.val is not None:
                    text_pieces.append(str(val_obj.val))
        return " ".join(text_pieces)

    def get_text_llm(self, contain_numbers: bool) -> str:
        text_pieces = []

        for row_idx, row in enumerate(self.rows):
            if contain_numbers:
                text_pieces.append(f"(row {row_idx}) \n")
            col_idx = 0
            last_val_written = None
            for k, val_obj in enumerate(row.final_values):
                if k not in self.empty_columns:
                    if contain_numbers:
                        if val_obj.val is not None and val_obj.val.strip() != "":
                            text_pieces.append(f"(col {col_idx}): {val_obj.val}")
                    else:
                        if val_obj.val is not None and not is_number_cell(val_obj.val) and len(words_contained(val_obj.val)) > 0 and (last_val_written is None or str(val_obj.val) != last_val_written):
                            text_pieces.append(str(val_obj.val))
                            last_val_written = str(val_obj.val)
                    col_idx += 1
            if contain_numbers:
                text_pieces.append(f" (row end);\n")
        if contain_numbers:
            text = " ".join(text_pieces)
        else:
            text = ", ".join(text_pieces)
        text = re.sub(r' +', ' ', text)
        return "table: " + text

    def get_text_and_surrounding_elements_text(self, elements: List[ExtractedEl]):

        main_text = self.get_text_llm(True)

        max_elements = 3

        text_pieces = []
        for a in range(self.source.element_index-1, max(-1, self.source.element_index-max_elements-1), -1):
            if isinstance(elements[a], StructuredTable):
                break
            text_pieces.append(elements[a].get_text_llm(True))
        text_pieces.reverse()
        text_before = "\n".join(text_pieces)
        return f"Table element - Text before table: {text_before}; {main_text}"

    def get_data_simplified(self):
        rows = []

        for row in self.rows:
            temp_row = []
            for cell_obj in row.final_values:
                temp_row.append({"colspan": cell_obj.colspan, "val": cell_obj.val})
            rows.append(temp_row)

        return rows

    def num_columns_final(self):
        if len(self.rows) > 0:
            return len(self.rows[0].final_values)
        return 0

    def dimension(self):
        return len(self.rows), self.num_columns_final()

    def df_format(self):
        output = []

        for row in self.rows:
            row_helper = {}
            for col_index in range(0, self.num_columns_final()):
                row_helper["col_" + str(col_index)] = row.final_values[col_index]
            output.append(row_helper)
        return output

    def finalise_table(self):
        # determine number of columns
        # this includes the colspan, colspan > 1 is like an additional cell
        col_numbers = [len(row.values_normalized) for row in self.rows]
        num_columns_normalized = max(col_numbers) if len(col_numbers) > 0 else 0

        # connect all cells
        for row_index in range(0, len(self.rows)):
            for col_index in range(0, num_columns_normalized):
                # item is not present, create empty one
                if col_index > len(self.rows[row_index].values_normalized) - 1:
                    val_obj = StructuredTableCell("")
                    self.rows[row_index].add_value(val_obj)

        # consolidate/delete columns with exactly equal data
        duplicate_columns = []
        for col_index in range(1, num_columns_normalized):
            is_identical_with_previous = True
            for row in self.rows:
                # object comparison works here
                if row.values_normalized[col_index] != row.values_normalized[col_index - 1] and (
                        row.values_normalized[col_index].contains_value() or row.values_normalized[col_index - 1].contains_value()):
                    is_identical_with_previous = False
                    break
            if is_identical_with_previous:
                duplicate_columns.append(col_index)

        for row in self.rows:
            row.finalise_values(duplicate_columns)

    def get_numeric_values_for_matching(self):

        numeric_values = []

        for row in self.rows:
            for col_index in range(0, self.num_columns_final()):
                cell = row.final_values[col_index]
                if cell.cell_type == "numeric" and cell.clean_value() is not None:
                    # do some more transformations, take abs and remove trailing zeros
                    val_cleaned = clean_number_for_matching(cell.clean_value())
                    numeric_values.append(val_cleaned)

        return numeric_values

    def get_values_for_column(self, col_index: int) -> List:
        values = []
        for row in self.rows:
            values.append(row.final_values[col_index].numeric_value_cleaned)
        return values

    # detect line items, clean values, detect header rows
    def structure_table(self):

        self.empty_columns = []

        # min number of numeric cells per column
        min_numeric = 1

        # determine numeric columns
        numeric_cols = []
        for col_index in range(0, self.num_columns_final()):
            col_value_types = [self.rows[x].final_values[col_index].cell_type for x in range(0, len(self.rows))]

            num_numeric = len([x for x in col_value_types if x == "numeric"])
            num_text = len([x for x in col_value_types if x == "text"])
            if num_numeric >= min_numeric and num_numeric > num_text:
                numeric_cols.append(col_index)
            if len(set(col_value_types)) == 1 and col_value_types[0] == "null":
                self.empty_columns.append(col_index)

        # check if numeric columns can be consolidated because of duplicates
        numeric_cols = list(sorted(numeric_cols))
        if len(numeric_cols) > 1:
            duplicate_cols = []
            # compile all values for all columns
            all_values = []
            for col_index in numeric_cols:
                all_values.append([self.rows[x].final_values[col_index] for x in range(0, len(self.rows)) if self.rows[x].final_values[col_index].cell_type == "numeric"])
            for k, col_index in enumerate(numeric_cols):
                all_contained_in_next = True
                if k < len(numeric_cols) - 1:
                    for val_obj in all_values[k]:
                        if val_obj not in all_values[k + 1]:
                            all_contained_in_next = False
                            break
                else:
                    all_contained_in_next = False

                # mark as duplicate
                if all_contained_in_next:
                    duplicate_cols.append(col_index)
                else:
                    # check that values are not in previous column AND the current column has less values
                    if k > 0 and len(all_values[k]) < len(all_values[k - 1]):
                        not_contained_in_previous = [x for x in all_values[k] if x not in all_values[k - 1]]
                        if len(not_contained_in_previous) == 0:
                            duplicate_cols.append(col_index)

            numeric_cols = [x for x in numeric_cols if x not in duplicate_cols]

        self.numeric_cols_indices = numeric_cols

        # determine numeric rows
        numeric_rows = []
        for row_index in range(0, len(self.rows)):
            # rows with type header can't be numeric
            if self.rows[row_index].row_type == "header":
                continue
            # this is to exclude years at the very top
            for numeric_col_index in numeric_cols:
                cell_obj = self.rows[row_index].final_values[numeric_col_index]

                if cell_obj.cell_type == "numeric" and not is_year_cell(cell_obj.val):
                    numeric_rows.append(row_index)

        self.numeric_rows_indices = list(sorted(list(set(numeric_rows))))

        # make line items
        # get all numeric values so that they dont get merged with labels
        all_numeric_values = []
        for row_index in self.numeric_rows_indices:
            for col_index in self.numeric_cols_indices:
                if self.rows[row_index].final_values[col_index].cell_type == "numeric":
                    all_numeric_values.append(self.rows[row_index].final_values[col_index])

        self.line_items = []
        for row_index in self.numeric_rows_indices:

            li_caption_items = [x for x in self.rows[row_index].final_values if x not in all_numeric_values]

            li = StructuredLineItem(li_caption_items, [self.rows[row_index].final_values[col_index] for col_index in numeric_cols], row_index)
            self.line_items.append(li)

        # determine header area and other
        self.header_rows = []
        self.other_rows = []
        if len(self.numeric_rows_indices) > 0:
            min_numeric_row_index = min(self.numeric_rows_indices)
            for row_index in range(0, len(self.rows)):
                if row_index < min_numeric_row_index:
                    self.header_rows.append(self.rows[row_index])
                elif row_index not in self.numeric_rows_indices:
                    self.other_rows.append(self.rows[row_index])


@dataclass
class StandardDocumentFormat:
    source_type: DocumentType
    source_identifier: str
    elements: List[ExtractedEl]
    file_path: Optional[str]

    def to_json_dict(self):
        return {"source_type": self.source_type.value, "source_identifier": self.source_identifier, "elements": [x.to_json_dict() for x in self.elements]}

    def __str__(self):
        return self.to_string(False)

    def __repr__(self):
        return str(self)

    def to_string(self, show_chunk_index: bool):
        output = ""
        for el in self.elements:
            output += (f"[chunk {el.source.element_index}] " if show_chunk_index else "") + el.get_text_llm(True) + "\n"
        return output


@dataclass
class FileReference:
    source_identifier: str
    source_type: DocumentType
    element_index: Optional[int] = None

    def __str__(self):
        return f"[FILE] id: {self.source_identifier}"

    def __repr__(self):
        return str(self)

    def reference_id(self) -> str:
        return f"{self.source_identifier}_{self.source_type}_{self.element_index}"
