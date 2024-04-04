from typing import *
import json
import decimal

from parsee.datasets.dataset_dataclasses import MappingUniqueIdentifier
from parsee.extraction.extractor_elements import FinalOutputTable
from parsee.extraction.extractor_dataclasses import ParseeBucket
from parsee.utils.helper import clean_text_for_word_vectors2
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.templates.mappings import MappingBucket
from parsee.datasets.dataset_dataclasses import DatasetRow
from parsee.extraction.tasks.mappings.mapping_model import MappingSchema
from parsee.utils.sample_items import samples


class MappingFeatureBuilder:

    _memory: Dict[MappingUniqueIdentifier, Dict]

    def __init__(self):
        self._memory = {}
        self.number_replacement = "xnumberx"

    def get_text_all(self, table: FinalOutputTable, schema_id: str) -> str:

        items = []
        for kv_index, item in enumerate(table.line_items):
            items.append(self._make_base_features(table, schema_id, kv_index)["caption_cleaned"])

        return ", ".join(items)

    def _get_cleaned_caption(self, table: FinalOutputTable, schema_id: str, kv_index: int) -> str:
        if kv_index < 0 or kv_index > len(table.line_items)-1:
            return ""
        features = self._make_base_features(table, schema_id, kv_index)
        return features["caption_cleaned"]

    def _make_base_features(self, table: FinalOutputTable, schema_id: str, kv_index: int) -> Dict[str, str]:

        unique_identifier = MappingUniqueIdentifier(schema_id, table.li_identifier, kv_index)

        if unique_identifier in self._memory:
            return self._memory[unique_identifier]

        line_item_org = table.line_items[kv_index]
        total_items = len(table.line_items)

        self._memory[unique_identifier] = {"class_id": table.detected_class, "caption_org": line_item_org, "caption_cleaned": clean_text_for_word_vectors2(line_item_org, None, True, True, self.number_replacement), "item_idx": kv_index, "total_items": total_items}

        return self._memory[unique_identifier]

    def make_features(self, source_identifier: Optional[str], template_id: Optional[str], table: FinalOutputTable, schema_id: str, kv_index: int) -> DatasetRow:

        base_features = self._make_base_features(table, schema_id, kv_index)
        unique_identifier = MappingUniqueIdentifier(schema_id, table.li_identifier, kv_index)

        all_text = self.get_text_all(table, schema_id)

        item_before = self._get_cleaned_caption(table, schema_id, kv_index-1)
        item_after = self._get_cleaned_caption(table, schema_id, kv_index+1)

        features = {**base_features, "all_items": all_text, "item_before": item_before, "item_after": item_after}

        return DatasetRow(source_identifier, template_id, unique_identifier, features)


class LLMMappingFeatureBuilder(MappingFeatureBuilder):

    def __init__(self):
        super().__init__()
        self.number_replacement = "[number]"

    def build_raw_answer(self, choices: List[ParseeBucket], schema: MappingSchema) -> str:
        output = {}
        for bucket in schema.buckets:
            output[bucket.id] = [self.line_item_formatter(x.kv_index) for x in choices if x.bucket_id == bucket.id]
        return json.dumps(output)

    def item_id_string(self, bucket: MappingBucket):
        return f"{bucket.id}"

    def format_table(self, table: FinalOutputTable) -> str:
        output = []
        for k, line_item in enumerate(table.line_items):
            row = [f"{self.line_item_formatter(k)}: {line_item}"]
            for col in table.columns:
                row.append(str(col.key_value_pairs[k][1]))
            output.append(row)
        return json.dumps(output)

    def line_item_formatter(self, line_item_idx: int) -> str:
        return f"LI{line_item_idx}"

    def full_example(self) -> str:
        sample_table = samples.table
        schema = samples.schema
        output = {}
        if len(sample_table.line_items) != 3 or len(schema.buckets) != 3:
            raise Exception("needs 3 items")
        output[schema.buckets[0].id] = [self.line_item_formatter(0), self.line_item_formatter(1)]
        output[schema.buckets[1].id] = [self.line_item_formatter(2)],
        output[schema.buckets[2].id] = []
        return json.dumps(output)

    def format_schema(self, schema: MappingSchema):
        bucket_str = ""
        for schema_item in schema.buckets:
            bucket_str += f"{self.item_id_string(schema_item)}: {schema_item.caption}" + (f" ({schema_item.description})"if schema_item.description is not None and schema_item.description.strip() != "" else "") +"\n"
        return bucket_str

    def build_prompt(self, table: FinalOutputTable, schema: MappingSchema) -> Prompt:

        return Prompt(
            "You are supposed to classify line-items from a table into several possible buckets. Each item can only be placed in one bucket. If possible, make sure that buckets that represent a sum of some other items are roughly adding up, using the numbers in the table.",
            f"The table with data will be provided as an array of rows, each line item has an ID that will be included in the first cell at the beginning. After the first cell, all other cells are the values of the line item for each column. The buckets are presented with an ID followed by a descriptive name.", "Format your output as a JSON dictionary, where the keys correspond to the IDs of the buckets and the values are arrays of the line item IDs.",
            f"For example for the following table: {self.format_table(samples.table)} \n"
            f"And the following buckets: {self.format_schema(samples.schema)}"
            f"a valid output would be: {self.full_example()}",
            f"The actual available buckets are the following: {self.format_schema(schema)}\n"
            f"The actual table with line items to be classified is the following: {self.format_table(table)}"
        )

    def make_prompt(self, table: FinalOutputTable, schema: MappingSchema) -> Prompt:

        return self.build_prompt(table, schema)