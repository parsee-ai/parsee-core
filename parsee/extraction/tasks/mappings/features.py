from typing import *

from parsee.datasets.dataset_dataclasses import MappingUniqueIdentifier
from parsee.extraction.extractor_elements import FinalOutputTable
from parsee.extraction.extractor_dataclasses import ParseeBucket
from parsee.utils.helper import clean_text_for_word_vectors2
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.templates.mappings import MappingBucket
from parsee.datasets.dataset_dataclasses import DatasetRow
from parsee.extraction.tasks.mappings.mapping_model import MappingSchema


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

    def build_raw_answer(self, bucket_choice: ParseeBucket, schema: MappingSchema) -> str:
        mapping_bucket = [x for x in schema.buckets if x.id == bucket_choice.bucket_id]
        if len(mapping_bucket) == 0:
            return "n/a"
        return self.item_id_string(mapping_bucket[0])

    def item_id_string(self, bucket: MappingBucket):
        return f"[{bucket.id}]"

    def build_prompt(self, features: DatasetRow, schema: MappingSchema) -> Prompt:

        bucket_str = ""
        for schema_item in schema.buckets:
            bucket_str += f"{self.item_id_string(schema_item)}: {schema_item.caption}\n"
        return Prompt(
            "You are supposed to classify a line-item from a table into exactly one of several possible buckets.",
            f"The item is called: '{features.get_feature('caption_org')}' (index of item in table: {features.get_feature('item_idx')}). You are only supposed to classify this item.\n",
            f"{'' if features.get_feature('item_before') == '' else 'For reference, the item before the relevant item in the table is the following: '+features.get_feature('item_before')+'.'}" +
            f"{'' if features.get_feature('item_after') == '' else 'Also for reference, the item after the relevant item in the table is the following: '+features.get_feature('item_after')+'.'}",
            f"Each bucket has an ID, only use the ID of the bucket in your answer and nothing else. Your answer could be for example: {self.item_id_string(schema.buckets[0])}",
            f"The possible buckets are the following (each bucket name and description is preceded by the ID in square brackets): {bucket_str}"
        )

    def make_prompt(self, table: FinalOutputTable, schema: MappingSchema, kv_index: int) -> Prompt:

        features = self.make_features(None, None, table, schema.id, kv_index)

        return self.build_prompt(features, schema)