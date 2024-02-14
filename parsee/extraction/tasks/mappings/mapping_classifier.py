from typing import *
from decimal import Decimal

from src.extraction.extractor_dataclasses import ParseeBucket
from src.extraction.templates.element_schema import ElementSchema
from src.extraction.templates.mappings import MappingSchema
from src.datasets.dataset_dataclasses import MappingUniqueIdentifier
from src.extraction.ml.tasks.mappings.features import MappingFeatureBuilder
from src.extraction.extractor_elements import FinalOutputTable
from src.utils.labeling_helper_dataclasses import SimpleMapping
from src.extraction.extractor_dataclasses import ID_NOT_AVAILABLE
from src.utils.helper import partial_ratio


class MappingClassifier:

    classifier_name = ""
    feature_builder: MappingFeatureBuilder

    def __init__(self, items: List[ElementSchema], **kwargs):
        self.items = items
        self.memory: Dict[str, List[ParseeBucket]] = {}

    def classify_with_schema(self, table: FinalOutputTable, schema: MappingSchema) -> List[ParseeBucket]:
        raise NotImplemented

    def classify_elements(self, table: FinalOutputTable) -> Tuple[List[ParseeBucket], Union[None, MappingSchema]]:
        for item in self.items:
            if item.id == table.detected_class and item.mapRows is not None:
                mapping_schema = item.mapRows
                full_signature = f"{mapping_schema.id}_{table.li_identifier}"
                if full_signature in self.memory:
                    return self.memory[full_signature], mapping_schema
                else:
                    self.memory[full_signature] = self.classify_with_schema(table, mapping_schema)
                    return self.memory[full_signature], mapping_schema
        return [], None


class SimpleMappingClassifier(MappingClassifier):

    def __init__(self, items: List[ElementSchema], manual_answers_mapping: Optional[Dict] = None, manual_answers_mapping_by_values: Optional[Dict] = None, **kwargs):
        super().__init__(items)
        self.classifier_name = "manual"
        self.manual_answers = manual_answers_mapping
        self.manual_answers_by_values = manual_answers_mapping_by_values
        self.feature_builder = MappingFeatureBuilder()

    def classify_with_schema(self, table: FinalOutputTable, schema: MappingSchema) -> List[ParseeBucket]:
        output: List[ParseeBucket] = []
        if self.manual_answers is not None:
            for k, v in enumerate(table.line_items):
                unique_identifier = MappingUniqueIdentifier(schema.id, table.li_identifier, k)
                if unique_identifier in self.manual_answers:
                    bucket_id = self.manual_answers[unique_identifier]["bucket_id"]
                    confidence = self.manual_answers[unique_identifier]["confidence"]
                    definition_index = self.manual_answers[unique_identifier]["definition_index"]
                    multiplier = self.manual_answers[unique_identifier]["multiplier"]
                    bucket_choice = ParseeBucket(schema.id, bucket_id, table.li_identifier, self.classifier_name, confidence, k, definition_index, multiplier)
                    output.append(bucket_choice)
        elif self.manual_answers_by_values is not None and table.detected_class in self.manual_answers_by_values:
            li_submitted = self.manual_answers_by_values[table.detected_class]
            numbers_by_row_idx = {}
            for k, (li_name, li_numbers) in enumerate(table.li_number_matching()):
                # check for best match
                matches = []
                for li_sub in li_submitted:
                    matches.append((li_sub, partial_ratio(li_name, li_sub[0]), len(li_numbers.intersection(li_sub[1])) / len(li_sub[1])))
                sorted_matches = sorted(matches, key=lambda x: (-x[2], -x[1]))
                if (sorted_matches[0][2] > 0.9 and sorted_matches[0][1] > 0.6) or (sorted_matches[0][1] > 0.9 and sorted_matches[0][2] > 0.5):
                    bucket_choice_assigned: SimpleMapping = sorted_matches[0][0][2]
                    bucket_choice = ParseeBucket(schema.id, bucket_choice_assigned.bucketId, table.li_identifier, self.classifier_name, 1, k, bucket_choice_assigned.definitionIndex, Decimal(bucket_choice_assigned.multiplier))
                else:
                    # no match found, assign bucket also
                    bucket_choice = ParseeBucket(schema.id, ID_NOT_AVAILABLE, table.li_identifier, self.classifier_name, 1, k, 0, Decimal(1))
                output.append(bucket_choice)

        return output
