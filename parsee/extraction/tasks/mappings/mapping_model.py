from typing import *
from decimal import Decimal

from parsee.extraction.extractor_dataclasses import ParseeBucket
from parsee.templates.element_schema import ElementSchema
from parsee.templates.mappings import MappingSchema
from parsee.datasets.dataset_dataclasses import MappingUniqueIdentifier
from parsee.extraction.tasks.mappings.features import MappingFeatureBuilder
from parsee.extraction.extractor_elements import FinalOutputTable


class MappingModel:

    model_name = ""
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
