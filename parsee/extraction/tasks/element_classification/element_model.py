from typing import List, Dict, Optional
from functools import reduce

from parsee.extraction.extractor_elements import StandardDocumentFormat
from parsee.templates.element_schema import ElementSchema
from parsee.extraction.extractor_dataclasses import ParseeLocation, AssignedLocation, ParseeMeta
from parsee.extraction.tasks.element_classification.features import LocationFeatureBuilder


class ElementModel:

    model_name = ""
    feature_builder: LocationFeatureBuilder

    def __init__(self, items: List[ElementSchema], **kwargs):
        self.items = items

    def classify_elements(self, document: StandardDocumentFormat) -> List[ParseeLocation]:
        raise NotImplemented


class AssignedElementModel(ElementModel):

    def __init__(self, items: List[ElementSchema], truth_locations: List[AssignedLocation], **kwargs):
        super().__init__(items)
        self.model_name = "manual"
        self.assigned = truth_locations

        self.feature_builder = LocationFeatureBuilder()

    def classify_elements(self, document: StandardDocumentFormat) -> List[ParseeLocation]:
        output: List[ParseeLocation] = []
        added = set()
        for answer in self.assigned:
            id_tuple = (answer.class_id, answer.is_partial, answer.element_index)
            if id_tuple not in added:
                el = document.elements[answer.element_index]
                meta_values = [ParseeMeta(self.model_name, meta.column_index if meta.column_index is not None else 0, [el.source], meta.class_id, meta.class_value, 1) for meta in answer.meta]
                output.append(ParseeLocation(self.model_name, float(answer.is_partial), answer.class_id, 2, el.source, meta_values))
                added.add(id_tuple)

        return output
