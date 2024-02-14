from typing import List, Dict, Optional
from functools import reduce

from src.utils.helper import clean_number_for_matching
from src.extraction.extractor_elements import StandardDocumentFormat
from src.extraction.templates.element_schema import ElementSchema
from src.extraction.extractor_dataclasses import ParseeLocation, AssignedLocation, ParseeMeta
from src.extraction.ml.tasks.element_classification.features import LocationFeatureBuilder
from src.utils.enums import ElementType


class ElementClassifier:

    classifier_name = ""
    feature_builder: LocationFeatureBuilder

    def __init__(self, items: List[ElementSchema], **kwargs):
        self.items = items

    def classify_elements(self, document: StandardDocumentFormat) -> List[ParseeLocation]:
        raise NotImplemented


class SimpleElementClassifier(ElementClassifier):

    def __init__(self, items: List[ElementSchema], manual_answers: Optional[Dict] = None, manual_answers_by_values: Optional[Dict] = None, **kwargs):
        super().__init__(items)
        self.classifier_name = "manual"

        self.manual_predictions = manual_answers
        self.manual_predictions_by_values = manual_answers_by_values
        self.feature_builder = LocationFeatureBuilder()

    def classify_elements(self, document: StandardDocumentFormat) -> List[ParseeLocation]:
        """
        relies on manual predictions entirely
        """
        output: List[ParseeLocation] = []
        if self.manual_predictions is not None:
            added = set()
            for answer in self.manual_predictions:
                class_value: str = answer["class_id"]
                partial: bool = answer['partial']
                el_idx = answer['element_index']
                id_tuple = (class_value, partial, el_idx)
                if id_tuple not in added:
                    el = document.elements[answer['element_index']]
                    output.append(ParseeLocation(self.classifier_name, float(partial), class_value, 2, el.source, []))
                    added.add(id_tuple)
        elif self.manual_predictions_by_values is not None:
            for item in self.items:
                if item.id in self.manual_predictions_by_values:
                    all_values = set(reduce(lambda a, b: a+b, [[clean_number_for_matching(y) for y in x.values] for x in self.manual_predictions_by_values[item.id]], []))
                    # go table by table and check if there is a sufficient match
                    candidates = []
                    for el in document.elements:
                        if el.el_type == ElementType.TABLE:
                            numeric_values = set(el.get_numeric_values())
                            if len(numeric_values) > 5 and len(all_values.intersection(numeric_values)) / len(all_values) > 0.8:
                                candidates.append(el)

                    if len(candidates) == 0:
                        continue
                    elif len(candidates) == 1:
                        output.append(ParseeLocation(self.classifier_name, 0, item.id, 2, candidates[0].source, []))
                    else:
                        # non partial matches
                        for candidate in candidates:
                            output.append(ParseeLocation(self.classifier_name, 0.7, item.id, 0.8, candidate.source, []))
        return output


class AssignedElementClassifier(ElementClassifier):

    def __init__(self, items: List[ElementSchema], truth_locations: List[AssignedLocation], **kwargs):
        super().__init__(items)
        self.classifier_name = "manual"
        self.assigned = truth_locations

        self.feature_builder = LocationFeatureBuilder()

    def classify_elements(self, document: StandardDocumentFormat) -> List[ParseeLocation]:
        output: List[ParseeLocation] = []
        added = set()
        for answer in self.assigned:
            id_tuple = (answer.class_id, answer.is_partial, answer.element_index)
            if id_tuple not in added:
                el = document.elements[answer.element_index]
                meta_values = [ParseeMeta(self.classifier_name, meta.column_index if meta.column_index is not None else 0, [el.source], meta.class_id, meta.class_value, 1) for meta in answer.meta]
                output.append(ParseeLocation(self.classifier_name, float(answer.is_partial), answer.class_id, 2, el.source, meta_values))
                added.add(id_tuple)

        return output
