from typing import List, Dict, Optional

from src.extraction.extractor_elements import ExtractedEl, FinalOutputTableColumn
from src.extraction.templates.general_structuring_schema import StructuringItemSchema
from src.extraction.extractor_dataclasses import ParseeMeta, ExtractedSource
from src.utils.helper import clean_number_for_matching, determine_unit
from src.extraction.ml.tasks.meta_info_structuring.features import MetaFeatureBuilder
from src.datasets.dataset_dataclasses import MetaUniqueIdentifier


def detected_meta_from_json(json_dict: Dict[str, any], classifier_name: str, column_index: int, sources: List[ExtractedSource]) -> ParseeMeta:
    return ParseeMeta(classifier_name, column_index, sources, json_dict["class_id"], json_dict["value"], json_dict["prob"])


class MetaInfoClassifier:

    classifier_name = ""
    feature_builder: MetaFeatureBuilder

    def __init__(self, items: List[StructuringItemSchema]):
        self.items = items

    def predict_meta(self, columns: List[FinalOutputTableColumn], elements: List[ExtractedEl]) -> List[List[ParseeMeta]]:
        # returns meta information
        raise NotImplemented


class SimpleMetaInfoClassifier(MetaInfoClassifier):

    def __init__(self, items: List[StructuringItemSchema], manual_answers_meta: Optional[Dict] = None, manual_answers_meta_by_values: Optional[Dict] = None, **kwargs):
        super().__init__(items)
        self.classifier_name = "manual"
        self.manual_answers = manual_answers_meta
        self.manual_answers_by_values = manual_answers_meta_by_values
        self.feature_builder = MetaFeatureBuilder()

    def predict_meta(self, columns: List[FinalOutputTableColumn], elements: List[ExtractedEl]) -> List[List[ParseeMeta]]:
        """
        relies on manual predictions entirely
        """
        output = []
        for column in columns:
            if self.manual_answers is not None:
                identifier = MetaUniqueIdentifier(column.detected_class, column.col_idx, column.kv_identifier)
    
                if identifier in self.manual_answers:
                    output.append([detected_meta_from_json(x, self.classifier_name, column.col_idx, column.sources) for x in self.manual_answers[identifier]])
                    continue
            elif self.manual_answers_by_values is not None:
                col_output = []
                if column.detected_class in self.manual_answers_by_values:
                    candidates = self.manual_answers_by_values[column.detected_class]
    
                    number_ids = [x.number_id() for x in candidates]
                    column_number_id = set([clean_number_for_matching(x[1]) for x in column.key_value_pairs])
    
                    matches = [len(x.intersection(column_number_id)) / len(x) for x in number_ids]
                    best_match = max(matches)
                    if best_match > 0.8:
                        idx = matches.index(best_match)
                        labeled_col_chosen = candidates[idx]
                        for meta_id, meta_value in labeled_col_chosen.metaInfo.items():
                            col_output.append(ParseeMeta(self.classifier_name, column.col_idx, [], meta_id, meta_value, 1))
                        # check if unit should be determined
                        if labeled_col_chosen.inferUnit:
                            unit = determine_unit(labeled_col_chosen.values, [x[1] for x in column.key_value_pairs])
                            col_output.append(ParseeMeta(self.classifier_name, column.col_idx, [], "unit", unit, 1))
    
                output.append(col_output)
                continue

            # default case
            output.append([])

        return output
