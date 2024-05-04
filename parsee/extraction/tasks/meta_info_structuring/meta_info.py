from typing import List, Dict, Optional

from parsee.extraction.extractor_elements import ExtractedEl, FinalOutputTableColumn
from parsee.templates.general_structuring_schema import StructuringItemSchema
from parsee.extraction.extractor_dataclasses import ParseeMeta, ExtractedSource
from parsee.extraction.tasks.meta_info_structuring.features import MetaFeatureBuilder


class MetaInfoModel:

    model_name = ""
    feature_builder: MetaFeatureBuilder

    def __init__(self, items: List[StructuringItemSchema]):
        self.items = items

    def predict_meta(self, columns: List[FinalOutputTableColumn], elements: List[ExtractedEl]) -> List[List[ParseeMeta]]:
        # returns meta information
        raise NotImplementedError
