from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import datetime
from typing import List, Dict, Tuple, Union, Optional
from hashlib import sha256
from decimal import Decimal

from parsee.utils.enums import DocumentType


@dataclass
class ParseeBucket:
    schema_id: str
    bucket_id: str
    li_identifier: str
    model: str
    prob: float
    kv_index: int
    definition_idx: int
    multiplier: Decimal

    def dict_json(self) -> Dict:
        return {"schema_id": self.schema_id, "bucket_id": self.bucket_id, "li_identifier": self.li_identifier, "prob": self.prob, "kv_index": self.kv_index, "definition_idx": self.definition_idx, "multiplier": float(self.multiplier)}


@dataclass
class AssignedBucket:
    bucket_id: str
    kv_index: int
    definition_index: int
    multiplier: Decimal


@dataclass
class CoreExtractionRequest:
    source_type: str
    source_identifier: str


@dataclass
class ExtractedSource:
    source_type: DocumentType
    coordinates: Union[None, Dict]
    xpath: Union[None, str]
    element_index: int
    other_info: Union[Dict, None]

    def to_json_dict(self):
        return {"source_type": self.source_type.value, "coordinates": self.coordinates, "xpath": self.xpath, "element_index": self.element_index, "other_info": self.other_info}

    def identifier(self):
        return self.element_index

    def to_location_id(self) -> str:
        key = "coordinates" if self.coordinates is not None else "xpath"
        return sha256(str(self.to_json_dict()[key]).encode('utf-8')).hexdigest()


def source_from_json(json_dict: Dict) -> ExtractedSource:
    return ExtractedSource(DocumentType(json_dict["source_type"]), json_dict["coordinates"], json_dict["xpath"], json_dict["element_index"], json_dict["other_info"])


@dataclass
class ParseeLocation:
    model: str
    partial_prob: float
    detected_class: str
    prob: float
    source: ExtractedSource
    meta: List[ParseeMeta]


@dataclass
class AssignedLocation:
    class_id: str
    element_index: int
    is_partial: bool
    meta: List[AssignedMeta]
    mappings: List[AssignedBucket]


@dataclass
class ParseeMeta:
    model: str
    column_index: int
    source: List[ExtractedSource]
    class_id: str
    class_value: str
    prob: float

    def to_json_dict(self):
        return {"model": self.model, "class_id": self.class_id, "value": self.class_value, "prob": self.prob}


def meta_from_json(json_dict: Dict, col_idx: int, source: List[ExtractedSource]) -> ParseeMeta:
    return ParseeMeta(json_dict["model"], col_idx, source, json_dict["class_id"], json_dict["value"], json_dict["prob"])


@dataclass
class AssignedMeta:
    class_id: str
    class_value: str
    column_index: Optional[int] = None


@dataclass
class ParseeAnswer:
    model: str
    sources: List[ExtractedSource]
    class_id: str
    class_value: str
    raw_value: str
    parse_successful: bool
    meta: List[ParseeMeta]

    def to_json_dict(self):
        return {"class_id": self.class_id, "parsed": self.class_value, "raw": self.raw_value, "parseSuccessful": self.parse_successful, "meta": [x.to_json_dict() for x in self.meta], "sources": [x.to_json_dict() for x in self.sources]}

    def meta_hash(self) -> str:
        relevant_values = {"class_id": self.class_id, "meta": [x.to_json_dict() for x in self.meta]}
        return sha256(str(relevant_values).encode('utf-8')).hexdigest()

    def meta_key(self) -> str:
        if len(self.meta) == 0:
            return ""
        meta_sorted = sorted(self.meta, key=lambda x: x.class_id)
        return "_".join([f"{x.class_id}:{x.class_value}" for x in meta_sorted])

    def unique_id(self) -> str:
        return f"{self.class_id}:{self.meta_key()}"




@dataclass
class AssignedAnswer:
    class_id: str
    class_value: str
    meta: List[AssignedMeta]
    sources: List[ExtractedSource]
