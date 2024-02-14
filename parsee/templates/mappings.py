from pydantic.dataclasses import dataclass
from typing import Union, List, Dict
from decimal import Decimal

from parsee.utils.enums import AggregationMethod


@dataclass
class MappingBucket:
    id: str
    parentId: Union[str, None]
    caption: str
    description: Union[str, None]
    orderIndex: int
    multiplier: Decimal
    otherInfo: Dict
    aggregationMethod: AggregationMethod

    def to_json_dict(self) -> Dict:
        return {"id": self.id, "parentId": self.parentId, "caption": self.caption, "description": self.description,
                "orderIndex": self.orderIndex, "multiplier": self.multiplier, "otherInfo": self.otherInfo, "aggregationMethod": self.aggregationMethod.value}


@dataclass
class MappingSchema:
    id: str
    caption: str
    description: Union[str, None]
    buckets: List[MappingBucket]

    def to_json_dict(self) -> Dict:
        return {"id": self.id, "caption": self.caption, "description": self.description, "buckets": [x.to_json_dict() for x in self.buckets]}
