from pydantic.dataclasses import dataclass
from typing import Dict, List, Any, Union, Optional

from parsee.utils.enums import SearchStrategy
from parsee.templates.mappings import MappingSchema
from parsee.templates.general_structuring_schema import StructuringItemSchema


@dataclass
class ElementSchema:
    id: str
    title: str
    additionalInfo: str
    keywords: str
    takeBestInProximity: bool
    model: Optional[str]
    searchStrategy: SearchStrategy
    mapRows: Union[None, MappingSchema]
    mappingModel: Optional[str]
    metaInfoIds: List[str]

    def to_json_dict(self) -> Dict:
        return {"id": self.id, "title": self.title, "additionalInfo": self.additionalInfo, "keywords": self.keywords,
                "takeBestInProximity": self.takeBestInProximity, "model": self.model, "searchStrategy": self.searchStrategy.value,
                "mapRows": self.mapRows.to_json_dict() if self.mapRows is not None else None,
                "mappingModel": self.mappingModel,
                "metaInfoIds": self.metaInfoIds
                }


@dataclass
class ElementDetectionSchema:
    items: List[ElementSchema]
    settings: Dict[str, Any]

    def to_json_dict(self) -> Dict:
        return {"items": [x.to_json_dict() for x in self.items], "settings": self.settings}
