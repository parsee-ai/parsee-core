import json
from typing import List, Optional, Dict, Union, Any
from pydantic.dataclasses import dataclass

from parsee.utils.enums import OutputType, ContextType, SearchStrategy


@dataclass
class StructuringItemSchema:
    type: OutputType
    context: ContextType
    id: str
    title: str
    additionalInfo: str
    model: Optional[str]
    searchStrategy: SearchStrategy
    example: Union[None, str]
    keywords: Union[None, str]
    valuesList: Union[None, List[str]]
    defaultValue: Union[None, str]
    customArgsJson: Union[None, str]

    def parsed_json_args(self) -> any:
        if self.customArgsJson is None:
            return None
        try:
            return json.loads(self.customArgsJson)
        except Exception as e:
            return None

    def __hash__(self):
        return hash(self.id)

    def to_json_dict(self) -> Dict:
        return {"type": self.type.value, "context": self.context.value, "id": self.id, "title": self.title, "additionalInfo": self.additionalInfo,
                "model": self.model, "searchStrategy": self.searchStrategy.value, "example": self.example, "keywords": self.keywords, "valuesList": self.valuesList,
                "defaultValue": self.defaultValue, "customArgsJson": self.customArgsJson
                }


@dataclass
class StructuringSchema:
    items: List[StructuringItemSchema]
    settings: Dict[str, Any]

    def to_json_dict(self) -> Dict:
        return {"items": [x.to_json_dict for x in self.items], "settings": self.settings}


@dataclass
class GeneralQueryItemSchema (StructuringItemSchema):
    metaInfoIds: List[str]

    def to_json_dict(self) -> Dict:
        return {**super().to_json_dict(), "metaInfoIds": self.metaInfoIds}


@dataclass
class GeneralQuerySchema:
    items: List[GeneralQueryItemSchema]
    settings: Dict[str, Any]

    def to_json_dict(self) -> Dict:
        return {"items": [x.to_json_dict() for x in self.items], "settings": self.settings}
