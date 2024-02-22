from pydantic.dataclasses import dataclass
from typing import Dict, Optional, List, Union

from parsee.templates.element_schema import ElementDetectionSchema
from parsee.templates.general_structuring_schema import GeneralQuerySchema, StructuringItemSchema
from parsee.extraction.models.model_dataclasses import MlModelSpecification


@dataclass
class JobTemplate:
    id: Union[None, str]
    title: str
    description: str
    questions: GeneralQuerySchema
    detection: ElementDetectionSchema
    meta: List[StructuringItemSchema]
    general_settings: Optional[Dict] = None

    # sets all models to one default model
    def set_default_model(self, model: MlModelSpecification):
        for item in self.questions.items:
            item.model = model.internal_name
        for item in self.detection.items:
            item.model = model.internal_name
            if item.mapRows is not None:
                item.mappingModel = model.internal_name
        for item in self.meta:
            item.model = model.internal_name

    def to_json_dict(self) -> Dict:
        return {"id": self.id, "title": self.title, "description": self.description, "questions": self.questions.to_json_dict(), "detection": self.detection.to_json_dict(), "meta": [x.to_json_dict() for x in self.meta]}