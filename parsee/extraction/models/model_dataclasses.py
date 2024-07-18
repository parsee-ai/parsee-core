from typing import *
from dataclasses import dataclass
from decimal import Decimal
import os

from parsee.utils.enums import ModelType


@dataclass
class MlModelSpecification:
    name: str
    model_id: str
    internal_name: str
    model_type: ModelType
    file_path: Optional[str]
    price_per_1k_tokens: Optional[Decimal]
    price_per_1k_output_tokens: Optional[Decimal]
    price_per_image: Optional[Decimal]
    max_tokens: Optional[int]
    api_key: Optional[str]
    only_questions: Optional[List[str]]
    only_elements: Optional[List[str]]
    only_meta: Optional[List[str]]
    only_mappings: Optional[List[str]]
    stats: Optional[Dict]
    multimodal: bool
    max_images: Optional[int]
    max_image_pixels: Optional[int]
    max_output_tokens: Optional[int]
    system_message: Optional[str]

    def model_path(self) -> Union[None, str]:
        if self.file_path is None:
            return None
        return self.file_path

    def settings_path(self) -> Union[None, str]:
        if self.file_path is None:
            return None
        model_dir = os.path.dirname(self.file_path)
        return os.path.join(model_dir, "settings.pkl")