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

    def __key(self):
        return (self.name, self.model_id, self.internal_name, self.model_type, self.file_path, self.price_per_1k_tokens,
                self.price_per_1k_output_tokens, self.price_per_image, self.max_tokens, self.api_key,
                self.only_questions, self.only_elements, self.only_meta, self.only_mappings, self.stats,
                self.multimodal, self.max_images, self.max_image_pixels, self.max_output_tokens, self.system_message)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, MlModelSpecification):
            return self.__key() == other.__key()
        return NotImplemented