from typing import *

from parsee.datasets.dataset_dataclasses import DatasetRow
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.storage.interfaces import Modality


class Prompt:

    def __init__(self, intro: Optional[str], main_task: str, images: list[Base64Image] | None, text: str | None,
                 additional_info: Optional[str] = None,
                 full_example: Optional[str] = None,
                 history: Optional[List[str]] = None):
        self.intro = f"{intro} \n" if intro is not None else ""
        self.main_task = main_task
        self.additional_info = f"{additional_info} \n" if additional_info is not None else ""
        self.full_example = f"{full_example} \n" if full_example is not None else ""
        self.images = images
        self.text = text
        self.history = ""
        if history is not None and len(history) > 0:
            self.history = "[PREVIOUS MESSAGES]\n"
            self.history += "\n".join(history) + "\n [END PREVIOUS MESSAGES]\n"

    def __str__(self) -> str:
        return f'''{self.history} {self.instructions()} {self.available_data_string()}'''

    def instructions(self) -> str:
        return f"""{self.intro} {self.main_task} \n {self.additional_info} {self.full_example}"""

    def available_data_string(self) -> str:
        return self.text if self.text else ''

    def __key(self):
        available_data_key = hash(self.text) + hash(tuple(self.images))
        return self.intro, self.main_task, self.additional_info, self.full_example, available_data_key, self.history

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Prompt):
            result = self.__key() == other.__key()
            return result
        return NotImplemented
