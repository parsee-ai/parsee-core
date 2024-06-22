from typing import *

from parsee.datasets.dataset_dataclasses import DatasetRow
from parsee.extraction.extractor_dataclasses import Base64Image


class Prompt:

    def __init__(self, description: Optional[str], main_task: str, additional_info: Optional[str] = None, full_example: Optional[str] = None, available_data: Optional[Union[str, List[Base64Image]]] = None, history: Optional[List[str]] = None):

        self.description = f"{description} \n" if description is not None else ""
        self.main_task = main_task
        self.additional_info = f"{additional_info} \n" if additional_info is not None else ""
        self.full_example = f"{full_example} \n" if full_example is not None else ""
        self.available_data = available_data if available_data is not None else ""
        self.history = ""
        if self.history is not None and len(self.history) > 0:
            self.history = "[PREVIOUS MESSAGES]\n"
            self.history += "\n".join(self.history) + "\n [END PREVIOUS MESSAGES]\n"

    def __str__(self) -> str:
        return f'''{self.history}{self.description}

                    {self.main_task} \n
                    
                    {self.additional_info}

                    {self.full_example}
                    
                    {self.available_data if isinstance(self.available_data, str) else ''}
                    '''
