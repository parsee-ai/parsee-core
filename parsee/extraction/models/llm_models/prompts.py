from typing import *

from parsee.datasets.dataset_dataclasses import DatasetRow
from parsee.extraction.extractor_dataclasses import Base64Image


class Prompt:

    def __init__(self, description: str, main_task: str, additional_info: str, full_example: str, available_data: Union[str, List[Base64Image]]):

        self.description = description
        self.main_task = main_task
        self.additional_info = additional_info
        self.full_example = full_example
        self.available_data = available_data

    def __str__(self) -> str:
        return f'''{self.description} \n

                    {self.main_task} \n
                    
                    {self.additional_info} \n

                    {self.full_example} \n
                    
                    {self.available_data if isinstance(self.available_data, str) else ''}
                    '''
