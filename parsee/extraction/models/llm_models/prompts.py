from parsee.datasets.dataset_dataclasses import DatasetRow


class Prompt:

    def __init__(self, description: str, main_task: str, additional_info: str, full_example: str, available_text: str, summarize_if_overflow: bool):

        self.description = description
        self.main_task = main_task
        self.additional_info = additional_info
        self.full_example = full_example
        self.available_text = available_text
        self.summarize_if_overflow = summarize_if_overflow

    def __str__(self) -> str:
        return f'''{self.description} \n

                    {self.main_task} \n
                    
                    {self.additional_info} \n

                    {self.full_example} \n
                    
                    This is the available data to fulfill the task:\n
                    {self.available_text}
                    '''


class SummarizationPrompt(Prompt):

    def __init__(self, data: str):

        super().__init__("", "Please summarize the following text in 3-4 sentences maximum:", "", "", data, True)
