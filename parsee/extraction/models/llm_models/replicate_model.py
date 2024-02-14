import os
import time
from typing import List, Tuple
from dataclasses import dataclass
from decimal import Decimal
import math

import tiktoken
import replicate

from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel, get_tokens_encoded, truncate_prompt
from parsee.extraction.models.llm_models.prompts import Prompt, SummarizationPrompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification


class ReplicateModel(LLMBaseModel):

    def __init__(self, model: MlModelSpecification):
        super().__init__(model.name)
        self.model = model
        self.max_retries = 5
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_answer = 256
        self.max_requests_for_summarization = 30
        self.max_tokens_question = self.model.max_tokens - self.max_tokens_answer

    def slice_data(self, data: str) -> List[str]:
        tokens = get_tokens_encoded(data, self.encoding)
        slices = []
        num_slices = math.ceil(len(tokens) / self.max_tokens_question)
        for slice_num in range(0, num_slices):
            current_slice = self.encoding.decode(tokens[slice_num*self.max_tokens_question:slice_num*self.max_tokens_question+self.max_tokens_question])
            slices.append(current_slice)
        return slices

    def _call_api(self, prompt: str, retries: int = 0, wait: int =5 ) -> Tuple[str, Decimal]:
        try:
            response = replicate.run(self.model.internal_name, input={
                "prompt": prompt,
                "top_k": 50,
                "top_p": 0.9,
                "temperature": 0,
                "max_new_tokens": 1024,
                "presence_penalty": 0,
                "frequency_penalty": 0
            })
            answer = "".join(response)
            final_cost = Decimal(0) # TODO
            return answer, final_cost
        except Exception as e:
            if retries < self.max_retries:
                time.sleep(wait * 2 ** retries)
                return self._call_api(prompt, retries + 1)
            else:
                return "", Decimal(0)

    def make_prompt_request(self, prompt: str) -> Tuple[str, Decimal]:
        return self._call_api(truncate_prompt(prompt, self.encoding, self.max_tokens_question))
