import os
import time
from typing import List, Tuple
from dataclasses import dataclass
from decimal import Decimal
import math

import tiktoken
import ollama
from ollama import Client

from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel, get_tokens_encoded, truncate_prompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification


class OllamaModel(LLMBaseModel):

    def __init__(self, model: MlModelSpecification):
        super().__init__(model.name)
        self.model = model
        self.max_retries = 0
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_answer = 1024
        self.max_tokens_question = self.model.max_tokens - self.max_tokens_answer

        self.client = Client(host='http://localhost:11434' if model.file_path is None else model.file_path)

    def _call_api(self, prompt: str, retries: int = 0, wait: int = 5) -> str:
        try:
            response = self.client.chat(model=self.model.internal_name, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            return response["message"]["content"]
        except Exception as e:
            if retries < self.max_retries:
                time.sleep(wait * 2 ** retries)
                return self._call_api(prompt, retries + 1)
            else:
                return ""

    def make_prompt_request(self, prompt: str) -> Tuple[str, Decimal]:
        final_prompt, num_tokens_input = truncate_prompt(prompt, self.encoding, self.max_tokens_question)
        return self._call_api(final_prompt), Decimal(0)
