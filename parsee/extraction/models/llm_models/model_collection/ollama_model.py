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
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.extractor_dataclasses import Base64Image


class OllamaModel(LLMBaseModel):

    def __init__(self, model: MlModelSpecification):
        super().__init__(model)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_answer = 1024 if model.max_output_tokens is None else model.max_output_tokens
        self.max_tokens_question = self.spec.max_tokens - self.max_tokens_answer

        self.client = Client(host='http://localhost:11434' if model.file_path is None else model.file_path)

    def _call_api(self, prompt: str, images: List[Base64Image]) -> str:
        message_content = {
            'role': 'user',
            'content': prompt
        }
        if self.spec.multimodal:
            message_content['images'] = [x.data.encode() for x in images]
        messages = [message_content]
        if self.spec.system_message is not None:
            messages.insert(0, {
            'role': 'system',
            'content': self.spec.system_message
        })
        response = self.client.chat(model=self.spec.internal_name, messages=messages)
        return response["message"]["content"]

    def make_prompt_request(self, prompt: Prompt) -> Tuple[str, Decimal]:
        final_prompt, num_tokens_input = truncate_prompt(prompt, self.encoding, self.max_tokens_question)
        return self._call_api(final_prompt, prompt.available_data if self.spec.multimodal else []), Decimal(0)
