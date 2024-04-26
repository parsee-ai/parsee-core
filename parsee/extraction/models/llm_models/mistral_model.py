import os
import time
from typing import List, Tuple
from dataclasses import dataclass
from decimal import Decimal
import math

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import tiktoken

from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel, truncate_prompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.extractor_dataclasses import Base64Image


class MistralModel(LLMBaseModel):

    def __init__(self, model: MlModelSpecification):
        super().__init__(model)
        self.max_retries = 5
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_answer = 1024
        self.max_tokens_question = self.spec.max_tokens - self.max_tokens_answer
        self.client = MistralClient(api_key=model.api_key)

    def _call_api(self, prompt: str, images: List[Base64Image], retries: int = 0, wait: int = 5) -> Tuple[str, Decimal]:

        chat_response = self.client.chat(
            model=self.spec.internal_name,
            messages=[ChatMessage(role="user", content=prompt)]
        )

        answer = chat_response.choices[0].message.content

        final_cost = (Decimal(chat_response.usage.prompt_tokens + chat_response.usage.completion_tokens) * Decimal(self.spec.price_per_1k_tokens / 1000)) if self.spec.price_per_1k_tokens is not None else Decimal(0)
        return answer, final_cost

    def make_prompt_request(self, prompt: Prompt) -> Tuple[str, Decimal]:
        final_prompt, _ = truncate_prompt(str(prompt), self.encoding, self.max_tokens_question)
        return self._call_api(final_prompt, prompt.available_data if self.spec.multimodal else [])
