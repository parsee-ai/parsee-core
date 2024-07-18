import os
import time
from typing import List, Tuple
from dataclasses import dataclass
from decimal import Decimal
import math

from together import Together
import tiktoken

from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel, get_tokens_encoded, truncate_prompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.llm_models.prompts import Prompt


class TogetherModel(LLMBaseModel):

    def __init__(self, model: MlModelSpecification):
        super().__init__(model)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_answer = 1024 if model.max_output_tokens is None else model.max_output_tokens
        self.max_tokens_question = self.spec.max_tokens - self.max_tokens_answer
        self.client = Together(api_key=model.api_key if model.api_key is not None else os.getenv("TOGETHER_API_KEY"), max_retries=5)

    def _call_api(self, prompt: str) -> Tuple[str, Decimal]:
        messages = [{"role": "user", "content": prompt}]
        if self.spec.system_message is not None:
            messages.insert(0, {"role": "system", "content": self.spec.system_message})
        response = self.client.chat.completions.create(
            model=self.spec.internal_name,
            messages=messages,
            temperature=0,
            max_tokens=self.max_tokens_answer,
            top_p=1,
        )
        answer = response.choices[0].message.content
        cost_input = (int(response.usage.prompt_tokens) * Decimal(self.spec.price_per_1k_tokens / 1000)) if self.spec.price_per_1k_tokens is not None else Decimal(0)
        cost_output = (int(response.usage.completion_tokens) * Decimal(self.spec.price_per_1k_output_tokens / 1000)) if self.spec.price_per_1k_output_tokens is not None else Decimal(0)
        final_cost = cost_input + cost_output
        return answer, final_cost

    def make_prompt_request(self, prompt: Prompt) -> Tuple[str, Decimal]:
        final_prompt, _ = truncate_prompt(prompt, self.encoding, self.max_tokens_question)
        return self._call_api(final_prompt)
