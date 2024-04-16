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
        self.max_tokens_answer = 1024
        self.max_tokens_question = self.spec.max_tokens - self.max_tokens_answer
        self.client = Together(api_key=model.api_key if model.api_key is not None else os.getenv("TOGETHER_API_KEY"), max_retries=5)

    def _call_api(self, prompt: str) -> Tuple[str, Decimal]:
        response = self.client.chat.completions.create(
            model=self.spec.internal_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=self.max_tokens_answer,
            top_p=1,
        )
        answer = response.choices[0].message.content
        price = Decimal(int(response.usage.total_tokens) * (self.spec.price_per_1k_tokens / 1000)) if self.spec.price_per_1k_tokens is not None else Decimal(0)
        return answer, price

    def make_prompt_request(self, prompt: Prompt) -> Tuple[str, Decimal]:
        final_prompt, _ = truncate_prompt(str(prompt), self.encoding, self.max_tokens_question)
        return self._call_api(final_prompt)
