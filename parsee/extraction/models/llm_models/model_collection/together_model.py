import os
from functools import lru_cache
from typing import Tuple
from decimal import Decimal

from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_random_exponential, after_log
from together import Together
import tiktoken
from together.error import RateLimitError

from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel, truncate_prompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.settings import chat_settings
import logging

logger = logging.getLogger(__name__)


class TogetherModel(LLMBaseModel):

    def __init__(self, model: MlModelSpecification):
        super().__init__(model)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_answer = 1024 if model.max_output_tokens is None else model.max_output_tokens
        self.max_tokens_question = self.spec.max_tokens - self.max_tokens_answer
        self.client = Together(api_key=model.api_key if model.api_key is not None else os.getenv("TOGETHER_API_KEY"), max_retries=5)

    @retry(reraise=True,
           stop=stop_after_attempt(chat_settings.retry_attempts),
           retry=retry_if_exception_type(RateLimitError),
           wait=wait_random_exponential(multiplier=chat_settings.retry_wait_multiplier,
                                 min=chat_settings.retry_wait_min,
                                 max=chat_settings.retry_wait_max),
           after=after_log(logger, logging.DEBUG))
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

    @lru_cache(maxsize=chat_settings.max_cache_size)
    def make_prompt_request(self, prompt: Prompt) -> Tuple[str, Decimal]:
        final_prompt, _ = truncate_prompt(prompt, self.encoding, self.max_tokens_question)
        return self._call_api(final_prompt)
