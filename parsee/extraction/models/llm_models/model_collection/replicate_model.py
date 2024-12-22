from functools import lru_cache
from typing import Tuple
from decimal import Decimal

import tiktoken
import replicate
from replicate.exceptions import ReplicateError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception, after_log

from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel, get_tokens_encoded, truncate_prompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.settings import chat_settings
import logging

logger = logging.getLogger(__name__)


class ReplicateModel(LLMBaseModel):

    def __init__(self, model: MlModelSpecification):
        super().__init__(model)
        self.max_retries = 5
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_answer = 1024 if model.max_output_tokens is None else model.max_output_tokens
        self.max_tokens_question = self.spec.max_tokens - self.max_tokens_answer

    @retry(reraise=True,
           stop=stop_after_attempt(chat_settings.retry_attempts),
           retry=retry_if_exception(lambda x: isinstance(x, ReplicateError) and len(x.args) > 0 and
                                              "Request was throttled." in x.args[0]),
           wait=wait_random_exponential(multiplier=chat_settings.retry_wait_multiplier,
                                 min=chat_settings.retry_wait_min,
                                 max=chat_settings.retry_wait_max),
           after=after_log(logger, logging.DEBUG) )
    def _call_api(self, prompt: str) -> str:

        response = replicate.run(self.spec.internal_name, input={
            "prompt": prompt,
            "system_prompt": self.spec.system_message if self.spec.system_message is not None else "",
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 0,
            "max_new_tokens": 1024,
            "presence_penalty": 0,
            "frequency_penalty": 0
        })
        answer = "".join(response)
        return answer

    @lru_cache(maxsize=chat_settings.max_cache_size)
    def make_prompt_request(self, prompt: Prompt) -> Tuple[str, Decimal]:
        final_prompt, num_tokens_input = truncate_prompt(prompt, self.encoding, self.max_tokens_question)
        response = self._call_api(final_prompt)
        tokens_response = len(get_tokens_encoded(response, self.encoding))
        cost_input = (int(num_tokens_input) * Decimal(self.spec.price_per_1k_tokens / 1000)) if self.spec.price_per_1k_tokens is not None else Decimal(0)
        cost_output = (int(tokens_response) * Decimal(self.spec.price_per_1k_output_tokens / 1000)) if self.spec.price_per_1k_output_tokens is not None else Decimal(0)
        final_cost = cost_input + cost_output
        return response, final_cost
