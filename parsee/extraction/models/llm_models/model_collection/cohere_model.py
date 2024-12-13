from functools import lru_cache
from typing import List, Tuple
from decimal import Decimal

import cohere
import tiktoken
from cohere import TooManyRequestsError
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential, after_log

from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel, truncate_prompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.settings import chat_settings
import logging

logger = logging.getLogger(__name__)


class CohereModel(LLMBaseModel):

    def __init__(self, model: MlModelSpecification):
        super().__init__(model)
        self.max_retries = 5
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_answer = 1024 if model.max_output_tokens is None else model.max_output_tokens
        self.max_tokens_question = self.spec.max_tokens - self.max_tokens_answer
        self.client = cohere.Client(model.api_key)

    @retry(reraise=True,
           stop=stop_after_attempt(chat_settings.retry_attempts),
           retry=retry_if_exception_type(TooManyRequestsError),
           wait=wait_exponential(multiplier=chat_settings.retry_wait_multiplier,
                                 min=chat_settings.retry_wait_min,
                                 max=chat_settings.retry_wait_max),
           after=after_log(logger, logging.DEBUG))
    def _call_api(self, prompt: str, images: List[Base64Image]) -> Tuple[str, Decimal]:
        response = self.client.chat(
            model=self.spec.internal_name,
            preamble=self.spec.system_message,
            message=prompt,
            temperature=0,
            chat_history=[],
            prompt_truncation='OFF',
            connectors=[]
        )

        answer = response.text
        cost_input = (int(response.meta.billed_units.input_tokens) * Decimal(self.spec.price_per_1k_tokens / 1000)) if self.spec.price_per_1k_tokens is not None else Decimal(0)
        cost_output = (int(response.meta.billed_units.output_tokens) * Decimal(self.spec.price_per_1k_output_tokens / 1000)) if self.spec.price_per_1k_output_tokens is not None else Decimal(0)
        cost_images = (len(images) * Decimal(self.spec.price_per_image)) if self.spec.price_per_image is not None else Decimal(0)
        final_cost = cost_input + cost_output + cost_images
        return answer, final_cost

    @lru_cache(maxsize=chat_settings.max_cache_size)
    def make_prompt_request(self, prompt: Prompt) -> Tuple[str, Decimal]:
        final_prompt, _ = truncate_prompt(prompt, self.encoding, self.max_tokens_question)
        return self._call_api(final_prompt, prompt.available_data if self.spec.multimodal else [])
