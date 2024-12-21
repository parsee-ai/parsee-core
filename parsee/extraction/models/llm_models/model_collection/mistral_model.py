from functools import lru_cache
from typing import List, Tuple
from decimal import Decimal

from mistralai import Mistral, SDKError
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception, after_log

from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel, truncate_prompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.settings import chat_settings
import logging

logger = logging.getLogger(__name__)


class MistralModel(LLMBaseModel):

    def __init__(self, model: MlModelSpecification):
        super().__init__(model)
        self.max_retries = 5
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_answer = 1024 if model.max_output_tokens is None else model.max_output_tokens
        self.max_tokens_question = self.spec.max_tokens - self.max_tokens_answer
        self.client = Mistral(api_key=model.api_key) if model.api_key is not None else None

    @retry(reraise=True,
           stop=stop_after_attempt(chat_settings.retry_attempts),
           retry=retry_if_exception(lambda x: isinstance(x, SDKError) and x.status_code == 429),
           wait=wait_random_exponential(multiplier=chat_settings.retry_wait_multiplier,
                                 min=chat_settings.retry_wait_min,
                                 max=chat_settings.retry_wait_max),
           after=after_log(logger, logging.DEBUG) )
    def _call_api(self, prompt: str, images: List[Base64Image]) -> Tuple[str, Decimal]:

        user_message_content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        if len(images) > 0:
            user_message_content += [
                {
                    "type": "image_url",
                    "image_url": f"data:{x.media_type};base64,{x.data}",
                }
                for x in images
            ]

        messages = [{"role": "user", "content": user_message_content}]
        if self.spec.system_message is not None:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": self.spec.system_message}]})

        chat_response = self.client.chat.complete(
            model=self.spec.internal_name,
            messages=messages,
            temperature=0
        )

        answer = chat_response.choices[0].message.content
        cost_input = (int(chat_response.usage.prompt_tokens) * Decimal(self.spec.price_per_1k_tokens / 1000)) if self.spec.price_per_1k_tokens is not None else Decimal(0)
        cost_output = (int(chat_response.usage.completion_tokens) * Decimal(self.spec.price_per_1k_output_tokens / 1000)) if self.spec.price_per_1k_output_tokens is not None else Decimal(0)
        cost_images = (len(images) * Decimal(self.spec.price_per_image)) if self.spec.price_per_image is not None else Decimal(0)
        final_cost = cost_input + cost_output + cost_images
        return answer, final_cost

    @lru_cache(maxsize=chat_settings.max_cache_size)
    def make_prompt_request(self, prompt: Prompt) -> Tuple[str, Decimal]:
        final_prompt, _ = truncate_prompt(prompt, self.encoding, self.max_tokens_question)
        return self._call_api(final_prompt, prompt.available_data if self.spec.multimodal else [])
