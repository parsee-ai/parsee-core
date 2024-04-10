import os
import time
from typing import List, Tuple
from dataclasses import dataclass
from decimal import Decimal
import math

import anthropic
import tiktoken

from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel, truncate_prompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.extractor_dataclasses import Base64Image


class AnthropicModel(LLMBaseModel):

    def __init__(self, model: MlModelSpecification):
        super().__init__(model)
        self.max_retries = 5
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_answer = 1024
        self.max_tokens_question = self.spec.max_tokens - self.max_tokens_answer
        self.client = anthropic.Anthropic(api_key=model.api_key if model.api_key is not None else os.getenv("ANTHROPIC_API_KEY"))

    def _call_api(self, prompt: str, images: List[Base64Image], retries: int = 0, wait: int = 5) -> Tuple[str, Decimal]:
        user_message_content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        if len(images) > 0:
            user_message_content += [
                {
                    "type": "text",
                    "text": "These are the images:"
                }
            ]
            user_message_content += [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": x.media_type,
                        "data": x.data,
                    },
                }
                for x in images
            ]
        try:
            message = self.client.messages.create(
                model=self.spec.internal_name,
                max_tokens=self.max_tokens_answer,
                temperature=0,
                messages=[
                    {"role": "user", "content": user_message_content}
                ]
            )

            answer = message.content[0].text if len(message.content) > 0 else ""
            final_cost = (Decimal(message.usage.input_tokens+message.usage.output_tokens) * Decimal(self.spec.price_per_1k_tokens/1000)) if self.spec.price_per_1k_tokens is not None else Decimal(0)
            return answer, final_cost
        except Exception as e:
            if retries < self.max_retries:
                time.sleep(wait * 2 ** retries)
                return self._call_api(prompt, images, retries + 1)
            else:
                return "", Decimal(0)

    def make_prompt_request(self, prompt: Prompt) -> Tuple[str, Decimal]:
        final_prompt, _ = truncate_prompt(str(prompt), self.encoding, self.max_tokens_question)
        return self._call_api(final_prompt, prompt.available_data if self.spec.multimodal else [])
