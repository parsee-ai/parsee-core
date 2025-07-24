from functools import lru_cache
from typing import List, Tuple
from decimal import Decimal

from google import genai
from google.genai import types, errors
import base64

import tiktoken
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_random_exponential, after_log

from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel, truncate_prompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.settings import chat_settings
import logging

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    pass


class GoogleModel(LLMBaseModel):

    def __init__(self, model: MlModelSpecification):
        super().__init__(model)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_answer = 5000 if model.max_output_tokens is None else model.max_output_tokens
        self.max_tokens_question = self.spec.max_tokens - self.max_tokens_answer
        if model.file_path is None or len(model.file_path.split("__")) != 2:
            raise Exception("for google models please provide the file_path argument in the format: PROJECT__LOCATION")
        project, location = tuple(model.file_path.split("__"))
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

    @retry(
        stop=stop_after_attempt(chat_settings.retry_attempts),
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_random_exponential(
            multiplier=chat_settings.retry_wait_multiplier,
            min=chat_settings.retry_wait_min,
            max=chat_settings.retry_wait_max
        ),
        after=after_log(logger, logging.DEBUG)
    )
    def _call_api(self, prompt: str, images: list) -> tuple[str, Decimal]:
        if prompt.strip() == "":
            if len(images) == 0:
                return "no prompt or images provided", Decimal(0)
            prompt = "These are the images:"
        parts = [types.Part(text=prompt)]
        for x in images or []:
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type=x.media_type,
                        data=base64.b64decode(x.data)
                    )
                )
            )
        # Prepend system message if provided
        contents = []
        if self.spec.system_message:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=self.spec.system_message)]
                )
            )
        contents.append(types.Content(role="user", parts=parts))

        config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=self.max_tokens_answer,
            top_p=1,
            safety_settings=[types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            )],
            # add more settings as needed
        )

        try:
            response = self.client.models.generate_content(
                model=self.spec.internal_name,
                contents=contents,
                config=config
            )
        except errors.ClientError as e:
            if getattr(e, "status_code", None) == 429:
                raise RateLimitError("Rate limit exceeded") from e
            raise
        except Exception as e:
            raise

        # This assumes response.candidates[0].content.parts[0].text is the answer
        answer = ""
        if response.candidates and response.candidates[0].content.parts:
            answer = response.candidates[0].content.parts[0].text

        # Usage/cost reporting depends on what is actually returned by the API.
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", 0)
            completion_tokens = getattr(usage, "candidates_token_count", 0)
            cost_input = (prompt_tokens * Decimal(self.spec.price_per_1k_tokens or 0) / 1000) if self.spec.price_per_1k_tokens else Decimal(0)
            cost_output = (completion_tokens * Decimal(self.spec.price_per_1k_output_tokens or 0) / 1000) if self.spec.price_per_1k_output_tokens else Decimal(0)
            cost_images = (len(images) * Decimal(self.spec.price_per_image)) if self.spec.price_per_image else Decimal(0)
            final_cost = cost_input + cost_output + cost_images
        else:
            final_cost = Decimal(0)
        return answer, final_cost

    @lru_cache(maxsize=chat_settings.max_cache_size)
    def make_prompt_request(self, prompt: Prompt) -> tuple[str, Decimal]:
        final_prompt, _ = truncate_prompt(prompt, self.encoding, self.max_tokens_question)
        return self._call_api(final_prompt, prompt.available_data if self.spec.multimodal else [])
