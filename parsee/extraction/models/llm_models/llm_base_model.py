from decimal import Decimal
from typing import *

from tiktoken.core import Encoding

from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification


def get_tokens_encoded(prompt: str, encoding: Encoding) -> List[int]:
    return encoding.encode(prompt)


def truncate_prompt(prompt: str, encoding: Encoding, max_tokens: int) -> Tuple[str, int]:
    tokens = get_tokens_encoded(prompt, encoding)
    num_tokens = len(tokens)
    if num_tokens > max_tokens:
        prompt = encoding.decode(tokens[0:max_tokens])
        num_tokens = max_tokens
    return prompt, num_tokens


class LLMBaseModel:

    spec: MlModelSpecification

    def __init__(self, spec: MlModelSpecification):
        self.spec = spec

    def make_prompt_request(self, prompt: Prompt) -> Tuple[str, Decimal]:
        raise NotImplemented
