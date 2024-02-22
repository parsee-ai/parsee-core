from decimal import Decimal
from typing import *

from tiktoken.core import Encoding

from parsee.extraction.models.llm_models.prompts import Prompt


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

    model_name: str

    def __init__(self, name: str):
        self.model_name = name

    def make_prompt_request(self, prompt: str) -> Tuple[str, Decimal]:
        raise NotImplemented
