from decimal import Decimal
from typing import *

from tiktoken.core import Encoding

from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.models.model_dataclasses import MlModelSpecification


def get_tokens_encoded(prompt: str, encoding: Encoding) -> List[int]:
    return encoding.encode(prompt)


def truncate_prompt(prompt: Prompt, encoding: Encoding, max_tokens: int) -> Tuple[str, int]:
    tokens_history = get_tokens_encoded(prompt.history, encoding)
    tokens_instructions = get_tokens_encoded(prompt.instructions(), encoding)
    tokens_data = get_tokens_encoded(prompt.available_data_string(), encoding)
    if len(tokens_instructions) > max_tokens:
        raise Exception("instructions exceed token limit")
    num_tokens = len(tokens_history) + len(tokens_instructions) + len(tokens_data)
    if num_tokens > max_tokens:
        # check if instructions + data are fitting
        if len(tokens_instructions) + len(tokens_data) <= max_tokens:
            # cut the history only
            history_truncated = encoding.decode(tokens_history[0:(max_tokens-len(tokens_instructions)-len(tokens_data))])
            return f"{history_truncated} {prompt.instructions()} {prompt.available_data_string()}", max_tokens
        else:
            # cut the data and don't return history
            available_data_truncated = encoding.decode(tokens_data[0:(max_tokens-len(tokens_instructions))])
            return f"{prompt.instructions()} \n {available_data_truncated}", max_tokens
    else:
        return str(prompt), num_tokens


class LLMBaseModel:

    spec: MlModelSpecification

    def __init__(self, spec: MlModelSpecification):
        self.spec = spec

    def make_prompt_request(self, prompt: Prompt) -> Tuple[str, Decimal]:
        raise NotImplementedError
