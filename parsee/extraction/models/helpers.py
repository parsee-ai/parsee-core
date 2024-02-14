from typing import *
import os

from parsee.extraction.models.model_dataclasses import *
from parsee.utils.constants import NUM_TOKENS_DEFAULT_LLM


def gpt_config(openai_api_key: str, token_limit: Optional[int] = None, openai_model_name: Optional[str] = None) -> MlModelSpecification:
    model_name = openai_model_name if openai_model_name is not None else "gpt-4-0314"
    token_limit = NUM_TOKENS_DEFAULT_LLM if token_limit is None else token_limit
    return MlModelSpecification("GPT 4", model_name, ModelType.GPT, None, None, token_limit, openai_api_key, None, None, None, None, None)


def replicate_config(replicate_api_key: str, model_name: str, token_limit: Optional[int] = None) -> MlModelSpecification:
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    token_limit = NUM_TOKENS_DEFAULT_LLM if token_limit is None else token_limit
    return MlModelSpecification(f"Replicate model: {model_name}", model_name, ModelType.REPLICATE, None, None, token_limit, None, None, None, None, None, None)