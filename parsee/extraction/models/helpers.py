from typing import *
import os

from parsee.extraction.models.model_dataclasses import *
from parsee.utils.constants import NUM_TOKENS_DEFAULT_LLM


def gpt_config(openai_api_key: str, token_limit: Optional[int] = None, openai_model_name: Optional[str] = None, multimodal: bool = False, max_images: int = 5, max_image_size: int = 2000, output_token_limit: Optional[int] = None, system_message: Optional[str] = None) -> MlModelSpecification:
    model_name = openai_model_name if openai_model_name is not None else "gpt-4-0314"
    token_limit = NUM_TOKENS_DEFAULT_LLM if token_limit is None else token_limit
    return MlModelSpecification(f"OpenAI model: {model_name}", model_name, model_name, ModelType.GPT, None, None, None, None, token_limit, openai_api_key, None, None, None, None, None, multimodal, max_images, max_image_size, output_token_limit, system_message)


def replicate_config(replicate_api_key: str, model_name: str, token_limit: Optional[int] = None, output_token_limit: Optional[int] = None, system_message: Optional[str] = None) -> MlModelSpecification:
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    token_limit = NUM_TOKENS_DEFAULT_LLM if token_limit is None else token_limit
    return MlModelSpecification(f"Replicate model: {model_name}", model_name, model_name, ModelType.REPLICATE, None, None, None, None, token_limit, None, None, None, None, None, None, False, None, None, output_token_limit, system_message)


def anthropic_config(anthropic_api_key: str, model_name: str, token_limit: Optional[int] = None, multimodal: bool = False, max_images: int = 5, max_image_size: int = 2000, output_token_limit: Optional[int] = None, system_message: Optional[str] = None) -> MlModelSpecification:
    token_limit = NUM_TOKENS_DEFAULT_LLM if token_limit is None else token_limit
    return MlModelSpecification(f"Anthropic model: {model_name}", model_name, model_name, ModelType.ANTHROPIC, None, None, None, None, token_limit, anthropic_api_key, None, None, None, None, None, multimodal, max_images, max_image_size, output_token_limit, system_message)


def ollama_config(model_name: str, custom_host: Optional[str] = None, token_limit: Optional[int] = None, multimodal: bool = False, max_images: int = 5, max_image_size: int = 2000, output_token_limit: Optional[int] = None, system_message: Optional[str] = None) -> MlModelSpecification:
    token_limit = NUM_TOKENS_DEFAULT_LLM if token_limit is None else token_limit
    return MlModelSpecification(f"Ollama model: {model_name}", model_name, model_name, ModelType.OLLAMA, custom_host, Decimal(0), None, None, token_limit, None, None, None, None, None, None, multimodal, max_images, max_image_size, output_token_limit, system_message)


def together_config(together_api_key: str, model_name: str, token_limit: Optional[int] = None, output_token_limit: Optional[int] = None, system_message: Optional[str] = None) -> MlModelSpecification:
    token_limit = NUM_TOKENS_DEFAULT_LLM if token_limit is None else token_limit
    return MlModelSpecification(f"Together AI model: {model_name}", model_name, model_name, ModelType.TOGETHER, None, None, None, None, token_limit, together_api_key, None, None, None, None, None, False, None, None, output_token_limit, system_message)


def cohere_config(cohere_api_key: str, model_name: str, token_limit: Optional[int] = None, output_token_limit: Optional[int] = None, system_message: Optional[str] = None) -> MlModelSpecification:
    token_limit = NUM_TOKENS_DEFAULT_LLM if token_limit is None else token_limit
    return MlModelSpecification(f"Cohere model: {model_name}", model_name, model_name, ModelType.COHERE, None, None, None, None, token_limit, cohere_api_key, None, None, None, None, None, False, None, None, output_token_limit, system_message)


def mistral_api_config(mistral_api_key: str, model_name: str, token_limit: Optional[int] = None, output_token_limit: Optional[int] = None, system_message: Optional[str] = None, multimodal: bool = False, max_images: int = 5, max_image_size: int = 2000) -> MlModelSpecification:
    token_limit = NUM_TOKENS_DEFAULT_LLM if token_limit is None else token_limit
    return MlModelSpecification(f"Mistral hosted model: {model_name}", model_name, model_name, ModelType.MISTRAL, None, None, None, None, token_limit, mistral_api_key, None, None, None, None, None, multimodal, max_images, max_image_size, output_token_limit, system_message)