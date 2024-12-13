from typing import Optional

import tiktoken
from tiktoken import Encoding
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    max_el_in_memory: int = 10000
    max_images_to_load_per_doc: int = 30
    min_tokens_for_instructions_and_history: int = 500
    encoding: Encoding = tiktoken.get_encoding("cl100k_base")
    max_cache_size: int = 128
    retry_attempts: int = 5
    retry_wait_multiplier: int = 1
    retry_wait_min: int = 2
    retry_wait_max: int = 20
    openai_key: Optional[str] = None
    replicate_key: Optional[str] = None
    together_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None


chat_settings = ChatSettings()

