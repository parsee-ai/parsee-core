import tiktoken
from tiktoken import Encoding
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatSettings(BaseSettings):
    max_el_in_memory: int = 10000
    max_images_to_load_per_doc: int = 30
    min_tokens_for_instructions_and_history: int = 500
    encoding: Encoding = tiktoken.get_encoding("cl100k_base")
    max_cache_size: int = 128
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

chat_settings = ChatSettings()
