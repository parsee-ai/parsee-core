from typing import *
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

import tiktoken
from tiktoken.core import Encoding

from parsee.utils.enums import SearchStrategy, DocumentType
from parsee.extraction.extractor_elements import FileReference


@dataclass
class Message:
    text: str
    references: List[FileReference]
    author: Optional[str] = None
    cost: Optional[Decimal] = None

    def __str__(self):
        if self.author is None:
            return self.text
        return f"{self.author}: {self.text}"

    def __repr__(self):
        return str(self)


@dataclass
class ChatSettings:
    max_el_in_memory: int = 10000
    max_images_to_load_per_doc: int = 30
    min_tokens_for_instructions_and_history = 500
    encoding: Encoding = tiktoken.get_encoding("cl100k_base")
