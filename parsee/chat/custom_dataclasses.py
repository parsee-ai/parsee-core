from typing import *
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

import tiktoken
from tiktoken.core import Encoding

from parsee.utils.enums import SearchStrategy, DocumentType
from parsee.extraction.extractor_elements import FileReference


class Role(Enum):
    USER = "USER"
    AGENT = "AGENT"


@dataclass
class Author:
    id: str
    role: Role
    type: Optional[str] = None

    def __str__(self):
        return f"[{self.role.value}] {self.id}"

    def __repr__(self):
        return str(self)


@dataclass
class Message:
    text: str
    references: List[FileReference]
    author: Author
    cost: Optional[Decimal] = None

    def __str__(self):
        return f"{self.author}: {self.text}"

    def __repr__(self):
        return str(self)


@dataclass
class ChatSettings:
    max_el_in_memory: int = 10000
    max_images_to_load_per_doc: int = 30
    min_tokens_for_instructions_and_history = 500
    encoding: Encoding = tiktoken.get_encoding("cl100k_base")
