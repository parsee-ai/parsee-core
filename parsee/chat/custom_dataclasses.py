from typing import *
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal


from parsee.utils.enums import SearchStrategy, DocumentType
from parsee.extraction.extractor_elements import FileReference


class Role(Enum):
    USER = "USER"
    AGENT = "AGENT"


@dataclass
class Author:
    id: str
    role: Role
    type: Optional[str]

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
    search_strategy: SearchStrategy = SearchStrategy.VECTOR
    max_el_in_memory: int = 10000
    max_images: int = 20
