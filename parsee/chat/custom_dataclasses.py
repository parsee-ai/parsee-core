from typing import *
from dataclasses import dataclass
from decimal import Decimal

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
