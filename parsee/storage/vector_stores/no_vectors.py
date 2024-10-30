from typing import *

from parsee.storage.vector_stores.interfaces import VectorStore
from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl
from parsee.utils.enums import ElementType


class NoVectors(VectorStore):

    def __init__(self):
        pass

    def find_closest_elements(self, document: StandardDocumentFormat, search_element_title: str, keywords: Optional[str], tables_only: bool = True) -> List[ExtractedEl]:
        return document.elements

    def sort_identifiers_by_relevance(self, source_identifiers: Set[str], search_query: str) -> Set[str]:
        return source_identifiers
