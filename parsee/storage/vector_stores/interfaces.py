from typing import *

from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl, FileReference


class VectorStore:

    def find_closest_elements(self, document: StandardDocumentFormat, search_element_title: str, keywords: str, tables_only: bool = True) -> List[ExtractedEl]:
        raise NotImplementedError

    def sort_identifiers_by_relevance(self, source_identifiers: Set[str], search_query: str) -> Set[str]:
        raise NotImplementedError