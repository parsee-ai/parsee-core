from typing import *

from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl


class VectorStore:

    def find_closest_elements(self, document: StandardDocumentFormat, search_element_title: str, keywords: str, tables_only: bool = True) -> List[ExtractedEl]:
        raise NotImplemented