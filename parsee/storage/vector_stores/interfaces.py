from typing import *

from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl


class VectorStore:

    def delete_all_from_document(self, source_identifier: str):
        raise NotImplemented

    def upload_document(self, document: StandardDocumentFormat):
        raise NotImplemented

    def find_closest_elements(self, document: StandardDocumentFormat, search_element_title: str, keywords: str, tables_only: bool = True) -> List[ExtractedEl]:
        raise NotImplemented