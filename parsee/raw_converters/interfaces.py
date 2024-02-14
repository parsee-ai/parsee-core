from typing import Tuple, List, Union

from src.utils.enums import DocumentType
from src.extraction.extractor_elements import ExtractedEl


class RawToJsonConverter:

    def __init__(self, doc_type: DocumentType):
        self.doc_type = doc_type

    def convert(self, file_path: str) -> List[ExtractedEl]:
        raise NotImplemented
