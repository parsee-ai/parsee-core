from typing import *
from decimal import Decimal

from parsee.extraction.extractor_elements import ExtractedEl, ElementType, ExtractedSource, DocumentType
from parsee.converters.interfaces import RawToJsonConverter


class SimpleTextConverter(RawToJsonConverter):

    def convert(self, file_path_or_content: str) -> Tuple[List[ExtractedEl], Decimal]:

        return [ExtractedEl(ElementType.TEXT, ExtractedSource(DocumentType.TEXT, None, None, 0, None), file_path_or_content)], Decimal(0)
