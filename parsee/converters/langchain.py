from typing import *

from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl, ExtractedSource
from parsee.utils.enums import DocumentType, ElementType
from langchain_core.document_loaders.base import BaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document

# requires to install langchain via extras:
# poetry install -E
# or pip install -e


def documents_to_extracted_elements(documents: List[Document], document_type: DocumentType) -> List[ExtractedEl]:

    # langchain only has text elements
    return [ExtractedEl(ElementType.TEXT, ExtractedSource(document_type, None, None, k, el.metadata), el.page_content) for k, el in enumerate(documents)]


def langchain_loader_to_sdf(loader: BaseLoader, document_type: DocumentType, source_identifier: str) -> StandardDocumentFormat:

    pages = loader.load_and_split()

    elements = documents_to_extracted_elements(pages, document_type)

    return StandardDocumentFormat(document_type, source_identifier, elements)