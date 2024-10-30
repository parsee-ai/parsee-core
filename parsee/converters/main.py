import json
import tempfile
from decimal import Decimal
from typing import Tuple, Union, List, Dict

from parsee.utils.enums import DocumentType
from parsee.extraction.extractor_elements import StandardDocumentFormat
from parsee.converters.json_to_raw import load_document_from_json
from parsee.converters.interfaces import RawToJsonConverter
from parsee.converters.html_extraction import HtmlConverter
from parsee.converters.pdf_extraction import PdfConverter
from parsee.converters.simple_text import SimpleTextConverter
from parsee.utils.helper import get_source_identifier, get_source_identifier_simple


def determine_document_type(file_path: str) -> DocumentType:
    file_path = file_path.lower()
    if file_path.endswith(".pdf") or file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
        return DocumentType.PDF
    elif file_path.endswith(".html") or file_path.endswith(".xml"):
        return DocumentType.HTML
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls") or file_path.endswith(".csv"):
        return DocumentType.TABULAR
    else:
        return DocumentType.OTHER


def choose_converter(source_type: DocumentType) -> RawToJsonConverter:
    if source_type == DocumentType.HTML:
        return HtmlConverter()
    elif source_type == DocumentType.PDF:
        return PdfConverter(None)
    else:
        raise Exception("Unknown format")


def load_document(file_path: str) -> StandardDocumentFormat:
    doc_type = determine_document_type(file_path)
    source_identifier = get_source_identifier(file_path)
    doc, _ = doc_to_standard_format(source_identifier, doc_type, choose_converter(doc_type), file_path)
    return doc


def from_text(text: str) -> StandardDocumentFormat:
    source_identifier = get_source_identifier_simple(text)
    converter = SimpleTextConverter(DocumentType.TEXT)
    doc, _ = doc_to_standard_format(source_identifier, DocumentType.TEXT, converter, text)
    return doc


def doc_to_standard_format(source_identifier: str, source_type: DocumentType, converter: RawToJsonConverter,
                           file_path_or_content: str) -> Tuple[StandardDocumentFormat, Decimal]:
    elements, amount = converter.convert(file_path_or_content)
    return StandardDocumentFormat(source_type, source_identifier, elements, None if source_type == DocumentType.TEXT else file_path_or_content), amount


def save_doc_in_standard_format(source_identifier: str, source_type: DocumentType, converter: RawToJsonConverter, file_path: str) -> \
        Tuple[any, StandardDocumentFormat, Decimal]:
    doc, amount = doc_to_standard_format(source_identifier, source_type, converter, file_path)
    # unset file path
    doc.file_path = None
    json_string = json.dumps(doc.to_json_dict())
    tmp = tempfile.NamedTemporaryFile(delete=True)
    tmp.write(str.encode(json_string, 'utf-8'))
    tmp.flush()
    return tmp, doc, amount


def load_standard_document_from_file(json_file_path: str) -> StandardDocumentFormat:
    with open(json_file_path, "r") as f:
        contents = json.loads(f.read())
        return load_document_from_json(contents)
