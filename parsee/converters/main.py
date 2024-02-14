import json
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from typing import Tuple, Union, List, Dict

from pdf_reader.custom_dataclasses import RelativeAreaPrediction
from parsee.utils.enums import DocumentType, ConversionMethod
from parsee.extraction.extractor_elements import StandardDocumentFormat
from parsee.raw_converters.json_to_raw import load_document_from_json
from parsee.raw_converters.html_extraction import HtmlConverter
from parsee.raw_converters.pdf_extraction import PdfConverter
from parsee.utils.settings import temp_path
from parsee.utils.helper import get_source_identifier


def determine_document_type(file_path: str) -> DocumentType:
    if file_path.endswith(".pdf"):
        return DocumentType.PDF
    elif file_path.endswith(".html") or file_path.endswith(".xml"):
        return DocumentType.HTML
    else:
        raise Exception("document format not recognized")


def load_document(file_path: str) -> StandardDocumentFormat:
    doc_type = determine_document_type(file_path)
    source_identifier = get_source_identifier(file_path)
    doc, _ = doc_to_standard_format(source_identifier, doc_type, ConversionMethod.SIMPLE, file_path, None)
    return doc


def doc_to_standard_format(source_identifier: str, source_type: DocumentType, method: ConversionMethod,
                           file_path: str, areas: Union[None, Dict[int, List[RelativeAreaPrediction]]]) -> Tuple[StandardDocumentFormat, Decimal]:
    if source_type == DocumentType.HTML:
        converter = HtmlConverter()
    elif source_type == DocumentType.PDF:
        if method == ConversionMethod.SIMPLE or method == ConversionMethod.COMPLEX:
            converter = PdfConverter(areas)
        elif method == ConversionMethod.AWS:
            converter = PdfAWSConverter(source_identifier, areas)
        else:
            raise Exception("Unknown conversion method")
    else:
        raise Exception("Unknown format")

    elements, amount = converter.convert(file_path)

    return StandardDocumentFormat(source_type, source_identifier, elements), amount


def save_doc_in_standard_format(source_identifier: str, source_type: DocumentType, method: ConversionMethod,
                                file_path: str, areas: Union[None, Dict[int, List[RelativeAreaPrediction]]]) -> \
        Tuple[any, StandardDocumentFormat, Decimal]:
    doc, amount = doc_to_standard_format(source_identifier, source_type, method, file_path, areas)
    json_string = json.dumps(doc.to_json_dict())
    tmp = tempfile.NamedTemporaryFile(delete=True, dir=temp_path)
    tmp.write(str.encode(json_string))
    return tmp, doc, amount


def load_standard_document_from_file(json_file_path: str) -> StandardDocumentFormat:
    with open(json_file_path, "r") as f:
        contents = json.loads(f.read())
        return load_document_from_json(contents)
