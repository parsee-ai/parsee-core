import json
from typing import List

from parsee.extraction.extractor_elements import ExtractedEl, StructuredTable, StructuredTableCell, StructuredRow, StandardDocumentFormat
from parsee.utils.enums import ElementType, DocumentType
from parsee.extraction.extractor_dataclasses import ExtractedSource


def source_from_json(json_dict) -> ExtractedSource:
    return ExtractedSource(DocumentType(json_dict["source_type"]), json_dict["coordinates"], json_dict["xpath"], json_dict["element_index"], json_dict["other_info"])


def load_elements_from_json(elements_dict_list: List) -> List[ExtractedEl]:

    output: List[ExtractedEl] = []
    for el in elements_dict_list:

        if el["el_type"] == "text":
            output.append(ExtractedEl(ElementType.TEXT, source_from_json(el["source"]), el["text"]))
        elif el["el_type"] == "figure":
            output.append(ExtractedEl(ElementType.FIGURE, source_from_json(el["source"]), None))
        else:
            source = source_from_json(el["source"])
            rows_structured = []
            for row in el["rows"]:
                row_type = row['row_type']
                values_structured = []
                for val in row['values']:
                    values_structured.append(StructuredTableCell(val["val"], val["colspan"]))
                rows_structured.append(StructuredRow(row_type, values_structured))
            output.append(StructuredTable(source, rows_structured))
    return output


def load_document_from_json(json_contents) -> StandardDocumentFormat:

    source_type = DocumentType(json_contents["source_type"])
    source_identifier = json_contents["source_identifier"]
    elements = load_elements_from_json(json_contents["elements"])

    return StandardDocumentFormat(source_type, source_identifier, elements, None)
