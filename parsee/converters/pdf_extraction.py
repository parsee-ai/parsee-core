import os
from decimal import Decimal
from typing import List, Tuple, Dict, Union

from parsee.extraction.extractor_elements import ExtractedEl, StructuredTable, ExtractedSource, StructuredRow, \
    StructuredTableCell
from pdf_reader import get_elements_from_pdf
from parsee.converters.interfaces import RawToJsonConverter
from parsee.utils.enums import DocumentType, ElementType
from pdf_reader.custom_dataclasses import RelativeAreaPrediction
from pdf_reader.custom_dataclasses import ExtractedPage


PRICING_PDF_CONVERSION = Decimal(os.getenv("PRICING_CONVERSION")) if os.getenv("PRICING_CONVERSION") is not None else Decimal(0)


class PdfConverter(RawToJsonConverter):

    def __init__(self, predicted_areas: Union[None, Dict[int, List[RelativeAreaPrediction]]]):
        super().__init__(DocumentType.PDF)
        self.service_name = "simfin_pdf"
        self.areas = predicted_areas

    def pages_to_extracted_el(self, pages: List[ExtractedPage]) -> List[ExtractedEl]:
        elements: List[ExtractedEl] = []
        for page_num, page in enumerate(pages):
            for idx, el in enumerate(page.paragraphs):
                dict_el = el.dict_el
                element = None
                if dict_el["type"] == "et":
                    element = self._make_structured_table(
                        ExtractedSource(DocumentType.PDF, {"x0": el.x0, "x1": el.x1, "y0": el.y0, "y1": el.y1}, None,
                                        len(elements), {"page_idx": page_num, "page_size": [page.size.x0, page.size.y0, page.size.x1, page.size.y1]}), dict_el)
                elif dict_el["type"] == "em":
                    element = ExtractedEl(ElementType.TEXT, ExtractedSource(DocumentType.PDF,
                                                                            {"x0": el.x0, "x1": el.x1, "y0": el.y0,
                                                                             "y1": el.y1}, None, len(elements),
                                                                            {"page_idx": page_num,
                                                                             "page_size": [page.size.x0, page.size.y0, page.size.x1, page.size.y1]}), el.get_text())
                elif dict_el["type"] == "ef":
                    element = ExtractedEl(ElementType.FIGURE, ExtractedSource(DocumentType.PDF, {"x0": el.x0, "x1": el.x1, "y0": el.y0, "y1": el.y1}, None, len(elements), {"page_idx": page_num, "page_size": [page.size.x0, page.size.y0, page.size.x1, page.size.y1]}), None)
                elements.append(element)
        return elements

    def convert(self, file_path: str) -> Tuple[List[ExtractedEl], Decimal]:

        pages = get_elements_from_pdf(file_path, self.areas)

        return self.pages_to_extracted_el(pages), PRICING_PDF_CONVERSION

    def _make_structured_table(self, source: ExtractedSource, extracted_table_dict) -> StructuredTable:

        rows_structured: List[StructuredRow] = []
        # add meta rows
        for row in extracted_table_dict['m']:
            rows_structured.append(self._make_structured_row("header", row))

        # add value rows
        for row in extracted_table_dict['i']:
            rows_structured.append(self._make_structured_row("body", row))

        return StructuredTable(source, rows_structured)

    def _make_structured_row(self, row_type: str, row) -> StructuredRow:
        row_values: List[StructuredTableCell] = []
        if type(row) is list:
            # meta row
            for val in row:
                val_obj = self._make_structured_cell_from_dict(val)
                if val_obj.valid:
                    row_values.append(val_obj)
        elif type(row) is dict:
            # line item
            li_caption_cell = StructuredTableCell(row['c'])
            row_values.append(li_caption_cell)
            for val in row['v']:
                val_obj = self._make_structured_cell_from_dict(val)
                if val_obj.valid:
                    row_values.append(val_obj)
        return StructuredRow(row_type, row_values)

    def _make_structured_cell_from_dict(self, val_dict) -> StructuredTableCell:

        colspan = 1
        # meta items
        if "type" in val_dict:
            if val_dict['type'] == "null":
                val = ""
            else:
                val = str(val_dict['t'])
        # value item
        else:
            val = str(val_dict['v']) if val_dict['v'] is not None else ""

        return StructuredTableCell(val, colspan)
