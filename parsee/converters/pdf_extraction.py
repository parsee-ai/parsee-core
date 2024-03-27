import os
from decimal import Decimal
from typing import List, Tuple, Dict, Union

from parsee.extraction.extractor_elements import ExtractedEl, StructuredTable, ExtractedSource, StructuredRow, \
    StructuredTableCell
from pdf_reader import get_elements_from_pdf
from parsee.converters.interfaces import RawToJsonConverter
from parsee.utils.enums import DocumentType, ElementType
from pdf_reader.custom_dataclasses import RelativeAreaPrediction
from pdf_reader.custom_dataclasses import ExtractedPage, ExtractedTable, ExtractedPdfElement, ExtractedFigure, LineItem
from parsee.utils.helper import is_year_cell, is_number_cell


PRICING_PDF_CONVERSION = Decimal(os.getenv("PRICING_CONVERSION")) if os.getenv("PRICING_CONVERSION") is not None else Decimal(0)


class PdfConverter(RawToJsonConverter):

    def __init__(self, predicted_areas: Union[None, Dict[int, List[RelativeAreaPrediction]]]):
        super().__init__(DocumentType.PDF)
        self.service_name = "parsee_pdf"
        self.areas = predicted_areas

    def pages_to_extracted_el(self, pages: List[ExtractedPage]) -> List[ExtractedEl]:
        elements: List[ExtractedEl] = []
        for page_num, page in enumerate(pages):
            for idx, el in enumerate(page.paragraphs):
                if isinstance(el, ExtractedTable):
                    element = self._make_structured_table(
                        ExtractedSource(DocumentType.PDF, {"x0": el.x0, "x1": el.x1, "y0": el.y0, "y1": el.y1}, None,
                                        len(elements), {"page_idx": page_num, "page_size": [page.size.x0, page.size.y0, page.size.x1, page.size.y1]}), el)
                elif isinstance(el, ExtractedFigure):
                    element = ExtractedEl(ElementType.FIGURE, ExtractedSource(DocumentType.PDF, {"x0": el.x0, "x1": el.x1, "y0": el.y0, "y1": el.y1}, None, len(elements),
                                                                              {"page_idx": page_num, "page_size": [page.size.x0, page.size.y0, page.size.x1, page.size.y1]}), None)
                elif isinstance(el, ExtractedPdfElement):
                    element = ExtractedEl(ElementType.TEXT, ExtractedSource(DocumentType.PDF,
                                                                            {"x0": el.x0, "x1": el.x1, "y0": el.y0,
                                                                             "y1": el.y1}, None, len(elements),
                                                                            {"page_idx": page_num,
                                                                             "page_size": [page.size.x0, page.size.y0, page.size.x1, page.size.y1]}), el.get_text())
                else:
                    raise Exception("element not recognized")
                elements.append(element)
        return elements

    def convert(self, file_path_or_content: str, **kwargs) -> Tuple[List[ExtractedEl], Decimal]:

        pages = get_elements_from_pdf(file_path_or_content, self.areas, **kwargs)

        return self.pages_to_extracted_el(pages), PRICING_PDF_CONVERSION

    def _make_structured_table(self, source: ExtractedSource, table: ExtractedTable) -> StructuredTable:

        rows_structured: List[StructuredRow] = []

        # determine header end
        header_end_idx = 0
        for k, line_item in enumerate(table.items):
            found_valid = False
            for v in line_item.values:
                if is_number_cell(v.val) and not is_year_cell(v.val):
                    found_valid = True
                    break
            if found_valid:
                header_end_idx = k
                break

        # add value rows
        for k, row in enumerate(table.items):
            rows_structured.append(self._make_structured_row("body" if k >= header_end_idx else "header", row))

        return StructuredTable(source, rows_structured)

    def _make_structured_row(self, row_type: str, row: LineItem) -> StructuredRow:
        row_values: List[StructuredTableCell] = []
        li_caption_cell = StructuredTableCell(row.caption)
        row_values.append(li_caption_cell)
        for v in row.values:
            val_obj = StructuredTableCell(v.val)
            if val_obj.valid:
                row_values.append(val_obj)
        return StructuredRow(row_type, row_values)
