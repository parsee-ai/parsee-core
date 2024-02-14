import re
from decimal import Decimal
from typing import Tuple, List, Union
import os

from bs4 import BeautifulSoup

from parsee.extraction.extractor_elements import ExtractedEl, StructuredTable
from parsee.converters.interfaces import RawToJsonConverter
from parsee.extraction.extractor_elements import ExtractedSource, StructuredRow, StructuredTableCell
from parsee.utils.enums import DocumentType, ElementType


PRICING_HTML_CONVERSION = Decimal(os.getenv("PRICING_CONVERSION")) if os.getenv("PRICING_CONVERSION") is not None else Decimal(0)


def get_element_sibling_xpath(element, xpath=""):
    parent = element.parent
    if parent is None:
        return xpath
    siblings = parent.find_all(element.name, recursive=False)
    if len(siblings) == 1:
        xpath = '/' + element.name + xpath
    else:
        index = siblings.index(element)
        xpath = f'/{element.name}[{index}]' + xpath
    return get_element_sibling_xpath(parent, xpath)


class HtmlConverter(RawToJsonConverter):

    def __init__(self):
        super().__init__(DocumentType.HTML)
        self.service_name = "simfin_html"

    def convert(self, file_path: str) -> Tuple[List[ExtractedEl], Decimal]:
        with open(file_path) as html_file:
            html_content = html_file.read()
            return self._extract_elements_from_html_data(html_content), PRICING_HTML_CONVERSION

    def _extract_elements_from_html_data(self, html_data) -> List[ExtractedEl]:

        soup = BeautifulSoup(html_data, 'lxml')

        # remove xbrli tags
        xbrli_tags = soup.find_all(re.compile(r'(xbrli:.+)'))
        for tag in xbrli_tags:
            tag.decompose()

        # remove line break tags and replace with space
        br_tags = soup.find_all('br')
        for tag in br_tags:
            tag.replace_with(" ")

        return self._extract_from_soup(soup)

    # extracts elements from beautifulsoup object
    def _extract_from_soup(self, soup, parent_list_len=0):
        # currently if there is text before a table NOT INSIDE SOME ELEMENT, this text is lost (this is considered minor)
        el_list = []

        # basic check if element is displayed or not
        style_str = soup["style"].replace(" ", "").lower() if "style" in soup.attrs else ""
        if "display:none" in style_str:
            return []

        # check if element contains a table
        check = soup.find_all("table")
        if len(check) > 0:
            # go down element by element
            # direct children
            direct_children = soup.find_all(recursive=False)
            for child in direct_children:
                if child.name == "table":
                    element_index = parent_list_len + len(el_list)
                    el_list.append(self._make_structured_table(
                        ExtractedSource(DocumentType.HTML, None, get_element_sibling_xpath(child), element_index, None), child))
                else:
                    # child is not a table, call function again
                    el_list += self._extract_from_soup(child, len(el_list) + parent_list_len)
            # lastly, add also text that is not contained in a child element
            text_pieces_remaining = [x.strip() for x in soup.find_all(text=True, recursive=False) if x.strip() != ""]
            base_xpath = get_element_sibling_xpath(soup)
            for text_piece in text_pieces_remaining:
                element_index = parent_list_len + len(el_list)
                el_list.append(
                    ExtractedEl(ElementType.TEXT, ExtractedSource(DocumentType.HTML, None, base_xpath, element_index, None),
                                text_piece))
        else:
            # no table inside, just return all contained strings
            element_index = parent_list_len + len(el_list)
            full_string_value = (" ".join(soup.stripped_strings)).strip()
            if full_string_value != "":
                el_list.append(
                    ExtractedEl(ElementType.TEXT, ExtractedSource(DocumentType.HTML, None, get_element_sibling_xpath(soup), element_index, None),
                                full_string_value))

        return el_list

    def _make_structured_table(self, source: ExtractedSource, soup) -> StructuredTable:

        # check for header
        header = soup.find("thead", recursive=False)
        rows_structured: List[StructuredRow] = []

        if header is not None:
            rows = header.find_all("tr")
            for row in rows:
                rows_structured.append(self._make_structured_row("header", row))

        # check for body
        body = soup.find("tbody", recursive=False)

        if body is not None:
            rows = body.find_all("tr")
            for row in rows:
                rows_structured.append(self._make_structured_row("body", row))

        # rows not inside thead or tbody
        rows = soup.find_all("tr", recursive=False)
        for row in rows:
            rows_structured.append(self._make_structured_row("body", row))

        # connect cells and determine amount of cols etc.
        return StructuredTable(source, rows_structured)

    def _make_structured_row(self, row_type: str, soup) -> StructuredRow:

        values = soup.find_all(recursive=False)
        structured_values: List[StructuredTableCell] = []

        for value in values:
            val_obj = self._make_structured_cell(value)
            if val_obj.valid:
                structured_values.append(val_obj)
        return StructuredRow(row_type, structured_values)

    def _make_structured_cell(self, soup) -> StructuredTableCell:

        val = soup.get_text()
        colspan = 1

        if "colspan" in soup.attrs and str(soup['colspan']).isnumeric():
            colspan = int(soup['colspan'])

        cell = StructuredTableCell(val, colspan)

        # check if object is valid
        style_str = soup["style"].replace(" ", "").lower() if "style" in soup.attrs else ""

        if "display:none" in style_str:
            cell.valid = False

        return cell
