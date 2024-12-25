import csv
from typing import *

from parsee.extraction.extractor_elements import StructuredTable, ExtractedSource, StructuredRow, StructuredTableCell, DocumentType


def csv_delimiter_simple(csv_content: str) -> Tuple[str, float, int]:

    num_comma = csv_content.count(",")
    num_semicolon = csv_content.count(";")
    divider = 1 if num_semicolon + num_comma == 0 else num_semicolon + num_comma
    return "," if num_comma > num_semicolon else ";", (num_comma if num_comma > num_semicolon else num_semicolon) / divider, num_semicolon + num_comma


def parse_csv(file_path: str, max_rows: Optional[int] = None) -> StructuredTable:

    table = StructuredTable(ExtractedSource(DocumentType.TABULAR, None, None, 0, None), [])
    sniffer = csv.Sniffer()
    all_rows = []
    max_cols = 0
    try:
        f = open(file_path, "r", encoding='utf8')
        data = f.read()
    except UnicodeDecodeError:
        f = open(file_path, "r", encoding='windows-1254')
        data = f.read()
    f.seek(0)
    dialect = sniffer.sniff(data, delimiters=[",", ";", "|"])
    delimiter_alt, share_alt, total_alt = csv_delimiter_simple(data)
    if total_alt > 100 and share_alt > 0.6:
        dialect.delimiter = delimiter_alt
    reader = csv.reader(f, dialect)
    for k, row in enumerate(reader):
        if max_rows is not None and k+1 > max_rows:
            continue
        if len(row) > max_cols:
            max_cols = len(row)
        all_rows.append(row)
    for row in all_rows:
        r = StructuredRow("body", [])
        table.rows.append(r)
        for col_idx in range(0, max_cols):
            val = row[col_idx] if col_idx <= len(row) - 1 else ""
            r.values.append(StructuredTableCell(str(val), 1, True))
    f.close()

    return table
