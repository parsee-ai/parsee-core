from typing import *
import re
from decimal import Decimal
import time
import hashlib

import numpy as np


years_abs_strings = []
current_year = int(time.strftime("%Y"))
for yyyy in range(1980, current_year + 3):
    years_abs_strings.append(str(yyyy))


def clean_numeric_value_llm(string_val) -> Union[float, None]:
    matches = re.findall(r'(\d+(\d|,|\.|)*\d|\d)', string_val)
    last_numeric = None
    for match, _ in matches:
        if match.strip() != "":
            cleaned = clean_numeric_value(match)
            if cleaned is not None:
                last_numeric = float(cleaned)
    return last_numeric


def get_entity_value(string_val) -> Union[str, None]:
    match = re.search(r'(.+)( is:? | are:? |: )(.+)\.?', string_val)
    if match:
        return match.group(3)
    else:
        return None


def get_date_regex(string_val) -> Union[str, None]:
    match = re.search(r'(\d\d\d\d\-\d\d\-\d\d)', string_val)
    if match:
        return match.group(1)
    else:
        return None


def is_negative(cell_str):
    # minus
    if re.search(r'(-|—|–|‒|―|–|−)( | |)*\d', cell_str.strip()):
        return True
    # brackets
    if re.search(r'\([\d ,.%]+(\)|\b)', cell_str.strip()):
        return True
    return False


def comma_separator_thousands(cell_str):
    if re.search(r'\b[0-9]{1,3}[,][0-9]{3}\b', cell_str):
        return True
    return False


def dot_separator_thousands(cell_str):
    if re.search(r'\b[0-9]{1,3}[.][0-9]{3}\b', cell_str):
        return True
    return False


def clean_numeric_value(cell_str: str):
    if cell_str.strip() == "":
        return None
    mult = 1
    if is_negative(cell_str):
        mult = -1

    cell_str = re.sub(r'[^0-9,.]', '', cell_str)

    # clean thousands separator
    if comma_separator_thousands(cell_str):
        cell_str = re.sub(r',', "", cell_str)
    elif dot_separator_thousands(cell_str):
        cell_str = re.sub(r'\.', "", cell_str)

    # now also replace the comma with a dot should it be used in any case
    cell_str = re.sub(r',', ".", cell_str)

    if cell_str.replace('.', '', 1).isdigit():
        return Decimal(cell_str) * mult
    else:
        return None


def is_number_cell(cell_str):
    cell_str = str(cell_str)
    if cell_str is None:
        return False
    cell_str = re.sub(r'(\([^0-9 ]*\))|[^0-9A-Za-z/]', '', cell_str)
    if cell_str.isdigit():
        return True
    else:
        return False


def words_contained(cell_str, lower=False) -> List[str]:
    if lower:
        cell_str = cell_str.lower()
    return list(filter(lambda x: x != "", re.sub('[^A-Za-z0-9%$€£¥]', ' ', cell_str).split(" ")))


# cleans text for word embeddings
def clean_text_for_word_vectors2(text, base_year=None, remove_special_chars=False, remove_all_numbers=False, number_token=" xnumberx "):

    if text is None:
        return ""

    text = text.lower()

    # replace new lines with space
    text = " ".join(text.splitlines())

    # replace numbers with commas or dots in them with a token
    text = re.sub(r'(\b([0-9]{1,3}[,.])+[0-9]+\b|\b[0-9]+[,.][0-9]+\b)', number_token, text)

    if remove_special_chars:
        # remove all special characters apart from a few selected ones
        text = re.sub(r'[^\w\d\s,.;:#\'\"+*!$€%&/\[\]()-_=<>§`´]', " ", text)

    # insert a space around each special character
    text = re.sub(r'([^0-9A-Za-z ])', ' \\1 ', text)

    if base_year is not None:
        base_year = int(base_year)
        base_year_offsets = [-3, -2, -1, 0, 1, 2, 3]
        # replace years
        for offset in base_year_offsets:
            text = text.replace(str(base_year + offset),
                                " xbaseyear" + ("m" if offset <= 0 else "p") + str(abs(offset)) + "x ")

    if remove_all_numbers:
        text = re.sub(r'(\b[0-9]+\b)', number_token, text)

    # replace double space with single
    text = re.sub(r' +', ' ', text)

    return text.strip()


def composition_percentages(cell_str):
    output = {"numbers": 0, "text": 0, "special": 0}

    # clean
    cell_str_cleaned = re.sub('[^a-zA-Z0-9]', '', cell_str)

    chars = len(cell_str_cleaned)

    if chars == 0:
        return output

    numbers = re.sub(r'[^0-9]', "", cell_str_cleaned)

    output['numbers'] = len(numbers) / chars

    text = re.sub(r'[^a-zA-Z]', "", cell_str_cleaned)

    output['text'] = len(text) / chars

    # for special chars, look at all characters
    special = re.sub(r'[a-zA-Z0-9 ]', "", cell_str)

    output['special'] = len(special) / len(cell_str)

    return output


def is_year_cell(cell_str):
    pieces = words_contained(cell_str)
    for p in pieces:
        if p in years_abs_strings:
            return True
    return False


def get_mean_for_column(column_values: List) -> float:
    values_filtered = [abs(x) for x in column_values if x is not None and is_number_cell(x)]
    if len(values_filtered) == 0:
        return 0
    return float(np.mean(values_filtered))


def clean_spaces(cell_str):
    return " ".join([x for x in cell_str.split(" ") if x != ""])


def delete_trailing_zeros(integer) -> int:
    string = str(integer).rstrip('0')
    return int(string) if string != '' else 0


def clean_number_for_matching(num) -> int:
    if num is None:
        return 0
    # clean comma, dots etc.
    num = re.sub(r'[^0-9]', '', str(num))

    # make abs
    num = abs(int(num))

    # delete trailing zeros
    num = delete_trailing_zeros(num)

    return num


def get_source_identifier(file_path: str) -> str:
    BUF_SIZE = 65536

    sha = hashlib.sha256()

    with open(file_path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha.update(data)

    return sha.hexdigest()