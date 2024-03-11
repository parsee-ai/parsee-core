import re
from typing import *

from parsee.utils.helper import clean_numeric_value, words_contained
from parsee.extraction.extractor_dataclasses import ParseeMeta, ExtractedSource


MAIN_QUESTION_STR = "(main question)"


def get_cleaned_list(values: List[str]):
    values_cleaned = []
    for x in values:
        if '-' in x:
            for y in x.split('-'):
                values_cleaned.append(clean_numeric_value(y))
        else:
            values_cleaned.append(clean_numeric_value(x))
    return values_cleaned


def parse_sources(prompt_answer: str, total_elements: Optional[int]) -> Tuple[str, List[int]]:
    result = re.search(r'((Source(s|):\s*|)(\[([\d-]+)\](,\s*|))+)', prompt_answer)
    sources = []
    final_answer = prompt_answer

    if result is not None:
        values_cleaned = get_cleaned_list(result.group().split("]"))
        sources = [int(x) for x in values_cleaned if x is not None and (total_elements is None or int(x) < total_elements)]
        final_answer = final_answer.replace(result.group(), "")
    else:
        # try with outer brackets only
        result = re.search(r'((Source(s|):\s*|)\[(\d(, |,|-|))+(\]|))', prompt_answer)
        if result is not None:
            values_cleaned = get_cleaned_list(result.group().split(","))
            sources = [int(x) for x in values_cleaned if x is not None and (total_elements is None or int(x) < total_elements)]
            final_answer = final_answer.replace(result.group(), "")
        else:
            # check if just 'Sources(s):' is present and remove if so
            result = re.search(r'(Source(s|):\s*)', prompt_answer)
            if result is not None:
                final_answer = final_answer.replace(result.group(), "")

    return final_answer, sources


def parse_answer_blocks(prompt_answer: str) -> List[str]:
    blocks = prompt_answer.split("\n\n")
    return [x for x in blocks if x.strip() != ""]


def parse_main_and_meta(prompt_answer_block: str) -> Dict[Union[None, str], str]:

    # mistral uses these characters sometimes
    prompt_answer_block = prompt_answer_block.replace("）", ") ")
    prompt_answer_block = prompt_answer_block.replace("（", " (")

    # go line by line
    by_lines = prompt_answer_block.split("\n")
    output = {}
    for line in by_lines:
        match = re.search(r'\s*(\(.+\)\s*:|:|)([^\n]+)', line)
        if match is not None:
            meta_id = match.group(1) if len(words_contained(match.group(1))) > 0 else None
            value = match.group(2)
            if meta_id not in output:
                output[meta_id] = value
    return output