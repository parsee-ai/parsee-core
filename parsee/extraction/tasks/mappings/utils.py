from typing import *
from decimal import Decimal

from parsee.extraction.extractor_dataclasses import ParseeBucket
from parsee.templates.mappings import MappingSchema


def calc_buckets(key_value_pairs: List[Tuple[str, any]], mappings: List[ParseeBucket], mapping_schema: MappingSchema) -> Dict[str, Decimal]:

    if len(mappings) != len(key_value_pairs):
        return {}

    output = {}

    # TODO: improve

    for k, mapping in enumerate(mappings):

        if mapping is None:
            continue

        if mapping.bucket_id not in output:
            output[mapping.bucket_id] = Decimal(0)

        val = key_value_pairs[k][1]
        if val is not None:
            output[mapping.bucket_id] += Decimal(val)

    return output


def get_table_signature(key_value_pairs: List[Tuple[str, any]]) -> str:

    return ", ".join([f"{k}: {x[0]}" for k, x in enumerate(key_value_pairs)])
