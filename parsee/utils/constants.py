
TRUTH_VALUES_MAPPING = {
    "pl": 1,
    "bs": 2,
    "cf": 3
}

TRUTH_VALUES_MAPPING_META = {
    "period": {
        "Q1": 0,
        "Q2": 1,
        "Q3": 2,
        "FY": 3,
        "H1": 4,
        "9M": 5
    },
    "unit": {
        1: 0,
        1000: 1,
        1000000: 2,
        1000000000: 3
    },
    "year_offset": {
        0: 0,
        1: 1,
        2: 2,
        -1: 3,
        -2: 4,
        -3: 5
    }
}

MERGE_MIN_CONFIDENCE = 0.6
PARTIAL_MIN_CONFIDENCE = 0.1
THRESHOLD_INBETWEEN_MERGE = 0.7
TEXT_DISTANCE_MERGE_THRESHOLD = 1000
TEXT_DISTANCE_CLOSE_STATEMENT_DETECTION = TEXT_DISTANCE_MERGE_THRESHOLD
MAX_DISTANCE_UNIQUE_PROXIMITY = 20000
