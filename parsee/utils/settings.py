import os
import re
import time
from decimal import Decimal

BUILD_MODE = os.getenv('BUILD_MODE')
main_path = os.getenv('MAIN_PATH')
temp_path = os.path.join(main_path, "temp")
temp_path_images = os.path.join(temp_path, "images")
SOURCES_BUCKET = os.getenv("SOURCES_BUCKET")
JSON_BUCKET = os.getenv("JSON_BUCKET")
AREA_API_HOST = os.getenv("AREA_API_HOST")

# path to ml models for element ml and meta detection
model_path_elements = os.path.join(main_path, "assets", "locations")
model_path_meta = os.path.join(main_path, "assets", "meta")

local_path = os.path.join(main_path, "src")
assets_path = os.path.join(main_path, "assets")

relevant_doc_types = [1, 2, 4, 5, 7]
doc_types_extraction = [1, 2]
BATCH_SIZE_IMG = os.getenv('BATCH_SIZE_IMG')

PRICING_CONVERSION = Decimal(0.05)
PRICING_EXTRACTION = Decimal(0.15)
PRICING_EXTRACTION_DB = Decimal(0.01)
PRICING_INSTANT = Decimal(0.01)
MAX_CREDITS_PER_JOB = Decimal(5)

ELEMENTS_WORDS_TO_INCLUDE = 50
ELEMENTS_TABLES_TO_INCLUDE = 3

NUM_TOKENS_DEFAULT_OPENAI = 4000

years_abs_strings = []
current_year = int(time.strftime("%Y"))
for yyyy in range(1980, current_year + 2):
    years_abs_strings.append(str(yyyy))

# regex for number detection
to_filter_numbers = re.compile(r'(\([^0-9 ]*\))|[^0-9A-Za-z/]')


currency_groups_simfin = {
    "eur": 1,
    "usd": 2,
    "gbp": 3,
    "cny": 4,
    "jpy": 5,
    "chf": 6,
    "rub": 7,
    "cad": 10,
    "sek": 12,
    "dkk": 15,
}
