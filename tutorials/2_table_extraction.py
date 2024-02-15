import os

from parsee.templates.helpers import TableItem, MetaItem, create_template
from parsee.extraction.models.helpers import gpt_config, replicate_config
from parsee.converters.main import load_document
from parsee.extraction.run import run_job_with_single_model
from parsee.utils.enums import *

# TABLE EXTRACTION

# in this example we want to fully structure the data from a table
# this process is split in different parts, as this leads to the best performance in our experience
# The steps are the following: 1) Detect the relevant table(s) 2) Structure meta info for each column and 3) map the rows to standardized 'buckets' if needed

meta_currency_question = "What is the currency?"
meta_currency_output_type = OutputType.LIST # we want the model to use a pre-defined item from a list, this is basically a classification
meta_currency_list_values = ["USD", "EUR", "Other"] # any list of strings can be used here
meta_currency = MetaItem(meta_currency_question, meta_currency_output_type, list_values=meta_currency_list_values)

meta_date_question = "What is the date the period is ending in?"
meta_date_output_type = OutputType.DATE
meta_date = MetaItem(meta_date_question, meta_date_output_type)

table_item = TableItem("Profit & Loss Statement", "Revenues, Cost of goods sold, operating income, net profit, Financial statements", [meta_currency, meta_date])

job_template = create_template(None, [table_item])

# define a model
# requires an API key from replicate: https://replicate.com/
replicate_api_key = os.getenv("REPLICATE_KEY")
replicate_model = replicate_config(replicate_api_key, "mistralai/mixtral-8x7b-instruct-v0.1")

# load a document
file_path = "../tests/fixtures/MSFT_FY_2007.html" # modify file path here (use absolute file paths if possible), for this example we are using one of the example files included in this repo
document = load_document(file_path)

# Step 4: run the extraction
_, column_output, _ = run_job_with_single_model(document, job_template, replicate_model)

print(column_output)