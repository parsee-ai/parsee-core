import os

from parsee.templates.helpers import TableItem, MetaItem, create_template, StructuringItem
from parsee.extraction.models.helpers import gpt_config
from parsee.converters.main import load_document
from parsee.extraction.run import run_job_with_single_model
from parsee.extraction.final_structuring import final_tables_from_columns
from parsee.utils.enums import *
import pandas as pd

# TABLE EXTRACTION

# in this example we want to fully structure the data from a table
# this process is split in different parts, as this leads to the best performance in our experience
# The steps are the following: 1) Detect the relevant table(s) 2) Structure meta info for each column and 3) map the rows to standardized 'buckets' if needed

meta_currency_question = "What is the currency?"
meta_currency_output_type = OutputType.LIST # we want the model to use a pre-defined item from a list, this is basically a classification
meta_currency_list_values = ["USD", "EUR", "Other"] # any list of strings can be used here
meta_currency = MetaItem(meta_currency_question, meta_currency_output_type, list_values=meta_currency_list_values, assigned_id="currency")

meta_date_question = "What is the date the period is ending in?"
meta_date_output_type = OutputType.DATE
meta_date = MetaItem(meta_date_question, meta_date_output_type, assigned_id="period")

# we can set some keywords to help with finding the most relevant items inside the document, this is optional
table_item_keywords = "Revenues, Cost of goods sold, operating income, net profit, income statement, Financial statements"

# define the table item
table_item = TableItem("Profit & Loss Statement", table_item_keywords, [meta_currency, meta_date])

job_template = create_template(None, [table_item])

# define a model
# for this task it is best to take a rather powerful model
openai_key = os.getenv("OPENAI_KEY") # replace with your API key or use .env file
gpt_4o = gpt_config(openai_key, 100000, "gpt-4o")

# load a document
file_path = "../tests/fixtures/bayer1.pdf"
document = load_document(file_path)

# Step 4: run the extraction
_, column_output, _ = run_job_with_single_model(document, job_template, gpt_4o)

tables = final_tables_from_columns(column_output)

for table in tables:
    # save as csv
    df = table.to_pandas()
    df.to_csv(f"{table.detected_class}.csv")
