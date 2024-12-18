import os

from parsee.templates.helpers import StructuringItem, MetaItem, create_template
from parsee.extraction.models.helpers import gpt_config, ollama_config
from parsee.converters.main import load_document, from_text
from parsee.extraction.run import run_job_with_single_model
from parsee.utils.enums import *

# BASIC EXAMPLE with multiple structuring items and meta items

# GOALS:
# 1) we want to extract the invoice total, but not just the number, also the currency attached to it.
# 2) we want to extract the issuer of the invoice


# Step 1: create an extraction template
question_to_be_answered = "What is the invoice total?"
output_type = OutputType.NUMERIC

meta_currency_question = "What is the currency?"
meta_currency_output_type = OutputType.LIST # we want the model to use a pre-defined item from a list, this is basically a classification
meta_currency_list_values = ["USD", "EUR", "Other"] # any list of strings can be used here

meta_item = MetaItem(meta_currency_question, meta_currency_output_type, list_values=meta_currency_list_values)

invoice_total = StructuringItem(question_to_be_answered, output_type, meta_info=[meta_item])

# let's also define an item for the issuer of the invoice
invoice_issuer = StructuringItem("Who is the issuer of the invoice?", OutputType.ENTITY)

job_template = create_template([invoice_total, invoice_issuer])

# Step 2: define a model
model = ollama_config("llama3")

# Step 3: load a document
# We can use the example string included in the readme for example
input_string = "The invoice total amounts to 12,5 Euros and is due on Feb 28th 2024. Invoice to: Some company LLC. Thanks for using the services of CloudCompany Inc."
document = from_text(input_string)

_, _, answers_open_source_model = run_job_with_single_model(document, job_template, model)

# Step 3: load a document
# Or we can load an actual PDF invoice
file_path = "../tests/fixtures/Midjourney_Invoice-DBD682ED-0005.pdf" # modify file path here
document = load_document(file_path)

# Step 4: run the extraction
_, _, answers_open_source_model = run_job_with_single_model(document, job_template, model)

# let's see if some other model can also predict the right answer
open_ai_api_key = os.getenv("OPENAI_KEY") # enter your key manually here instead of loading from an .env file
gpt_model = gpt_config(open_ai_api_key)

_, _, answers_gpt = run_job_with_single_model(document, job_template, gpt_model)

