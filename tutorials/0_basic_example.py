import os

from src.extraction.templates.helpers import StructuringItem, MetaItem, create_template
from src.extraction.ml.models.helpers import gpt_config, replicate_config
from src.extraction.raw_converters.main import load_document
from src.extraction_jobs import run_job_with_single_model
from src.utils.enums import *

# BASIC EXAMPLE

# Step 1: create an extraction template
question_to_be_answered = "What is the invoice total?"
output_type = OutputType.NUMERIC

structuring_item = StructuringItem(question_to_be_answered, output_type)

job_template = create_template([structuring_item])

# Step 2: define a model
open_ai_api_key = os.getenv("OPENAI_KEY") # enter your key manually here instead of loading from an .env file
gpt_model = gpt_config(open_ai_api_key)

# Step 3: load a document
file_path = "./tests/fixtures/documents/pdf/Midjourney_Invoice-DBD682ED-0005.pdf" # modify file path here (use absolute file paths if possible), for this example we are using one of the example files included in this repo
document = load_document(file_path)

# Step 4: run the extraction
_, _, answers = run_job_with_single_model(document, job_template, gpt_model)

print(answers[0].class_value)

# let's see if some other model can also predict the right answer
# requires an API key from replicate: https://replicate.com/
replicate_api_key = os.getenv("REPLICATE_KEY")
replicate_model = replicate_config(replicate_api_key, "mistralai/mixtral-8x7b-instruct-v0.1")

_, _, answers = run_job_with_single_model(document, job_template, replicate_model)

print(answers[0].class_value)
