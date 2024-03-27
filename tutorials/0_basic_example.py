import os

from parsee import *

# BASIC EXAMPLE

# Step 1: create an extraction template
question_to_be_answered = "What is the invoice total?"
output_type = OutputType.NUMERIC

structuring_item = StructuringItem(question_to_be_answered, output_type)

job_template = create_template([structuring_item])

# Step 2: define a model
# requires an API key from replicate: https://replicate.com/
replicate_api_key = os.getenv("REPLICATE_KEY")
replicate_model = replicate_config(replicate_api_key, "mistralai/mixtral-8x7b-instruct-v0.1")

# Step 3: load a document
# Parsee converts all data (strings, file contents etc.) to a standardized format, the class for this is called StandardDocumentFormat
# a) let's first create a StandardDocumentFormat object from a simple string
input_string = "The invoice total amounts to 12,5 Euros and is due on Feb 28th 2024."
document = from_text(input_string)

# Step 4: run the extraction
_, _, answers = run_job_with_single_model(document, job_template, replicate_model)

print(answers[0].class_value)

# b) we can also simply load and convert files into the StandardDocumentFormat with the help of the converters that are included in Parsee, let's use an actual PDF invoice now
file_path = "../tests/fixtures/Midjourney_Invoice-DBD682ED-0005.pdf" # modify file path here in or
document = load_document(file_path)

# Step 4: run the extraction
_, _, answers = run_job_with_single_model(document, job_template, replicate_model)

print(answers[0].class_value)

# let's see if some other model can also predict the right answer
# requires an API key from openai
open_ai_api_key = os.getenv("OPENAI_KEY") # enter your key manually here instead of loading from an .env file
gpt_model = gpt_config(open_ai_api_key)

_, _, answers = run_job_with_single_model(document, job_template, gpt_model)

print(answers[0].class_value)
