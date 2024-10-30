import os
"""
To evaluate a model or to compare several models with each other in terms of performance, we need a Parsee dataset.
You can create a simple dataset manually as shown in the previous tutorial.
You can also run extractions for one or several documents on Parsee Cloud (https://app.parsee.ai),
correct and see the output in a graphical user interface, and then create datasets from there.
"""
from parsee.datasets.evaluation.main import evaluate_llm_performance
from parsee.datasets.readers.disk_reader import SimpleCsvDiskReader
from parsee.templates.helpers import StructuringItem, MetaItem, create_template
from parsee.extraction.models.helpers import gpt_config, ollama_config
from parsee.utils.enums import *

# Let's first use the dataset we created in the previous example and run it for two different models
dataset_path = "/Users/thomasflassbeck/Desktop/temp/x/dataset_cf611191-2aa6-4c7f-8e53-777d29b92634/questions_invoice.csv"

# Let's use the same extraction template (would be better of course to save it to Parsee Cloud and load from there)
meta_currency = MetaItem("What is the currency?", OutputType.LIST, list_values=["USD", "EUR", "Other"])
invoice_total = StructuringItem("What is the invoice total?", OutputType.NUMERIC, meta_info=[meta_currency])
invoice_issuer = StructuringItem("Who is the issuer of the invoice?", OutputType.ENTITY)
job_template = create_template([invoice_total, invoice_issuer])

# let's create a dataset reader
reader = SimpleCsvDiskReader(dataset_path)

# let's define the models we want to evaluate
open_ai_api_key = os.getenv("OPENAI_KEY") # enter your key manually here instead of loading from an .env file
gpt_model = gpt_config(open_ai_api_key)
ollama_model = ollama_config("llama3")

# let's run predictions with several models
performance = evaluate_llm_performance(job_template, reader, [ollama_model, gpt_model])

print(performance)