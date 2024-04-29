
"""
You can also pass images to a model instead of text fragments, if the model permits it.
In the following we will use the ChatGPT 4 vision API to answer the "invoice total" question, instead of feeding the text and tables to the model.
Note: currently we support multimodal extractions only for PDF and image files. For PDFs, Parsee will automatically create images of the most relevant pages.
If you want to use the multimodal capabilities for HTML files, you should use a tool like pdfkit: https://pypi.org/project/pdfkit/
Then use the PDF output instead of the HTML file.
"""
import os
from parsee.templates.helpers import StructuringItem, MetaItem, create_template
from parsee.extraction.models.helpers import *
from parsee.converters.main import load_document
from parsee.utils.enums import *
from parsee.extraction.run import run_job_with_single_model

# Let's use the same extraction template as before for determining the invoice total and the invoice issuer
meta_currency = MetaItem("What is the currency?", OutputType.LIST, list_values=["USD", "EUR", "Other"])
invoice_total = StructuringItem("What is the invoice total?", OutputType.NUMERIC, meta_info=[meta_currency])
invoice_issuer = StructuringItem("Who is the issuer of the invoice?", OutputType.ENTITY)
job_template = create_template([invoice_total, invoice_issuer])

# load the document the same way as before
document = load_document("../tests/fixtures/Midjourney_Invoice-DBD682ED-0005.pdf")

# all we have to change is to set 'multimodal=True' in the model configuration. Of course the model should also really support multimodal queries, so make sure you select an appropriate model version.
open_ai_api_key = os.getenv("OPENAI_KEY") # enter your key manually here instead of loading from an .env file
# we can specify how many images we want to pass to the model at most
max_images = 3
# we can also specify how large a single image should be at most. The number here is the max of width and height, not the total pixels
max_dimension = 2000
gpt_model = gpt_config(open_ai_api_key, None, openai_model_name="gpt-4-turbo", multimodal=True, max_images=max_images, max_image_size=max_dimension)

# run the extraction the same way as before
_, _, answers_gpt = run_job_with_single_model(document, job_template, gpt_model)

print(answers_gpt)

# you can also run multimodal queries locally with ollama
llava_model = ollama_config("llava:13b", multimodal=True)
_, _, answers_llava = run_job_with_single_model(document, job_template, llava_model)

print(answers_llava)