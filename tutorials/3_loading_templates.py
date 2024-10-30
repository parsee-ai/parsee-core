import os

from parsee.cloud.api import ParseeCloud
from parsee.extraction.models.helpers import ollama_config
from parsee.converters.main import load_document
from parsee.extraction.run import run_job_with_single_model
# Extraction templates define all the aspects of an extraction job in a JSON file
# You can create extraction templates easily in Python (see examples 0,1 and 2) or for free on parsee cloud: https://app.parsee.ai
# In the following we will show you how to save templates to parsee cloud and load them from the cloud locally
# To use parsee cloud, you need an API key. Get your API key by registering here: https://app.parsee.ai

# Loading templates
# You can load any template created by you or a member of your organisation locally by specifying the template ID
# You can also load any public templates the same way.
# For the following, we will use the public template for invoice extraction: https://app.parsee.ai/template/654b562e9edc3c29cdfc8bb5
# The ID of the template is the text after the last slash in the URL.

parsee_cloud_template_id = "65f959afe34036446ee859ff"

# set credentials, you can find your API-key here after registering: https://app.parsee.ai/api

api_key = os.getenv("BACKEND_API_KEY") # replace with your API-key or set as environment variable
cloud = ParseeCloud(api_key)

# load template
template = cloud.get_template(parsee_cloud_template_id)

print(template.title)

# run an extraction job with a sample document
# Step 3: load a document
file_path = "../tests/fixtures/Midjourney_Invoice-DBD682ED-0005.pdf"
document = load_document(file_path)

# let's see if some other model can also predict the right answer
model = ollama_config("llama3")

_, _, answers_open_source_model = run_job_with_single_model(document, template, model)

print(answers_open_source_model)