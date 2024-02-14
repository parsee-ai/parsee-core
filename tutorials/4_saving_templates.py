import os

from parsee.cloud.api import ParseeCloud
from parsee.templates.helpers import StructuringItem, MetaItem, create_template
from parsee.utils.enums import *
# Extraction templates define all the aspects of an extraction job in a JSON file
# You can create extraction templates easily in Python (see examples 0,1 and 2) or for free on parsee cloud: https://app.parsee.ai
# In the following we will show you how to save templates to parsee cloud and load them from the cloud locally
# To use parsee cloud, you need an API key. Get your API key by registering here: https://app.parsee.ai

# Saving templates
# Let's create a template here in Python the same way we did in examples 0 and 1:
question_to_be_answered = "What is the invoice total?"
output_type = OutputType.NUMERIC
meta_currency_question = "What is the currency?"
meta_currency_output_type = OutputType.LIST
meta_currency_list_values = ["USD", "EUR", "Other"]
meta_item = MetaItem(meta_currency_question, meta_currency_output_type, list_values=meta_currency_list_values)
invoice_total = StructuringItem(question_to_be_answered, output_type, meta_info=[meta_item])
invoice_issuer = StructuringItem("Who is the issuer of the invoice?", OutputType.ENTITY)

job_template = create_template([invoice_total, invoice_issuer])

# save the template to the cloud
api_key = os.getenv("BACKEND_API_KEY") # replace with your API-key or set as environment variable
cloud = ParseeCloud(api_key)
template_id = cloud.save_template(job_template) # for public templates, set the second parameter to True

# you can find your template now in parsee cloud
print(template_id)
