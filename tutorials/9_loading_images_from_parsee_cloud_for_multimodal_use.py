"""
With the simple multimodal approach from the last tutorial, new images are being created every time the program is run.
This is of course slow and inefficient. When you upload document to Parsee Cloud, images are created automatically for PDF files.
We can simply load the images from Parsee Cloud for these cases.
"""
import os
import time

from parsee.cloud.api import ParseeCloud
from parsee.templates.helpers import StructuringItem, MetaItem, create_template
from parsee.utils.enums import *
from parsee.extraction.models.helpers import *
from parsee.extraction.run import run_job_with_single_model
from parsee.cloud.cloud_image_fetcher import CloudImageFetcher

# Let's upload a file to Parsee Cloud, we will get the "source_identifier" as a response. This is used to uniquely identify the document we uploaded.
api_key = os.getenv("BACKEND_API_KEY") # replace with your API-key or set as environment variable
cloud = ParseeCloud(api_key, custom_host="http://localhost:8091")

# we use the invoice again
source_identifier = cloud.upload_file("../tests/fixtures/Midjourney_Invoice-DBD682ED-0005.pdf")

print(source_identifier)

# we have to wait shortly until the document is converted to retrieve the images (there is an API call also to check for the status of the conversion, please see the Parsee Cloud docs for that).
time.sleep(10)

# Let's define the template again
meta_currency = MetaItem("What is the currency?", OutputType.LIST, list_values=["USD", "EUR", "Other"])
invoice_total = StructuringItem("What is the invoice total?", OutputType.NUMERIC, meta_info=[meta_currency])
invoice_issuer = StructuringItem("Who is the issuer of the invoice?", OutputType.ENTITY)
job_template = create_template([invoice_total, invoice_issuer])

# We can retrieve the document from Parsee Cloud now
document = cloud.get_document(source_identifier)

# We set up the retrieval of images from the cloud
image_fetcher = CloudImageFetcher(cloud)

# we can specify how many images we want to pass to the model at most
max_images = 1
# we can also specify how large a single image should be at most. The number here is the max of width and height, not the total pixels
max_dimension = 2000
gpt_model = gpt_config(os.getenv("OPENAI_KEY"), None, openai_model_name="gpt-4-turbo", multimodal=True, max_images=max_images, max_image_size=max_dimension)

# run the extraction the same way as before
_, _, answers_gpt = run_job_with_single_model(document, job_template, gpt_model, custom_image_creator=image_fetcher)

print(answers_gpt)