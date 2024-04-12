"""
You can also run custom prompts instead of using the extraction templates, if you prefer that.
In that case the data from the document you want to query will be appended to your prompt, either as text or as image(s), depending on the model type you chose
"""

from parsee.extraction.run import run_custom_prompt
from parsee.converters.main import load_document
from parsee.extraction.models.helpers import *

# Let's ask the model about the type of document using an image of the document:
custom_prompt = "What type of document is depicted here?"

# load the document the same way as before
document = load_document("../tests/fixtures/Midjourney_Invoice-DBD682ED-0005.pdf")

# we can specify how many images we want to pass to the model at most
max_images = 1
# we can also specify how large a single image should be at most. The number here is the max of width and height, not the total pixels
max_dimension = 2000
gpt_model = gpt_config(os.getenv("OPENAI_KEY"), None, openai_model_name="gpt-4-turbo", multimodal=True, max_images=max_images, max_image_size=max_dimension)

answer = run_custom_prompt(document, custom_prompt, gpt_model)

print(answer)