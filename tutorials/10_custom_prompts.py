"""
You can also run custom prompts instead of using the extraction templates, if you prefer that.
In that case the data from the document you want to query will be appended to your prompt, either as text or as image(s), depending on the model type you chose
"""

import cv2

from parsee.extraction.run import run_custom_prompt
from parsee.converters.main import load_document
from parsee.extraction.models.helpers import *
from parsee.extraction.models.model_loader import get_llm_base_model
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.converters.image_creation import from_numpy

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

# Another way to run your own prompts (if you want to send images directly to the APIs for example) is the following:

# create a connection to the llm you want to send the prompt to, based on the model you chose
llm = get_llm_base_model(gpt_model)

# read the image with e.g. cv2 (adjust the path)
image = cv2.imread("PATH_TO_IMAGE")

# create a base64 image
base64_image = from_numpy(image)

# define your prompt
prompt = Prompt(main_task="What can you see in this image?", available_data=[base64_image], intro=None)

# send prompt to the LLM
answer, _ = llm.make_prompt_request(prompt)

print(answer)