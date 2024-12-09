import os

from parsee import OutputType, StructuringItem, create_template, run_job_with_single_model, from_text
from parsee.chat.custom_dataclasses import *
from parsee.utils.enums import DocumentType
from parsee.storage.in_memory_storage import InMemoryStorageManager
from parsee.storage.local_file_manager import LocalFileManager
from parsee.converters.image_creation import DiskImageCreator
from parsee.extraction.models.helpers import ollama_config, gpt_config
from parsee.utils.helper import get_source_identifier
from parsee.chat.main import run_chat

"""
The previous examples revolve around extracting data from a single file. If you want to ask a question about more than one document ("document chat"), you can use the chat functionality.

Let's start with a fully local example:
If you want to use multimodal models, we need to specify a way to create images. For this we can either create images on the fly using the DiskImageCreator class
or we can read images from a directory if you already created them and use the DiskImageReader class. In the following example, we will be creating the images if they are being requested.
"""
storage = InMemoryStorageManager(None, DiskImageCreator())

# we can specify the paths to one or more files on the disk here
file_manager = LocalFileManager(storage, ["../tests/fixtures/fiver-march-FI15636047324.pdf", "../tests/fixtures/Midjourney_Invoice-DBD682ED-0005.pdf"])

# specify a model
model = ollama_config("llama3")

"""
Let's create a message and ask the model about the difference between the two files.
We will be feeding the two documents entirely to the model (given the token limits, which is handled automatically). You can also specify only snippets to be used (these could be returned also by a vector store or similar search of course),
for this you can use the "fragments" parameter and pass an ExtractedSource object, where you specify the index of the elements that should be used to answer the question.
Fragments will be set to None for this example, which means that the entire file(s) should be user to answer the question.
"""
message = Message("What is the difference between these two files?", [
    FileReference(get_source_identifier("../tests/fixtures/fiver-march-FI15636047324.pdf"), DocumentType.PDF, None),
    FileReference(get_source_identifier("../tests/fixtures/Midjourney_Invoice-DBD682ED-0005.pdf"), DocumentType.PDF, None)
])

chat_messages = run_chat(message, [], file_manager, [model], False)

print(chat_messages)

# # you can also pass a message history. You can decide if the references of the message history should be loaded or not with the 'most_recent_references_only' parameter of the 'run_chat' method.
# message1 = Message("Who is the issuer of the invoice?", [
#     FileReference(get_source_identifier("../tests/fixtures/fiver-march-FI15636047324.pdf"), DocumentType.PDF, None)
# ])
# message1_answer = Message("The invoice issuer is Fiverr Inc.", [])
# message2 = Message("Who is the issuer of the invoice?", [
#     FileReference(get_source_identifier("../tests/fixtures/Midjourney_Invoice-DBD682ED-0005.pdf"), DocumentType.PDF, None)
# ])
# message2_answer = Message("The invoice issuer is Midjourney Inc.", [])
# # we don't have to pass the references again as the history contains them
# message3 = Message("Which invoice total is higher?", [])
#
# chat_messages = run_chat(message3, [message1, message1_answer, message2, message2_answer], file_manager, [model], False)
#
# print(chat_messages)
#
# # we can also use multimodal models for the local files:
# message = Message("Which invoice total is higher?", [
#     FileReference(get_source_identifier("../tests/fixtures/fiver-march-FI15636047324.pdf"), DocumentType.PDF, None),
#     FileReference(get_source_identifier("../tests/fixtures/Midjourney_Invoice-DBD682ED-0005.pdf"), DocumentType.PDF, None)
# ])

# we are using ChatGPT-4o vision now
# model = gpt_config(os.getenv("OPENAI_KEY"), 10000, "gpt-4o", multimodal=True)
#
# chat_messages = run_chat(message, [], file_manager, [model], False)
#
# print(chat_messages)
#
# # Finally, we can also use the Parsee Extraction templates with the output from the chat, to structure the data:
# structuring_item = StructuringItem("Which invoice total is higher?", OutputType.LIST, ["Fiverr invoice", "Midjourney Invoice"])
#
# job_template = create_template([structuring_item])
#
# doc = from_text(str(chat_messages))
# model = gpt_config(os.getenv("OPENAI_KEY"), 10000, "gpt-4o", multimodal=False)
# _, _, answers = run_job_with_single_model(doc, job_template, model)
#
# print(answers)

