"""
A dataset requires to connect features (which in case of LLMs, are basically the prompts) with output values,
such that a model can learn the relation between the two.
In the following we will illustrate this process for LLMs.
You can also use Parsee Cloud (at https://app.parsee.ai) to easily create datasets and label/correct data in a graphical user interface.
Once we have a dataset, we can also run comparisons between different models (see tutorial 6).
"""
from parsee.templates.helpers import StructuringItem, MetaItem, create_template
from parsee.extraction.extractor_dataclasses import AssignedAnswer, AssignedMeta
from parsee.raw_converters.main import load_document
from parsee.datasets.main import create_dataset_rows
from parsee.datasets.writers.disk_writer import CsvDiskWriter
from parsee.utils.enums import *

# Let's use the invoice example again, with the two questions: invoice total and issuer of invoice
meta_currency = MetaItem("What is the currency?", OutputType.LIST, list_values=["USD", "EUR", "Other"])
invoice_total = StructuringItem("What is the invoice total?", OutputType.NUMERIC, meta_info=[meta_currency])
invoice_issuer = StructuringItem("Who is the issuer of the invoice?", OutputType.ENTITY)
job_template = create_template([invoice_total, invoice_issuer])

# Let's use two different documents to create a datasets
first_doc = load_document("./tests/fixtures/documents/pdf/Midjourney_Invoice-DBD682ED-0005.pdf")
second_doc = load_document("./tests/fixtures/documents/pdf/INV-CF12005.pdf")

# We can assign the correct values using the AssignedAnswer and AssignedMeta classes. For these, we have to provide IDs of our questions.
# Let's start with the first document
# Let's assign a value for our invoice total question
question_id = invoice_total.id
# Looking at the document, the correct answer is '11.90'. All answers are strings here.
correct_answer = "11.90"
# The currency is USD -> here we have to use the ID of the meta item, not the 'main' question. You can also modify the IDs using the 'assigned_id' property when you create the object.
currency_assigned = AssignedMeta(meta_currency.id, "USD")
# For training a model based on a dataset, it is also better to provide the used 'sources' for each item, so that the model can also improve in returning these. For this example we will omit the sources for simplicity.
invoice_total_answer_first_doc = AssignedAnswer(question_id, correct_answer, [currency_assigned], [])
# Let's create an answer for the invoice issuer question also
invoice_issuer_answer_first_doc = AssignedAnswer(invoice_issuer.id, "Midjourney Inc", [], [])
# Let' repeat the same for the second doc
invoice_total_answer_second_doc = AssignedAnswer(invoice_total.id, "5570.40", [currency_assigned], [])
# Let's create an answer for the invoice issuer question also
invoice_issuer_answer_second_doc = AssignedAnswer(invoice_issuer.id, "CloudFactory International Limited UK", [], [])

# in our dataset, each of these assigned answers is basically one row. We can now easily create a dataset with prompts and the assigned answers:
# by default, we limit the number of tokens to 4k, but you can modify this value (this is independent of the model used).
# If you provide a source for the assigned answers, parsee will also check that the source is really contained in the transformed document after applying the token limit. If not, no row will be returned (this is to make sure the model can only learn on samples where the answer is actually in the text and not cut off)
token_limit = 4000
dataset_rows = create_dataset_rows(job_template, first_doc, [invoice_total_answer_first_doc, invoice_issuer_answer_first_doc], max_tokens_prompt=token_limit)
# let's add the rows from the second document also
dataset_rows += create_dataset_rows(job_template, second_doc, [invoice_total_answer_second_doc, invoice_issuer_answer_second_doc], max_tokens_prompt=token_limit)
# to save the rows as csv, we can create a dataset writer
writer = CsvDiskWriter("/Users/thomasflassbeck/Desktop/temp/x")
# write the rows at the target destination as CSV
writer.write_rows(dataset_rows, "questions_invoice")

# in the next tutorial we will show how to evaluate different models on the dataset we just created
