# Parsee AI

Parsee AI is an opinionated, high-level, open source data extraction and structuring framework created by <a href="https://github.com/SimFin">SimFin</a>. Our goal is to make the structuring of data from the most common sources of unstructured data (mainly PDFs, HTML files and images) as easy as possible.

Parsee can be used entirely from your local Python environment. We also provide a hosted version of Parsee, where you can edit and share your <a>extraction templates</a> and also run jobs in the cloud: <a href="https://app.parsee.ai">app.parsee.ai</a>.

Parsee is specialized for the extraction of data from a financial domain (tables, numbers etc.) but can be used for other use-cases as well.

While Parsee has first class support for LLMs, the goal of Parsee is also not to limit itself to LLMs, as at least as of today, custom non-LLM models usually outperform LLMs in terms of speed, accuracy and cost-efficiency given that there is a large enough dataset available. At SimFin for example, we are using Parsee for extracting data from hundreds of documents daily, entirely without LLMs as we have already a substantial dataset built up for our tasks.

For handling tables from PDFs and images properly, we also released our open source PDF table extraction package: https://github.com/parsee-ai/parsee-pdf-reader

## Installation:

Recommended install with poetry: https://python-poetry.org/docs/

    poetry add parsee-core

Alternatively:

    pip install parsee-core

## Quick Example

*Goal:*

Given we have some invoices, we want to:
1) extract the invoice total, but not just the number, also the currency attached to it.
2) extract the issuer of the invoice

### Imports

    import os
    from parsee.templates.helpers import StructuringItem, MetaItem, create_template
    from parsee.extraction.models.helpers import *
    from parsee.converters.main import load_document, from_text
    from parsee.extraction.run import run_job_with_single_model
    from parsee.utils.enums import *
    
### Step 1: create an extraction template
    
    question_to_be_answered = "What is the invoice total?"
    output_type = OutputType.NUMERIC
    
    meta_currency_question = "What is the currency?"
    meta_currency_output_type = OutputType.LIST # we want the model to use a pre-defined item from a list, this is basically a classification
    meta_currency_list_values = ["USD", "EUR", "Other"] # any list of strings can be used here
    
    meta_item = MetaItem(meta_currency_question, meta_currency_output_type, list_values=meta_currency_list_values)
    
    invoice_total = StructuringItem(question_to_be_answered, output_type, meta_info=[meta_item])
    
let's also define an item for the issuer of the invoice

    invoice_issuer = StructuringItem("Who is the issuer of the invoice?", OutputType.ENTITY)
    
    job_template = create_template([invoice_total, invoice_issuer])

As an alternative to using extraction templates, you can also run your own custom prompts, more in <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/10_custom_prompts.py">tutorial 10</a>.
    
### Step 2: define a model

In the following we will use the Mixtral model from Replicate, requires an API key: https://replicate.com/
    
    replicate_api_key = os.getenv("REPLICATE_KEY")
    replicate_model = replicate_config(replicate_api_key, "mistralai/mixtral-8x7b-instruct-v0.1")

If you intend to use a model that has multimodal capabilities such as GPT 4 or Claude 3, you can enable the multimodal queries by setting the multimodal setting to True (you can also specify how many images should be passed to the modal at most and the maximum size for each image):

    gpt_model = gpt_config(os.getenv("OPENAI_KEY"), None, openai_model_name="gpt-4-turbo", multimodal=True, max_images=3, max_image_size=2000)

or for Anthropic models:
    
    anthropic_model = anthropic_config(os.getenv("ANTHROPIC_KEY"), "claude-3-opus-20240229", None, multimodal=True, max_images=1, max_image_size=800)

of course you can also load a locally hosted model with Ollama:

    ollama_model = ollama_config("mistral")
    
### Step 3: load a document
Parsee converts all data (strings, file contents etc.) to a standardized format, the class for this is called StandardDocumentFormat.

#### a) Let's first create a StandardDocumentFormat object from a simple string
    
    input_string = "The invoice total amounts to 12,5 Euros and is due on Feb 28th 2024. Invoice to: Some company LLC. Thanks for using the services of CloudCompany Inc."
    document = from_text(input_string)

#### b) We can also simply load and convert files into the StandardDocumentFormat with the help of the converters that are included in Parsee, let's use an actual PDF invoice now

    file_path = "../tests/fixtures/Midjourney_Invoice-DBD682ED-0005.pdf"
    document = load_document(file_path)
    
### Step 4: run the extraction

    _, _, answers_open_source_model = run_job_with_single_model(document, job_template, replicate_model)

If we look at the answers of the model we get the following:

    answers_open_source_model[0].class_value
    >> '11.9'
    answers_open_source_model[0].meta[0].class_value
    >> 'USD'

We can also use a different model to run the same extraction:
    
    # enter your key manually here or load from an .env file
    open_ai_api_key = os.getenv("OPENAI_KEY")
    gpt_model = gpt_config(open_ai_api_key)
    
    _, _, answers_gpt = run_job_with_single_model(document, job_template, gpt_model)


## Full Tutorials

0) Extraction Templates & Basics: <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/1_basic_example_meta.py">Python Code.</a>

1) Meta Items: <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/0_basic_example.py">Python Code.</a>

2) Table Extraction: <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/2_table_extraction.py">Python Code.</a>

3) Loading Templates from Parsee Cloud: <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/3_loading_templates.py">Python Code.</a>

4) Saving Templates: <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/4_saving_templates.py">Python Code.</a>

5) Datasets: <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/5_datasets.py">Python Code.</a>

6) Model Evaluations: <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/6_model_evaluations.py">Python Code.</a>

7) Langchain Integration: <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/7_langchain_integration.py">Python Code.</a>

8) Multimodal Models: <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/multimodal_models.py">Python Code.</a>

9) Using Parsee Cloud to Load Images for Multimodal Applications: <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/9_loading_images_from_parsee_cloud_for_multimodal_use.py">Python Code.</a>

10) Custom Prompts: <a href="https://github.com/parsee-ai/parsee-core/blob/master/tutorials/10_custom_prompts.py">Python Code.</a>

## Basic Rules for Extraction Templates

In the following, we will only focus on the "general questions" items of the extraction templates. The logic for table detection/structuring items is quite similar and we will add some more explanations for them in the future.

### Base Logic

In the most basic sense, every question you define under the "general questions" category can have exactly one answer.

If no answer can be found for a question (or meta item), the answer can always be „n/a“, meaning that the parsing of the values was not successful or the model did not have an answer. In that sense, all outputs are "nullable" but will be represented by the string value "n/a" in case they are null.

A question can have more than one answer only when there is a meta item defined, which will create an „axis“ along which the model can give different answers to the same question.

### Example

For the question: What is the invoice total? (output type numeric)

If there is no meta item defined, the model can answer this question only with one number (because of the numeric output type) or with "n/a".

e.g.
- Invoice total: 10.0

OR

- Invoice total: 21.5

OR

- Invoice total: n/a

If for some reason you want the model to not just respond with one answer, but in case there are maybe several different answers to a question for a single document, you can add a meta item.

If we define a meta item „invoice date“ and attach it to the invoice total, the model can now theoretically give several answers for the same document, differentiated by their meta ID:

e.g.
- (first answer) Invoice total: 10.0 as per 2022-03-01

AND

- (second answer) Invoice total: 24.0 as per 2022-06-01

So you can imagine the meta items as a sort of „key“, in the sense that as long as the meta values differ for 2 items, their keys will be different. All output values can be imagined as key value pairs.