# Parsee AI

Parsee AI is an open source data extraction and structuring framework created by <a href="https://github.com/SimFin">SimFin</a>. Parsee is designed to be able to combine different machine learning techniques (not limited to LLMs) in order to achieve maximum precision and minimize costs where possible, in order to extract fully structured data form a large amount of source documents (main focus on PDFs, HTML files and images).

Parsee can be used entirely from your local Python environment. We also provide a hosted version of Parsee, where you can edit and share your <a>extraction templates</a> and also run jobs in the cloud: <a href="https://app.parsee.ai">app.parsee.ai</a>.

Parsee is specialized for the extraction of data from a financial domain (tables, numbers etc.) but can be used for other use-cases as well.

## TLDR Version:

pip install parsee

...


## Detailed Overview

### Step 1: Define an Extraction Template

Extraction templates are the cornerstone for every extraction you run in Parsee. An extraction template is basically a JSON file, that defines exactly what information you want to extract from a document. These templates can then easily be shared, duplicated etc. The templates also contain information about the data type you want to have returned, such that you don't have to handle verbose output from LLMs for example.



### Use Cases

#### Extracting simple data from an invoice: Invoice total

*Main concepts used:* general queries, output data types

#### Extracting complex data from an invoice: Invoice total and associated currency

*Main concepts used:* general queries with meta information, such as currencies, time periods and units

#### Extracting strictly tabular data: Profit & Loss Statement Extraction from annual report of a company

*Main concepts used:* Table detection, column info structuring (meta information), row mapping
