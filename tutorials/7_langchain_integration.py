"""
You can also use langchain document loaders with Parsee, instead of our proprietary converters.
Note: you will usually get better results using the Parsee converters for PDFs/images instead of the langchain loaders.
"""

from parsee.converters.langchain import langchain_loader_to_sdf
from parsee.utils.enums import DocumentType
from langchain.document_loaders.pdf import PyPDFLoader

parsee_document = langchain_loader_to_sdf(PyPDFLoader("/Users/thomasflassbeck/Desktop/bayer1.pdf"), DocumentType.PDF, "test")

print(parsee_document)