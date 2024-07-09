from decimal import Decimal
from typing import *
import math
from functools import reduce

from parsee.templates.job_template import JobTemplate
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.storage.vector_stores.interfaces import VectorStore
from parsee.extraction.extractor_elements import FileReference
from parsee.converters.image_creation import ImageCreator
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.chat.custom_dataclasses import ChatSettings


class StorageManager:

    vector_store: VectorStore
    image_creator: ImageCreator

    def __init__(self, vector_store: VectorStore, image_creator: ImageCreator):
        self.vector_store = vector_store
        self.image_creator = image_creator

    def db_values_template(self, job_template: JobTemplate, strict: bool) -> JobTemplate:
        raise NotImplementedError

    def log_expense(self, service: str, amount: Decimal, class_id: str):
        raise NotImplementedError

    def get_available_models(self) -> List[MlModelSpecification]:
        raise NotImplementedError


class DocumentManager:

    storage: StorageManager

    def __init__(self, storage: StorageManager, settings: ChatSettings):
        self.storage = storage
        self.settings = settings

    def _load_documents(self, references: List[FileReference], multimodal: bool, search_term: Optional[str], max_images: Optional[int], max_tokens: Optional[int], load_function: Callable, show_chunk_index: bool) -> Union[str, List[Base64Image]]:
        # find and load the most relevant documents
        docs = []
        unique_identifiers = set([x.source_identifier for x in references])
        for source_identifier in self.storage.vector_store.sort_identifiers_by_relevance(unique_identifiers, search_term):
            total_added = 0
            doc = load_function(source_identifier)
            # check if all elements should be taken or not
            take_all = len([x for x in references if x.source_identifier == doc.source_identifier and x.element_index is None]) > 0
            if not take_all:
                allowed_element_indexes = [x.element_index for x in references if x.source_identifier == doc.source_identifier and x.element_index is not None]
                doc.elements = [x for x in doc.elements if x.source.element_index in allowed_element_indexes]
            if total_added + len(doc.elements) > self.settings.max_el_in_memory:
                to_add = self.settings.max_el_in_memory - total_added
                if to_add > 0:
                    doc.elements = doc.elements[0:to_add]
                else:
                    break
            docs.append(doc)

        if multimodal:
            if show_chunk_index:
                raise Exception("chunks can't be displayed with multimodal option")
            output_by_doc = {}
            total_images = 0
            for doc in docs:
                output_by_doc[doc.source_identifier] = self.storage.image_creator.get_images(doc, doc.elements, self.settings.max_images_to_load_per_doc, None)
                total_images += len(output_by_doc[doc.source_identifier])
            if max_images is not None and total_images > max_images:
                max_images_per_file = math.floor(max_images / len(output_by_doc.keys()))
                output = []
                for k, values in output_by_doc.items():
                    if len(values) > max_images_per_file:
                        if k == list(output_by_doc.keys())[-1]:
                            images_left = max_images - len(output)
                            output += values[0:images_left]
                        else:
                            output += values[0:max_images_per_file]
                    else:
                        output += values
                return output
            else:
                return reduce(lambda acc, x: acc + x, output_by_doc.values(), [])
        else:
            output_by_doc = {}
            total_tokens = 0
            for k, doc in enumerate(docs):
                output_by_doc[doc.source_identifier] = self.settings.encoding.encode(doc.to_string(show_chunk_index))
                total_tokens += len(output_by_doc[doc.source_identifier])
            if total_tokens > max_tokens:
                max_tokens_per_document = math.floor(max_tokens / len(output_by_doc.keys()))
                output = ""
                for k, tokens in enumerate(output_by_doc.values()):
                    output += f"[START OF DOCUMENT with index {k}]\n"
                    output += self.settings.encoding.decode(tokens[0:max_tokens_per_document])
                    output += f"[END OF DOCUMENT with index {k}]\n\n"
                return output
            else:
                output = ""
                for k, doc in enumerate(docs):
                    output += f"[START OF DOCUMENT with index {k}]\n"
                    output += doc.to_string(show_chunk_index)
                    output += f"[END OF DOCUMENT with index {k}]\n\n"
                return output

    def load_documents(self, references: List[FileReference], multimodal: bool, search_term: Optional[str], max_images: Optional[int], max_tokens: Optional[int], show_chunk_index: bool = False) -> Union[str, List[Base64Image]]:
        raise NotImplementedError
