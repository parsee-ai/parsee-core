from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import *
from typing import Any, Callable

import math
from functools import reduce

from parsee.templates.job_template import JobTemplate
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.storage.vector_stores.interfaces import VectorStore
from parsee.extraction.extractor_elements import FileReference, StandardDocumentFormat
from parsee.converters.image_creation import ImageCreator
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.settings import chat_settings
import logging
logger = logging.getLogger(__name__)


class Modality(Enum):
    """The representation to load for referenced documents."""

    IMAGES = "IMAGES"
    TEXT = "TEXT"
    IMAGES_AND_TEXT = "IMAGES_AND_TEXT"


@dataclass
class DocumentContent:
    """Loaded document content prepared for downstream chat/model prompts."""

    images: list[Base64Image] | None
    text: str | None


class StorageManager:

    vector_store: VectorStore
    image_creator: ImageCreator

    def __init__(self, vector_store: VectorStore, image_creator: ImageCreator):
        """Create a storage manager backed by a vector store and image creator."""
        self.vector_store = vector_store
        self.image_creator = image_creator

    def db_values_template(self, job_template: JobTemplate, strict: bool) -> JobTemplate:
        """Return a storage-specific copy of a job template with persisted values applied."""
        raise NotImplementedError

    def log_expense(self, service: str, amount: Decimal, class_id: str):
        """Persist or emit the cost incurred by a service for a class/template item."""
        raise NotImplementedError

    def get_available_models(self) -> List[MlModelSpecification]:
        """Return model specifications available to this storage backend."""
        raise NotImplementedError


class DocumentManager:

    storage: StorageManager

    def __init__(self, storage: StorageManager):
        """Create a document manager using the supplied storage services."""
        self.storage = storage

    def _find_docs(self, load_function: Callable[..., StandardDocumentFormat], references: list[FileReference], search_term: str | None) -> list[
        StandardDocumentFormat]:
        """Load referenced documents in relevance order and filter them to referenced elements.

        A reference without an element index loads the full document. If all references
        for a document point to specific elements, only those elements are retained.
        """
        docs = []
        unique_identifiers = set([x.source_identifier for x in references])
        for source_identifier in self.storage.vector_store.sort_identifiers_by_relevance(unique_identifiers,
                                                                                         search_term):
            total_added = 0
            doc = load_function(source_identifier)
            # check if all elements should be taken or not
            take_all = len(
                [x for x in references if x.source_identifier == doc.source_identifier and x.element_index is None]) > 0
            if not take_all:
                allowed_element_indexes = [x.element_index for x in references if
                                           x.source_identifier == doc.source_identifier and x.element_index is not None]
                doc.elements = [x for x in doc.elements if x.source.element_index in allowed_element_indexes]
            if total_added + len(doc.elements) > chat_settings.max_el_in_memory:
                to_add = chat_settings.max_el_in_memory - total_added
                if to_add > 0:
                    doc.elements = doc.elements[0:to_add]
                else:
                    break
            docs.append(doc)
        return docs

    def _extract_images(self, docs: list[StandardDocumentFormat], max_images: int | None) -> list[Base64Image]:
        """Extract images from documents and optionally cap the total returned image count."""
        output_by_doc = {}
        total_images = 0
        for doc in docs:
            output_by_doc[doc.source_identifier] = self.storage.image_creator.get_images(doc, doc.elements,
                                                                                         chat_settings.max_images_to_load_per_doc,
                                                                                         None)
            total_images += len(output_by_doc[doc.source_identifier])
        if max_images is not None and total_images > max_images:
            doc_identifiers = [doc.source_identifier for doc in docs]
            max_images_per_file = math.floor(max_images / len(output_by_doc.keys()))
            logger.warning(f"There are too many images to load, taking maximum {max_images_per_file} images per document. Document identifiers: {doc_identifiers}")
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

    def _extract_text(self, docs: list[StandardDocumentFormat], show_chunk_index: bool) -> str:
        """Render documents as prompt text, wrapped with stable document index markers."""
        output = []
        for k, doc in enumerate(docs):
            doc_text = doc.to_string(show_chunk_index)
            output.append(f"[START OF DOCUMENT with index {k}]\n{doc_text}[END OF DOCUMENT with index {k}]\n\n")
        return "".join(output)

    def _load_documents(self, references: List[FileReference], modality: Modality, search_term: Optional[str],
                        max_images: Optional[int], load_function: Callable, show_chunk_index: bool) -> DocumentContent:
        """Load referenced documents and return content for the requested modality."""
        if modality == Modality.IMAGES and show_chunk_index:
            raise ValueError(f"Chunks can't be displayed with {Modality.IMAGES} option")
        docs = self._find_docs(load_function, references, search_term)
        match modality:
            case Modality.IMAGES:
                images = self._extract_images(docs, max_images)
                return DocumentContent(images=images, text=None)
            case Modality.TEXT:
                text = self._extract_text(docs, show_chunk_index)
                return DocumentContent(images=None, text=text)
            case Modality.IMAGES_AND_TEXT:
                images = self._extract_images(docs, max_images)
                text = self._extract_text(docs, show_chunk_index)
                return DocumentContent(images=images, text=text)

    def load_documents(self, references: List[FileReference], modality: Modality, search_term: str | None, max_images: int | None, show_chunk_index: bool = False) -> DocumentContent:
        """Load document content for the references using the backend-specific loader."""
        raise NotImplementedError
