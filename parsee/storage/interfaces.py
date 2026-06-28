from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Callable, List, Optional

import math

from parsee.templates.job_template import JobTemplate
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.storage.vector_stores.interfaces import VectorStore
from parsee.extraction.extractor_elements import ExtractedEl, FileReference, StandardDocumentFormat
from parsee.converters.image_creation import ImageCreator
from parsee.extraction.extractor_dataclasses import Base64Image, ExtractedSource
from parsee.settings import chat_settings
import logging
logger = logging.getLogger(__name__)


class Modality(Enum):
    """Document representation requested from storage."""

    IMAGES = "IMAGES"
    TEXT = "TEXT"
    IMAGES_AND_TEXT = "IMAGES_AND_TEXT"


@dataclass
class DocumentContent:
    """Loaded document content prepared for chat prompts.

    Attributes:
        images: Base64-encoded page or element images grouped by document
            identifier, or `None` when images were not requested.
        text: Rendered page texts grouped by document identifier, or `None`
            when text was not requested.
    """

    images: dict[str, list[Base64Image]] | None
    texts: dict[str, list[str]] | None


class StorageManager:

    vector_store: VectorStore
    image_creator: ImageCreator

    def __init__(self, vector_store: VectorStore, image_creator: ImageCreator):
        """Initialize the storage manager.

        Args:
            vector_store: Vector store used to rank referenced document
                identifiers.
            image_creator: Service used to render document elements into images.

        Returns:
            None.
        """
        self.vector_store = vector_store
        self.image_creator = image_creator

    def db_values_template(self, job_template: JobTemplate, strict: bool) -> JobTemplate:
        """Return a job template with persisted database values applied.

        Args:
            job_template: Template whose classes and fields should be resolved
                against stored values.
            strict: Whether missing or incompatible stored values should be
                treated as errors by the concrete implementation.

        Returns:
            A storage-specific copy of `job_template` with persisted values
            populated.

        Raises:
            NotImplementedError: Always raised by the interface method.
        """
        raise NotImplementedError

    def log_expense(self, service: str, amount: Decimal, class_id: str):
        """Persist or emit the expense incurred by a service call.

        Args:
            service: Name or identifier of the service that incurred the cost.
            amount: Monetary amount to record.
            class_id: Identifier of the class or template item associated with
                the expense.

        Returns:
            None.

        Raises:
            NotImplementedError: Always raised by the interface method.
        """
        raise NotImplementedError

    def get_available_models(self) -> List[MlModelSpecification]:
        """Return model specifications available to this storage backend.

        Args:
            None.

        Returns:
            Model specifications that can be used by chat or extraction.

        Raises:
            NotImplementedError: Always raised by the interface method.
        """
        raise NotImplementedError


class DocumentManager:

    storage: StorageManager

    def __init__(self, storage: StorageManager):
        """Initialize the document manager.

        Args:
            storage: Storage services used for relevance ranking and image
                rendering.

        Returns:
            None.
        """
        self.storage = storage

    def _find_docs(self, load_function: Callable[..., StandardDocumentFormat], references: list[FileReference], search_term: str | None) -> list[
        StandardDocumentFormat]:
        """Load referenced documents in relevance order and filter them to referenced elements.

        A reference without an element index loads the full document. If all references
        for a document point to specific elements, only those elements are retained.

        Args:
            load_function: Callable that loads a document by source identifier.
            references: File references that identify documents and optionally
                specific document elements.
            search_term: Optional query used by the vector store to sort
                document identifiers by relevance.

        Returns:
            Loaded documents sorted by relevance and filtered to the referenced
            elements when applicable.
        """
        docs = []
        unique_identifiers = set([x.source_identifier for x in references])
        for source_identifier in self.storage.vector_store.sort_identifiers_by_relevance(unique_identifiers,
                                                                                         search_term or ""):
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

    def _extract_images(self, docs: list[StandardDocumentFormat], max_images: int | None) -> dict[str, list[Base64Image]]:
        """Extract images from documents and optionally cap the total count.

        Args:
            docs: Documents whose elements should be rendered as images.
            max_images: Maximum total number of images to return, or `None` to
                return all rendered images.

        Returns:
            Rendered images from the supplied documents grouped by document
            identifier. When `max_images` is set and the rendered total is
            larger, images are limited across documents.
        """
        output_by_doc = {}
        total_images = 0
        for doc in docs:
            element_selection: list[ExtractedEl | ExtractedSource] = list(doc.elements)
            output_by_doc[doc.source_identifier] = self.storage.image_creator.get_images(doc, element_selection,
                                                                                         chat_settings.max_images_to_load_per_doc,
                                                                                         None)
            total_images += len(output_by_doc[doc.source_identifier])
        if max_images is not None and total_images > max_images:
            doc_identifiers = [doc.source_identifier for doc in docs]
            max_images_per_file = math.floor(max_images / len(output_by_doc.keys()))
            logger.warning(f"There are too many images to load, taking maximum {max_images_per_file} images per document. Document identifiers: {doc_identifiers}")
            output = {}
            loaded_images = 0
            for k, values in output_by_doc.items():
                if len(values) > max_images_per_file:
                    if k == list(output_by_doc.keys())[-1]:
                        images_left = max_images - loaded_images
                        output[k] = values[0:images_left]
                    else:
                        output[k] = values[0:max_images_per_file]
                else:
                    output[k] = values
                loaded_images += len(output[k])
            return output
        else:
            return output_by_doc

    def _extract_texts(self, docs: list[StandardDocumentFormat], show_chunk_index: bool) -> dict[str, list[str]]:
        """Render documents as page texts grouped by document identifier.

        Args:
            docs: Documents to render as text.
            show_chunk_index: Whether each element should include chunk indexes
                in its rendered text.

        Returns:
            Page texts grouped by document identifier. Each page text contains
            elements with the same `source.other_info["page_idx"]` and is
            formatted like `StandardDocumentFormat.to_string`.
        """
        output_by_doc = {}
        for doc in docs:
            elements_by_page = {}
            for el in doc.elements:
                page_idx = 0
                if el.source.other_info is not None and "page_idx" in el.source.other_info:
                    page_idx = int(el.source.other_info["page_idx"])
                if page_idx not in elements_by_page:
                    elements_by_page[page_idx] = []
                elements_by_page[page_idx].append(el)

            output_by_doc[doc.source_identifier] = []
            for page_idx in sorted(elements_by_page.keys()):
                page_text = []
                for el in elements_by_page[page_idx]:
                    page_text.append((f"[chunk {el.source.element_index}] " if show_chunk_index else "") + el.get_text_llm(True) + "\n")
                output_by_doc[doc.source_identifier].append("".join(page_text))
        return output_by_doc

    def _load_documents(self, references: List[FileReference], modality: Modality, search_term: Optional[str],
                        max_images: Optional[int], load_function: Callable, show_chunk_index: bool) -> DocumentContent:
        """Load referenced documents and return content for the requested modality.

        Args:
            references: File references that identify the documents and elements
                to load.
            modality: Content representation to return.
            search_term: Optional query used to sort referenced documents by
                relevance.
            max_images: Maximum number of images to return when images are
                requested, or `None` for no explicit cap.
            load_function: Callable that loads a document by source identifier.
            show_chunk_index: Whether text output should include chunk indexes.

        Returns:
            Document content containing images, text, or both depending on
            `modality`.

        Raises:
            ValueError: If chunk indexes are requested for image-only loading.
        """
        if modality == Modality.IMAGES and show_chunk_index:
            raise ValueError(f"Chunks can't be displayed with {Modality.IMAGES} option")
        docs = self._find_docs(load_function, references, search_term)
        match modality:
            case Modality.IMAGES:
                images = self._extract_images(docs, max_images)
                return DocumentContent(images=images, texts=None)
            case Modality.TEXT:
                texts = self._extract_texts(docs, show_chunk_index)
                return DocumentContent(images=None, texts=texts)
            case Modality.IMAGES_AND_TEXT:
                images = self._extract_images(docs, max_images)
                texts = self._extract_texts(docs, show_chunk_index)
                return DocumentContent(images=images, texts=texts)

    def load_documents(self, references: List[FileReference], modality: Modality, search_term: str | None, max_images: int | None, show_chunk_index: bool = False) -> DocumentContent:
        """Load document content for references using a backend-specific loader.

        Args:
            references: File references that identify the documents and elements
                to load.
            modality: Content representation to return.
            search_term: Optional query used to sort referenced documents by
                relevance.
            max_images: Maximum number of images to return when images are
                requested, or `None` for no explicit cap.
            show_chunk_index: Whether text output should include chunk indexes.

        Returns:
            Document content containing images, text, or both depending on
            `modality`.

        Raises:
            NotImplementedError: Always raised by the interface method.
        """
        raise NotImplementedError
