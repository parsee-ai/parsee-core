from types import SimpleNamespace

import pytest

from parsee.extraction.extractor_dataclasses import Base64Image, ExtractedSource
from parsee.extraction.extractor_elements import ExtractedEl, FileReference, StandardDocumentFormat
from parsee.storage.interfaces import DocumentManager, Modality
from parsee.utils.enums import DocumentType, ElementType


class FakeVectorStore:
    def __init__(self, ordered_identifiers):
        self.ordered_identifiers = ordered_identifiers
        self.calls = []

    def sort_identifiers_by_relevance(self, source_identifiers, search_query):
        self.calls.append((source_identifiers, search_query))
        return [x for x in self.ordered_identifiers if x in source_identifiers]


class FakeImageCreator:
    def __init__(self, images_by_identifier):
        self.images_by_identifier = images_by_identifier
        self.calls = []

    def get_images(self, document, element_selection, max_images, max_image_size):
        self.calls.append((document.source_identifier, element_selection, max_images, max_image_size))
        return self.images_by_identifier[document.source_identifier]


def make_element(index, text):
    source = ExtractedSource(DocumentType.TEXT, None, None, index, None)
    return ExtractedEl(ElementType.TEXT, source, text)


def make_doc(source_identifier, texts):
    return StandardDocumentFormat(
        DocumentType.TEXT,
        source_identifier,
        [make_element(index, text) for index, text in enumerate(texts)],
        None,
    )


def make_reference(source_identifier, element_index=None):
    return FileReference(source_identifier, DocumentType.TEXT, element_index)


def make_manager(ordered_identifiers=None, images_by_identifier=None):
    storage = SimpleNamespace(
        vector_store=FakeVectorStore(ordered_identifiers or []),
        image_creator=FakeImageCreator(images_by_identifier or {}),
    )
    return DocumentManager(storage)


def test_find_docs_loads_in_relevance_order_and_filters_referenced_elements():
    manager = make_manager(["doc-b", "doc-a"])
    docs_by_identifier = {
        "doc-a": make_doc("doc-a", ["a0", "a1", "a2"]),
        "doc-b": make_doc("doc-b", ["b0", "b1", "b2"]),
    }
    references = [
        make_reference("doc-a", 2),
        make_reference("doc-b"),
        make_reference("doc-a", 0),
    ]

    docs = manager._find_docs(docs_by_identifier.__getitem__, references, "revenue")

    assert [doc.source_identifier for doc in docs] == ["doc-b", "doc-a"]
    assert [element.get_text() for element in docs[0].elements] == ["b0", "b1", "b2"]
    assert [element.get_text() for element in docs[1].elements] == ["a0", "a2"]
    assert manager.storage.vector_store.calls == [({"doc-a", "doc-b"}, "revenue")]


def test_load_documents_returns_text_content_for_text_modality():
    manager = make_manager(["doc-a"])
    doc = make_doc("doc-a", ["first", "second"])

    result = manager._load_documents(
        [make_reference("doc-a")],
        Modality.TEXT,
        "query",
        max_images=None,
        load_function=lambda _: doc,
        show_chunk_index=True,
    )

    assert result.images is None
    assert result.text == (
        "[START OF DOCUMENT with index 0]\n"
        "[chunk 0] first\n"
        "[chunk 1] second\n"
        "[END OF DOCUMENT with index 0]\n\n"
    )


def test_load_documents_returns_images_and_text_for_combined_modality():
    first_image = Base64Image("image/png", "first")
    second_image = Base64Image("image/png", "second")
    manager = make_manager(["doc-a"], {"doc-a": [first_image, second_image]})
    doc = make_doc("doc-a", ["first"])

    result = manager._load_documents(
        [make_reference("doc-a")],
        Modality.IMAGES_AND_TEXT,
        "query",
        max_images=None,
        load_function=lambda _: doc,
        show_chunk_index=False,
    )

    assert result.images == [first_image, second_image]
    assert result.text == "[START OF DOCUMENT with index 0]\nfirst\n[END OF DOCUMENT with index 0]\n\n"


def test_load_documents_rejects_chunk_indexes_for_image_only_modality():
    manager = make_manager(["doc-a"])

    with pytest.raises(ValueError, match="Chunks can't be displayed"):
        manager._load_documents(
            [make_reference("doc-a")],
            Modality.IMAGES,
            "query",
            max_images=None,
            load_function=lambda _: make_doc("doc-a", ["first"]),
            show_chunk_index=True,
        )


def test_extract_images_caps_images_across_documents():
    images_by_identifier = {
        "doc-a": [Base64Image("image/png", f"a-{index}") for index in range(3)],
        "doc-b": [Base64Image("image/png", f"b-{index}") for index in range(3)],
    }
    manager = make_manager(images_by_identifier=images_by_identifier)

    result = manager._extract_images([make_doc("doc-a", ["a"]), make_doc("doc-b", ["b"])], max_images=3)

    assert result == [
        Base64Image("image/png", "a-0"),
        Base64Image("image/png", "b-0"),
        Base64Image("image/png", "b-1"),
    ]
