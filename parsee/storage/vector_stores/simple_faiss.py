import os
import uuid
from typing import *
import pickle
import io

import numpy as np
import faiss
import tiktoken

from src.storage.vector_stores.interfaces import VectorStore
from src.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl
from src.utils.enums import ElementType


class SimpleFaissStore(VectorStore):

    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.min_chunk_size_characters = 1000
        self.k = 100

    def make_index(self, document: StandardDocumentFormat, tables_only: bool) -> Tuple[List[List[int]], any]:
        data = []
        current_batch = {"chars": 0, "element_indices": [], "text": ""}
        for k, el in enumerate(document.elements):
            if el.el_type == ElementType.TABLE:
                data.append(
                    {"element_indices": [el.source.element_index], "text": el.get_text()}
                )
            elif el.el_type == ElementType.TEXT and not tables_only:
                current_text = el.get_text_llm(False)

                current_batch["chars"] += len(current_text)
                current_batch["element_indices"].append(el.source.element_index)
                current_batch["text"] += "\n " if current_batch["text"] != "" else ""
                current_batch["text"] += current_text

                if current_batch["chars"] > self.min_chunk_size_characters:
                    data.append(
                        {"element_indices": current_batch["element_indices"], "text": current_batch["text"]}
                    )
                    current_batch = {"chars": 0, "element_indices": [], "text": ""}
        # check last batch
        if current_batch["chars"] > 0:
            data.append(
                {"element_indices": current_batch["element_indices"], "text": current_batch["text"]}
            )

        sentence_embeddings = self.encoder.encode([x["text"] for x in data])

        d = sentence_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(sentence_embeddings)

        return [x["element_indices"] for x in data], index

    def find_closest_elements(self, document: StandardDocumentFormat, search_element_title: str, keywords: Optional[str], tables_only: bool = True) -> List[ExtractedEl]:

        element_indices, index = self.make_index(document, tables_only)

        query = f"{search_element_title}" + (f"; {keywords}" if keywords is not None else "")

        xq = self.encoder.encode(query)

        _, results = index.search(xq, self.k)  # search

        indices = results.tolist()[0]

        all_element_indices = []
        for idx in indices:
            if idx >= 0:
                all_element_indices += element_indices[idx]

        return [document.elements[el_idx] for el_idx in all_element_indices]
