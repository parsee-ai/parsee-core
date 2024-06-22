from functools import reduce
import math

from parsee.storage.interfaces import *
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.cloud.api import ParseeCloud
from parsee.cloud.cloud_image_fetcher import CloudImageFetcher


class CloudFileManager(DocumentManager):

    def __init__(self, storage: StorageManager, settings: ChatSettings, cloud: ParseeCloud):
        super().__init__(storage, settings)
        self.cloud = cloud

    def load_documents(self, references: List[FileReference], multimodal: bool, search_term: Optional[str], max_images: Optional[int], max_tokens: Optional[int]) -> Union[str, List[Base64Image]]:
        unique_identifiers = set([x.source_identifier for x in references])

        if self.settings.search_strategy == SearchStrategy.VECTOR and search_term is None:
            raise Exception("search terms has to be provided for vector search")

        if self.settings.search_strategy == SearchStrategy.VECTOR:
            docs = self.storage.vector_store.find_closest_elements_from_references(references, search_term, "", False, self.settings.max_el_in_memory)
        elif self.settings.search_strategy == SearchStrategy.START:
            docs = []
            total_added = 0
            for identifier in unique_identifiers:
                doc = self.cloud.get_document(identifier)
                if total_added + len(doc.elements) > self.settings.max_el_in_memory:
                    to_add = self.settings.max_el_in_memory - total_added
                    if to_add > 0:
                        doc.elements = doc.elements[0:to_add]
                    else:
                        break
                docs.append(doc)
        else:
            raise Exception("unknown search strategy")

        if multimodal:
            output_by_doc = {}
            total_images = 0
            for doc in docs:
                output_by_doc[doc.source_identifier] = self.storage.image_creator.get_images(doc, doc.elements, self.settings.max_images_to_load_per_doc, None)
                total_images += len(output_by_doc[doc.source_identifier])
            if max_images is not None and total_images > max_images:
                max_images_per_file = math.floor(max_images/len(output_by_doc.keys()))
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
                output_by_doc[doc.source_identifier] = self.settings.encoding.encode(str(doc))
                total_tokens += len(output_by_doc[doc.source_identifier])
            if total_tokens > max_tokens:
                max_tokens_per_document = math.floor(max_tokens / len(output_by_doc.keys()))
                output = ""
                for k, tokens in enumerate(output_by_doc.values()):
                    output += f"[START OF DOCUMENT #{k + 1}]\n"
                    output += self.settings.encoding.decode(tokens[0:max_tokens_per_document])
                    output += f"[END OF DOCUMENT #{k + 1}]\n"
                return output
            else:
                output = ""
                for k, doc in enumerate(docs):
                    output += f"[START OF DOCUMENT #{k + 1}]\n"
                    output += str(doc)
                    output += f"[END OF DOCUMENT #{k + 1}]\n"
                return output
