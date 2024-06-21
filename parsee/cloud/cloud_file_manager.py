from parsee.storage.interfaces import *
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.cloud.api import ParseeCloud
from parsee.cloud.cloud_image_fetcher import CloudImageFetcher


class CloudFileManager(DocumentManager):

    def __init__(self, storage: StorageManager, settings: ChatSettings, cloud: ParseeCloud):
        super().__init__(storage, settings)
        self.cloud = cloud

    def load_documents(self, references: List[FileReference], multimodal: bool, search_term: Optional[str]) -> Union[str, List[Base64Image]]:
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
            output = []
            for doc in docs:
                images = self.storage.image_creator.get_images(doc, doc.elements, self.settings.max_images, None)
                output += images
            return output
        else:
            output = ""
            for k, doc in enumerate(docs):
                output += f"[START OF DOCUMENT #{k+1}]\n"
                output += str(doc)
                output += f"[END OF DOCUMENT #{k+1}]\n"
            return output
