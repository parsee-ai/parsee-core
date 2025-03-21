from parsee.storage.interfaces import *
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.cloud.api import ParseeCloud


class CloudFileManager(DocumentManager):

    def __init__(self, storage: StorageManager, cloud: ParseeCloud):
        super().__init__(storage)
        self.cloud = cloud

    def load_documents(self, references: List[FileReference], multimodal: bool, search_term: Optional[str], max_images: Optional[int], max_tokens: Optional[int], show_chunk_index: bool = False) -> Union[str, List[Base64Image]]:
        return self._load_documents(references, multimodal, search_term, max_images, max_tokens, self.cloud.get_document, show_chunk_index)
