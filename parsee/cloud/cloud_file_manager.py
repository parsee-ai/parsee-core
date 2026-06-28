from parsee.storage.interfaces import *
from parsee.cloud.api import ParseeCloud


class CloudFileManager(DocumentManager):

    def __init__(self, storage: StorageManager, cloud: ParseeCloud):
        super().__init__(storage)
        self.cloud = cloud

    def load_documents(self, references: List[FileReference], modality: Modality, search_term: Optional[str], max_images: Optional[int], show_chunk_index: bool = False) -> DocumentContent:
        return self._load_documents(references, modality, search_term, max_images, self.cloud.get_document, show_chunk_index)
