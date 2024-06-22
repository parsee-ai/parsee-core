
from parsee.storage.interfaces import *


class LocalFileManager(DocumentManager):

    def __init__(self, storage: StorageManager, settings: ChatSettings):
        super().__init__(storage, settings)

    def load_documents(self, references: List[FileReference], multimodal: bool, search_term: Optional[str], max_images: Optional[int], max_tokens: Optional[int]) -> Union[str, List[Base64Image]]:
        pass # TODO