
from parsee.storage.interfaces import *


class LocalFileManager(DocumentManager):

    def __init__(self, storage: StorageManager, settings: ChatSettings):
        super().__init__(storage, settings)

    def load_documents(self, references: List[FileReference], multimodal: bool, search_term: Optional[str]) -> Union[str, List[Base64Image]]:
        pass # TODO

    def load_fragments(self, references: List[FileReference]) -> List[ExtractedEl]:
        pass # TODO