from parsee.utils.helper import get_source_identifier
from parsee.storage.interfaces import *
from parsee.extraction.extractor_elements import StandardDocumentFormat
from parsee.converters.main import load_document


class LocalFileManager(DocumentManager):

    def __init__(self, storage: StorageManager, settings: ChatSettings, document_paths: List[str]):
        super().__init__(storage, settings)
        self.source_identifier_to_paths = {}
        for path in document_paths:
            self.source_identifier_to_paths[get_source_identifier(path)] = path

    def add_files(self, document_paths: List[str]):
        for path in document_paths:
            self.source_identifier_to_paths[get_source_identifier(path)] = path

    def load_with_source_identifier(self, source_identifier: str) -> StandardDocumentFormat:
        if source_identifier not in self.source_identifier_to_paths:
            raise Exception("file not found")
        return load_document(self.source_identifier_to_paths[source_identifier])

    def load_documents(self, references: List[FileReference], multimodal: bool, search_term: Optional[str], max_images: Optional[int], max_tokens: Optional[int], show_chunk_index: bool = False) -> Union[str, List[Base64Image]]:
        return self._load_documents(references, multimodal, search_term, max_images, max_tokens, self.load_with_source_identifier, show_chunk_index)
