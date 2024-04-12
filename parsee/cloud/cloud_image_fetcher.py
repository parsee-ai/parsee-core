from typing import *

from parsee.converters.image_creation import ImageCreator
from parsee.cloud.api import ParseeCloud
from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl
from parsee.extraction.extractor_dataclasses import Base64Image


class CloudImageFetcher(ImageCreator):

    def __init__(self, cloud: ParseeCloud):
        self.cloud = cloud

    def get_images(self, document: StandardDocumentFormat, element_selection: List[ExtractedEl], max_images: Optional[int], max_image_size: Optional[int]) -> List[Base64Image]:

        # collect all relevant pages
        page_indexes = []
        for el in element_selection:
            if el.source.other_info is not None and "page_idx" in el.source.other_info and int(el.source.other_info["page_idx"]) not in page_indexes and (
                    max_images is None or len(page_indexes) < max_images):
                page_indexes.append(int(el.source.other_info["page_idx"]))

        return [self.cloud.get_image(document.source_identifier, page_index, max_image_size) for page_index in page_indexes]
