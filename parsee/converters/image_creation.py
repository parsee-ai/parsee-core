from typing import *
import tempfile
import base64
import shutil

import cv2
from numpy import ndarray

from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl
from parsee.utils.enums import DocumentType
from pdf_reader.helper import make_images_from_pdf
from pdf_reader.converter import is_image
from parsee.extraction.extractor_dataclasses import Base64Image


def resize(image_cv2: ndarray, max_image_size: Optional[int]) -> ndarray:
    height, width, channels = image_cv2.shape
    if not (height > max_image_size or width > max_image_size):
        return image_cv2
    bigger_val = max(width, height)
    scale_factor = max_image_size / bigger_val
    algo = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC
    img_resized = cv2.resize(image_cv2, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=algo)
    return img_resized


def from_file_paths(file_paths: List[str], max_image_size: int) -> List[Base64Image]:
    output = []
    for fp in file_paths:
        # type is always jpeg
        media_type = "image/jpeg"
        # open and resize image if necessary
        img = cv2.imread(fp)
        # resize
        img = resize(img, max_image_size)
        retval, buffer = cv2.imencode('.jpg', img)
        encoded_string = base64.b64encode(buffer).decode("utf-8")
        output.append(Base64Image(media_type, encoded_string))

    return output


class ImageCreator:

    def get_images(self, document: StandardDocumentFormat, element_selection: List[ExtractedEl], max_images: Optional[int], max_image_size: Optional[int]) -> List[any]:
        raise NotImplemented


class DiskImageCreator(ImageCreator):

    def get_images(self, document: StandardDocumentFormat, element_selection: List[ExtractedEl], max_images: Optional[int], max_image_size: Optional[int]) -> List[Base64Image]:

        if document.file_path is None:
            return []

        output = []

        if document.source_type == DocumentType.PDF:
            if is_image(document.file_path):
                output += from_file_paths([document.file_path], max_image_size)
            else:
                # collect all relevant pages
                page_indexes = []
                for el in element_selection:
                    if el.source.other_info is not None and "page_idx" in el.source.other_info and int(el.source.other_info["page_idx"]) not in page_indexes and (max_images is None or len(page_indexes) < max_images):
                        page_indexes.append(int(el.source.other_info["page_idx"]))
                temp_dir = tempfile.TemporaryDirectory()
                images = make_images_from_pdf(document.file_path, temp_dir.name, [max_image_size], None)
                file_paths = [images[max_image_size][x] for x in page_indexes]
                output += from_file_paths(file_paths, max_image_size)
                # delete temp dir
                shutil.rmtree(temp_dir.name)
        else:
            raise Exception("unsupported document type, images can only be created from PDFs. To create a PDF from a HTML file, use a tool like pdfkit, then run pdfkit.from_file('your_file') and use the output file instead of the HTML.")

        return output
