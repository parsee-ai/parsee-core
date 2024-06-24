from typing import *
import tempfile
import base64
import shutil
import os

import cv2
from numpy import ndarray, frombuffer
import numpy as np

from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedEl, ExtractedSource
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


def from_bytes(file_content: bytes, max_image_size: int) -> Base64Image:
    # open and resize image if necessary
    jpg_as_np = frombuffer(file_content, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    # resize
    img = resize(img, max_image_size)
    return from_numpy(img)


def from_numpy(numpy_img: ndarray) -> Base64Image:
    retval, buffer = cv2.imencode('.jpg', numpy_img)
    encoded_string = base64.b64encode(buffer).decode("utf-8")
    return Base64Image("image/jpeg", encoded_string)


def get_media_type_simple(file_path: str) -> str:
    if file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
        return "image/jpeg"
    elif file_path.endswith(".png"):
        return "image/png"
    else:
        raise Exception("unsupported image type")


def from_file_paths(file_paths: List[str], max_image_size: int) -> List[Base64Image]:
    output = []
    for fp in file_paths:
        with open(fp, "rb") as f:
            output.append(from_bytes(f.read(), max_image_size))

    return output


class ImageCreator:

    def get_images(self, document: StandardDocumentFormat, element_selection: List[Union[ExtractedEl, ExtractedSource]], max_images: Optional[int], max_image_size: Optional[int]) -> List[Base64Image]:
        raise NotImplementedError


class DiskImageCreator(ImageCreator):

    def get_images(self, document: StandardDocumentFormat, element_selection: List[Union[ExtractedEl, ExtractedSource]], max_images: Optional[int], max_image_size: Optional[int]) -> List[Base64Image]:

        max_image_size = 2000 if max_image_size is None else max_image_size

        if document.file_path is None:
            return []

        output = []

        if document.source_type == DocumentType.PDF:
            if document.file_path is not None and is_image(document.file_path):
                output += from_file_paths([document.file_path], max_image_size)
            else:
                # collect all relevant pages
                page_indexes = []
                for el in element_selection:
                    if isinstance(el, ExtractedEl):
                        source = el.source
                    elif isinstance(el, ExtractedSource):
                        source = el
                    else:
                        raise Exception("unknown element type")
                    if source.other_info is not None and "page_idx" in source.other_info and int(source.other_info["page_idx"]) not in page_indexes and (max_images is None or len(page_indexes) < max_images):
                        page_indexes.append(int(source.other_info["page_idx"]))
                temp_dir = tempfile.TemporaryDirectory()
                images = make_images_from_pdf(document.file_path, temp_dir.name, [max_image_size], None)
                file_paths = [images[max_image_size][x] for x in page_indexes]
                output += from_file_paths(file_paths, max_image_size)
                # delete temp dir
                shutil.rmtree(temp_dir.name)
        else:
            raise Exception("unsupported document type, images can only be created from PDFs. To create a PDF from a HTML file, use a tool like pdfkit, then run pdfkit.from_file('your_file') and use the output file instead of the HTML.")

        return output


class DiskImageReader(ImageCreator):

    def __init__(self, images_dir: str):
        self.images_dir = images_dir

    def get_images(self, document: StandardDocumentFormat, element_selection: List[Union[ExtractedEl, ExtractedSource]], max_images: Optional[int], max_image_size: Optional[int]) -> List[Base64Image]:

        if document.source_type == DocumentType.PDF:
            if document.file_path is not None and is_image(document.file_path):
                page_indexes = [0]
            else:
                # collect all relevant pages
                page_indexes = []
                for el in element_selection:
                    if isinstance(el, ExtractedEl):
                        source = el.source
                    elif isinstance(el, ExtractedSource):
                        source = el
                    else:
                        raise Exception("unknown element type")
                    if source.other_info is not None and "page_idx" in source.other_info and int(source.other_info["page_idx"]) not in page_indexes and (
                            max_images is None or len(page_indexes) < max_images):
                        page_indexes.append(int(source.other_info["page_idx"]))
            # check that all pages can be found
            file_paths = []
            for page_idx in page_indexes:
                file_path = os.path.join(self.images_dir, f"{document.source_identifier}_p{page_idx}.jpg")
                if os.path.exists(file_path):
                    file_paths.append(file_path)
                else:
                    raise Exception(f"file not found, expected location: {file_path}")
            return from_file_paths(file_paths, max_image_size)
        else:
            raise Exception("unsupported document type, images can only be created from PDFs. To create a PDF from a HTML file, use a tool like pdfkit, then run pdfkit.from_file('your_file') and use the output file instead of the HTML.")