import os
from typing import *

import requests
from enum import Enum

from parsee.templates.job_template import JobTemplate
from parsee.templates.template_from_json import from_json_dict
from parsee.extraction.extractor_elements import StandardDocumentFormat
from parsee.converters.json_to_raw import load_document_from_json
from parsee.extraction.extractor_dataclasses import ParseeAnswer, ParseeMeta, source_from_json, AssignedAnswer
from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.converters.image_creation import from_bytes


class RequestType(Enum):
    POST = "post"
    GET = "get"


class ParseeCloud:

    def __init__(self, api_key: Optional[str], custom_host: Optional[str] = None):

        self.api_key = api_key
        self.host = custom_host if custom_host is not None else (os.getenv("BACKEND_HOST") if os.getenv("BACKEND_HOST") is not None else "https://backend.parsee.ai")
        self.default_image_size = 2000

    def _headers(self):
        return {"Authorization": self.api_key}

    def _make_request(self, url: str, request_type: RequestType, data: Optional[Dict] = None, max_retries: int=3, retry=0) -> any:
        try:
            if request_type == RequestType.POST:
                return requests.post(url, json=data, headers=self._headers())
            else:
                return requests.get(url, headers=self._headers())
        except Exception as e:
            if retry < max_retries:
                return self._make_request(url, request_type, data, max_retries, retry+1)
            else:
                return None

    def get_template(self, template_id: str) -> JobTemplate:
        url = f"{self.host}/{'public/' if self.api_key is None else ''}api/extraction/template/id/{template_id}"
        template_request = self._make_request(url, RequestType.GET)
        if template_request is None or template_request.content == b'':
            raise Exception("template not found or not accessible")
        return from_json_dict(template_request.json())

    def get_document(self, source_identifier: str) -> StandardDocumentFormat:
        url = f"{self.host}/api/document/get-json-document?identifier={source_identifier}"
        data = requests.get(url, headers=self._headers()).json()
        return load_document_from_json(data)

    def save_template(self, template: JobTemplate, public: bool = False) -> str:
        url = f"{self.host}/api/extraction/template"
        template_json_dict = template.to_json_dict()
        template_json_dict = {**template_json_dict, "public": public}
        request = requests.post(url, json=template_json_dict, headers=self._headers())
        return request.text

    # currently only output for general questions is supported here (i.e. textual output)
    def get_output(self, source_identifier: str, template_id: str) -> List[ParseeAnswer]:
        url = f"{self.host}/api/extraction/output/json/{source_identifier}"
        data = requests.get(url, headers=self._headers()).json()
        output = []
        for entry in data:
            if entry["templateId"] == template_id and entry["text"] is not None:
                class_id = entry["detectedClass"]
                class_value = entry["text"]["answerParsed"]
                meta_answers = []
                for item in entry["text"]["metaInfo"]:
                    meta_answers.append(ParseeMeta(item["model"], 0, [], item["class_id"], item["value"], item["prob"]))
                sources = [source_from_json(x) for x in entry["sources"]]
                output.append(ParseeAnswer(entry["model"], sources, class_id, class_value, "", True, meta_answers))
        return output

    def upload_file(self, file_path: str) -> str:
        """
        upload a PDF, HTML or image file to Parsee Cloud
        """
        with open(file_path, "rb") as f:
            data = f.read()
        file_data = {'file': (os.path.basename(file_path), data)}

        extractor_url = f"{self.host}/api/document/upload?method=simple"

        r = requests.post(extractor_url, {}, files=file_data, headers=self._headers())

        data = r.json()

        return list(data[0].values())[0]

    def add_assigned_answers(self, template_id: str, source_identifier: str, answers: List[AssignedAnswer]) -> bool:
        """
        adds one or more assigned answers to the Parsee Cloud database
        """
        failed = False
        for answer in answers:
            data = {
                "sourceIdentifier": source_identifier,
                "templateId": template_id,
                "itemId": answer.class_id,
                "newValue": answer.class_value,
                "newMeta": [
                    {"model": "manual", "class_id": meta_item.class_id, "value": meta_item.class_value, "prob": 1.0} for meta_item in answer.meta
                ],
                "sources": [source.to_json_dict() for source in answer.sources]
            }

            url = f"{self.host}/api/extraction/output/general-query"

            r = self._make_request(url, RequestType.POST, data)

            if r is None or r.status_code != 200:
                failed = True
        return not failed

    def _get_image(self, source_identifier: str, page_index: int) -> bytes:

        url = f"{self.host}/api/document/images?identifier={source_identifier}&page-index={page_index}"

        return requests.get(url, headers=self._headers()).content

    def get_image(self, source_identifier: str, page_index: int, max_image_size: Optional[int]) -> Base64Image:

        data = self._get_image(source_identifier, page_index)

        return from_bytes(data, max_image_size if max_image_size is not None else self.default_image_size)

    def get_image_and_save(self, source_identifier: str, page_index: int, output_path: str):
        """
        :param output_path: Output image is always a JPEG.
        """

        with open(output_path, "wb") as f:
            f.write(self._get_image(source_identifier, page_index))