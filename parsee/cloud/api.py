import os
from typing import *

import requests

from parsee.templates.job_template import JobTemplate
from parsee.templates.template_from_json import from_json_dict
from parsee.extraction.extractor_elements import StandardDocumentFormat
from parsee.converters.json_to_raw import load_document_from_json
from parsee.extraction.extractor_dataclasses import ParseeAnswer, ParseeMeta, source_from_json


class ParseeCloud:

    def __init__(self, api_key: str, custom_host: Optional[str] = None):

        self.api_key = api_key
        self.host = custom_host if custom_host is not None else (os.getenv("BACKEND_HOST") if os.getenv("BACKEND_HOST") is not None else "https://backend.parsee.ai")

    def _headers(self):
        return {"Authorization": self.api_key}

    def get_template(self, template_id: str) -> JobTemplate:
        url = f"{self.host}/api/extraction/template/id/{template_id}"
        template_request = requests.get(url, headers=self._headers())
        if template_request.content == b'':
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
