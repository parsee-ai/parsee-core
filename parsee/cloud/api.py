import os

import requests

from parsee.templates.job_template import JobTemplate
from parsee.templates.job_from_json import from_json_dict


class ParseeCloud:

    def __init__(self, api_key: str):

        self.api_key = api_key
        self.host = os.getenv("BACKEND_HOST") if os.getenv("BACKEND_HOST") is not None else "https://backend.parsee.ai"

    def get_template(self, template_id: str) -> JobTemplate:
        url = f"{self.host}/api/extraction/template/id/{template_id}"
        template_request = requests.get(url, headers={"Authorization": self.api_key})
        if template_request.content == b'':
            raise Exception("template not found or not accessible")
        return from_json_dict(template_request.json())

    def save_template(self, template: JobTemplate, public: bool = False) -> str:
        url = f"{self.host}/api/extraction/template"
        template_json_dict = template.to_json_dict()
        template_json_dict = {**template_json_dict, "public": public}
        request = requests.post(url, json=template_json_dict, headers={"Authorization": self.api_key})
        return request.text
