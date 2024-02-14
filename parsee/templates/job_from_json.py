import json
from typing import Dict
from pydantic.tools import parse_obj_as

from parsee.templates.job_template import JobTemplate


def from_json(json_string: str) -> JobTemplate:

    data = json.loads(json_string)

    job = parse_obj_as(JobTemplate, data)

    return job


def from_json_dict(data: Dict) -> JobTemplate:

    job = parse_obj_as(JobTemplate, data)

    return job
