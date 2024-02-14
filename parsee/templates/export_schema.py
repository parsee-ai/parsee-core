from pydantic.dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class WebhookSchema:
    url: str
    headers: Dict
    sendData: bool


@dataclass
class ExportSchema:
    webhook: Optional[WebhookSchema] = None
