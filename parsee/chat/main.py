from typing import *
from parsee.chat.custom_dataclasses import Message, ChatSettings, Author, Role
from parsee.storage.interfaces import DocumentManager
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.model_loader import get_llm_base_model
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.utils.enums import SearchStrategy


def run_chat(message: Message, message_history: List[Message], document_manager: DocumentManager, receivers: List[MlModelSpecification]) -> List[Message]:

    output = []

    models = [get_llm_base_model(spec) for spec in receivers]

    for model in models:
        prompt = Prompt(main_task=f"{message}", description="", available_data=document_manager.load_documents(message.references, model.spec.multimodal, str(message)), history=[str(m) for m in message_history])
        answer, cost = model.make_prompt_request(prompt)
        output.append(Message(answer, [], Author(model.spec.model_id, Role.AGENT, None), cost))
