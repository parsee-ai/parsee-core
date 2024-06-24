from typing import *
from parsee.chat.custom_dataclasses import Message, ChatSettings, Author, Role
from parsee.storage.interfaces import DocumentManager
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.model_loader import get_llm_base_model
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.utils.enums import SearchStrategy


def run_chat(message: Message, message_history: List[Message], document_manager: DocumentManager, receivers: List[MlModelSpecification], most_recent_references_only: bool) -> List[Message]:

    output = []

    models = [get_llm_base_model(spec) for spec in receivers]

    # collect all references if requested
    references = message.references if most_recent_references_only else []
    if not most_recent_references_only:
        added_references = set()
        all_messages = [message] + message_history
        for m in all_messages:
            new_references = [x for x in m.references if x.reference_id() not in added_references]
            for ref in new_references:
                references.append(ref)
                added_references.add(ref.reference_id())

    for model in models:
        prompt = Prompt(None, f"{message}", available_data=document_manager.load_documents(references, model.spec.multimodal, str(message), model.spec.max_images, model.spec.max_tokens-document_manager.settings.min_tokens_for_instructions_and_history), history=[str(m) for m in message_history])
        answer, cost = model.make_prompt_request(prompt)
        output.append(Message(answer, [], Author(model.spec.model_id, Role.AGENT), cost=cost))

    return output
