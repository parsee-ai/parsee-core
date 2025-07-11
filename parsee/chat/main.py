from typing import *
from parsee.chat.custom_dataclasses import Message
from decimal import Decimal
from parsee.storage.interfaces import DocumentManager
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.model_loader import get_llm_base_model
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.utils.helper import merge_answer_pieces
from parsee.settings import chat_settings
from tenacity import RetryError
import logging

logger = logging.getLogger(__name__)


def run_chat_with_fallback(message: Message, message_history: List[Message],
             document_manager: DocumentManager, receivers: List[MlModelSpecification],
             most_recent_references_only: bool, show_chunk_index: bool = False,
             single_page_processing_max_images_trigger: Optional[int] = None) -> List[Message]:
    """Runs a chat with a fallback to other models if the first one fails to process the message."""
    logger.info(f"Running chat with fallback")
    logger.debug(f"Message: {message}")
    for spec in receivers:
        try:
            output = run_chat(message, message_history, document_manager, spec,
                     most_recent_references_only, show_chunk_index, single_page_processing_max_images_trigger)
        except RetryError:
            logger.warning(f"RetryError occurred for model {spec.model_id}. Continuing with next model.")
            continue
        logger.debug(f"Output from the model: {output}")
        return output
    logger.warning(f"No model was able to process the message")
    return []


def run_chat(message: Message, message_history: List[Message],
             document_manager: DocumentManager, spec: MlModelSpecification,
             most_recent_references_only: bool, show_chunk_index: bool = False,
             single_page_processing_max_images_trigger: Optional[int] = None) -> List[Message]:
    """Run a chat with a specific model."""
    logger.info(f"Running chat with {spec.model_id}")
    output = []

    model = get_llm_base_model(spec)

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

    data = document_manager.load_documents(references, model.spec.multimodal, str(message), model.spec.max_images, chat_settings.min_tokens_for_instructions_and_history, show_chunk_index)
    # for multimodal queries, check if pages have to be processed individually
    process_pages_individually = False
    if type(data) is list:
        # check if pages can be processed one by one
        if single_page_processing_max_images_trigger is not None and len(data) >= single_page_processing_max_images_trigger:
            process_pages_individually = True
    if process_pages_individually:
        answers = []
        cost = Decimal(0)
        for k, img in enumerate(data):
            if k > 0:
                additional_info = f"We are showing you the images contained in the document one by one. The current image is number {k + 1} out of a total of {len(data)}.\n Your last answer ended with the following (make sure that your new answer is valid JSON or similar, as requested; last 500 characters are shown):\n" \
                                  f"{answers[-1][-500:]}"
            else:
                additional_info = f"We are showing you the images contained in the document one by one. The current image is number {k + 1} out of a total of {len(data)}."
            prompt = Prompt(None, f"{message}", additional_info=additional_info, available_data=[img], history=[str(m) for m in message_history])
            current_answer, current_cost = model.make_prompt_request(prompt)
            answers.append(current_answer)
            cost += current_cost
        answer = merge_answer_pieces(answers)
    else:
        prompt = Prompt(None, f"{message}", available_data=data, history=[str(m) for m in message_history])
        answer, cost = model.make_prompt_request(prompt)

    output.append(Message(answer, [], model.spec.model_id, cost=cost))
    cache_info = model.make_prompt_request.cache_info()
    logger.info(f"Chat with {spec.model_id} done. Cache info: {cache_info}")
    return output
