from dataclasses import dataclass
from typing import *
from parsee.chat.custom_dataclasses import Message, SingleImageProcessingSettings
from decimal import Decimal

from parsee.extraction.extractor_dataclasses import Base64Image
from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel
from parsee.storage.interfaces import DocumentManager, Modality, DocumentContent
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.model_loader import get_llm_base_model
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.utils.helper import merge_answer_pieces
from parsee.settings import chat_settings
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
import logging

logger = logging.getLogger(__name__)


@dataclass
class RunSpecification:
    """Configuration for a single chat run.

    Attributes:
        modality: Document content representation to load for the prompt.
        max_load_images: Maximum number of referenced document images to load,
            or `None` when no explicit image cap is needed.
        single_image_processing: Settings for pagewise image prompting, or
            `None` to process all images in one prompt.
        most_recent_references_only: Whether to use only references attached to
            the current message.
        show_chunk_index: Whether loaded text should include chunk indexes.
    """

    modality: Modality
    max_load_images: int | None
    single_image_processing: SingleImageProcessingSettings | None
    most_recent_references_only: bool
    show_chunk_index: bool


class ReceiverLoopRetryError(Exception):
    """Raised when every receiver failed and the fallback loop should retry."""


@retry(
    stop=stop_after_attempt(chat_settings.receiver_loop_retry_attempts),
    retry=retry_if_exception_type(ReceiverLoopRetryError),
    wait=wait_random_exponential(
        multiplier=chat_settings.receiver_loop_retry_wait_multiplier,
        min=chat_settings.receiver_loop_retry_wait_min,
        max=chat_settings.receiver_loop_retry_wait_max
    ),
    reraise=True,
)
def _run_receiver_loop(message: Message, message_history: List[Message],
                       document_manager: DocumentManager, receivers: List[MlModelSpecification], run_spec: RunSpecification) -> Message:
    """Run chat against receivers until one succeeds.

    Args:
        message: Current user message to process.
        message_history: Previous chat messages, ordered by the caller.
        document_manager: Document manager used to load referenced content.
        receivers: Ordered model specifications to try.
        run_spec: Configuration controlling document loading and prompting.

    Returns:
        The first successful model response message.

    Raises:
        ReceiverLoopRetryError: If all receivers fail for an attempt, causing
            the tenacity retry wrapper to retry the full receiver loop.
    """
    for spec in receivers:
        try:
            output = run_chat(message, message_history, document_manager, spec, run_spec)
        except RetryError:
            logger.warning(f"RetryError occurred for model {spec.model_id}. Continuing with next model.")
            continue
        logger.debug(f"Output from the model: {output}")
        return output
    raise ReceiverLoopRetryError()


def run_chat_with_fallback(message: Message, message_history: List[Message],
                           document_manager: DocumentManager, receivers: List[MlModelSpecification],
                           run_spec: RunSpecification) -> Message | None:
    """Run chat with fallback to later models when earlier models fail.

    Args:
        message: Current user message to process.
        message_history: Previous chat messages, ordered by the caller.
        document_manager: Document manager used to load referenced content.
        receivers: Ordered model specifications to try.
        run_spec: Configuration controlling document loading and prompting.

    Returns:
        The successful model response, or `None` when no receiver is configured
        or all receivers fail.
    """
    logger.info(f"Running chat with fallback")
    logger.debug(f"Message: {message}")

    if len(receivers) == 0:
        logger.warning("No fallback models configured")
        return None

    try:
        return _run_receiver_loop(message, message_history, document_manager, receivers,
                                  run_spec)
    except ReceiverLoopRetryError:
        logger.warning(f"No model was able to process the message")
        return None


def prompt_pagewise(images: list[Base64Image], message: Message, message_history: list[Message],
                    model: LLMBaseModel, single_image_processing: SingleImageProcessingSettings, text: str | None = None) -> tuple[str, Decimal]:
    """Prompt a model with images one at a time and merge the answers.

    Args:
        images: Images to send to the model in separate prompt requests.
        message: Current user message to process.
        message_history: Previous chat messages to include as prompt history.
        model: Loaded LLM implementation used to make prompt requests.
        single_image_processing: Settings that control answer merging and the
            threshold for pagewise processing.
        text: Optional document text to include with each image prompt.

    Returns:
        A tuple containing the merged answer text and the accumulated request
        cost.
    """
    answers = []
    cost = Decimal(0)
    for k, img in enumerate(images):
        if k > 0:
            additional_info = f"We are showing you the images contained in the document one by one. The current image is number {k + 1} out of a total of {len(images)}.\n Your last answer ended with the following (make sure that your new answer is valid JSON or similar, as requested; last 500 characters are shown):\n" \
                              f"{answers[-1][-500:]}"
        else:
            additional_info = f"We are showing you the images contained in the document one by one. The current image is number {k + 1} out of a total of {len(images)}."
        prompt = Prompt(None, str(message), additional_info=additional_info, images=[img], text=text,
                        history=[str(m) for m in message_history])
        current_answer, current_cost = model.make_prompt_request(prompt)
        answers.append(current_answer)
        cost += current_cost

    # Use custom merge strategy if provided, otherwise use default
    if single_image_processing.merge_strategy is not None:
        answer = single_image_processing.merge_strategy(answers)
    else:
        answer = merge_answer_pieces(answers)
    return answer, cost


def run_chat(message: Message, message_history: List[Message],
             document_manager: DocumentManager, spec: MlModelSpecification, run_spec: RunSpecification) -> Message:
    """Run chat with a specific model.

    Args:
        message: Current user message to process.
        message_history: Previous chat messages, ordered by the caller.
        document_manager: Document manager used to load referenced content.
        spec: Model specification to load and prompt.
        run_spec: Configuration controlling document loading and prompting.

    Returns:
        Message containing the model answer, model identifier, and accumulated
        cost.

    Raises:
        ValueError: If image-processing settings are provided for text-only
            mode, if an image limit is provided for text-only mode, or if image
            content is missing for a modality that requires images.
    """
    if run_spec.modality == Modality.TEXT and run_spec.single_image_processing is not None:
        raise ValueError("Run specification is wrongly set up: single_image_processing should be None if modality is TEXT.")
    if run_spec.modality == Modality.TEXT and run_spec.max_load_images is not None:
        raise ValueError(
            "Run specification is wrongly set up: max_load_images should be None if modality is TEXT.")

    logger.info(f"Running chat with {spec.model_id}")
    model = get_llm_base_model(spec)

    # collect all references if requested
    references = message.references if run_spec.most_recent_references_only else []
    if not run_spec.most_recent_references_only:
        added_references = set()
        all_messages = [message] + message_history
        for m in all_messages:
            new_references = [x for x in m.references if x.reference_id() not in added_references]
            for ref in new_references:
                references.append(ref)
                added_references.add(ref.reference_id())

    document_content = document_manager.load_documents(references,
                                           run_spec.modality,
                                           str(message),
                                           run_spec.max_load_images,
                                           run_spec.show_chunk_index)

    match run_spec.modality:
        case Modality.IMAGES:
            if document_content.images is None:
                raise ValueError("Images are not specified")
            if run_spec.single_image_processing is not None and len(document_content.images) >= run_spec.single_image_processing.min_images_trigger:
                answer, cost = prompt_pagewise(document_content.images, message, message_history, model, run_spec.single_image_processing)
            else:
                prompt = Prompt(None, f"{message}", images=document_content.images, text=None, history=[str(m) for m in message_history])
                answer, cost = model.make_prompt_request(prompt)
        case Modality.TEXT:
            prompt = Prompt(None, f"{message}", images=[], text=document_content.text,
                            history=[str(m) for m in message_history])
            answer, cost = model.make_prompt_request(prompt)
        case Modality.IMAGES_AND_TEXT:
            if document_content.images is None:
                raise ValueError("Images are not specified")
            if run_spec.single_image_processing is not None and len(document_content.images) >= run_spec.single_image_processing.min_images_trigger:
                answer, cost = prompt_pagewise(document_content.images, message, message_history, model, run_spec.single_image_processing, text=document_content.text)
            else:
                prompt = Prompt(None, f"{message}", images=document_content.images, text=document_content.text, history=[str(m) for m in message_history])
                answer, cost = model.make_prompt_request(prompt)

    output_message = Message(answer, [], model.spec.model_id, cost=cost)
    cache_info = model.make_prompt_request.cache_info()
    logger.info(f"Chat with {spec.model_id} done. Cache info: {cache_info}")
    return output_message
