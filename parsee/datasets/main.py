from decimal import Decimal
from functools import reduce
from typing import *

import tiktoken

from parsee.extraction.extractor_elements import StandardDocumentFormat
from parsee.templates.job_template import JobTemplate
from parsee.storage.interfaces import StorageManager
from parsee.storage.in_memory_storage import InMemoryStorageManager
from parsee.extraction.models.model_loader import question_models_from_schema, ModelLoader
from parsee.extraction.tasks.questions.features import GeneralQueriesPromptBuilder
from parsee.extraction.models.llm_models.llm_base_model import truncate_prompt
from parsee.datasets.dataset_dataclasses import DatasetRow
from parsee.extraction.extractor_dataclasses import AssignedAnswer


def create_dataset_rows(template: JobTemplate, document: StandardDocumentFormat, assigned_answers: List[AssignedAnswer], storage: Optional[StorageManager] = None, max_tokens_prompt=4000, custom_model_loader: Optional[ModelLoader] = None) -> List[DatasetRow]:
    encoding = tiktoken.get_encoding("cl100k_base")
    storage = InMemoryStorageManager(None) if storage is None else storage
    model_loader = ModelLoader(storage) if custom_model_loader is None else custom_model_loader
    template = storage.db_values_template(template, False)
    question_feature_builder = GeneralQueriesPromptBuilder(storage)
    question_models = question_models_from_schema(template.questions, template.meta, model_loader, {"truth_questions": assigned_answers})
    question_rows = []
    if len(question_models) > 0:
        question_model = question_models[0]
        answers = question_model.predict_answers(document)
        if len(answers) > 0:
            for item in question_model.items:
                meta_items_filtered = [x for x in template.meta if x.id in item.metaInfoIds]
                prompt = question_feature_builder.build_prompt(item, meta_items_filtered, document)
                real_prompt, _ = truncate_prompt(str(prompt), encoding, max_tokens_prompt)
                answers_filtered = [x for x in answers if x.class_id == item.id]
                if len(answers_filtered) > 0:
                    # join together in case the answer has multiple blocks
                    full_answer = "\n\n".join([question_feature_builder.build_raw_value(answer.class_value, answer.meta, answer.sources, item, meta_items_filtered) for answer in answers_filtered])
                    row = DatasetRow(document.source_identifier, template.id, item.id, {"full_prompt": real_prompt})
                    row.assign_truth_values({"answer": full_answer})
                    question_rows.append(row)
    return question_rows
