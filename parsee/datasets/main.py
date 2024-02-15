from decimal import Decimal
from functools import reduce
from typing import *

import tiktoken

from parsee.datasets.writers.interfaces import DatasetWriter
from parsee.extraction.extractor_elements import StandardDocumentFormat
from parsee.extraction.extractor_dataclasses import ID_NOT_AVAILABLE
from parsee.templates.job_template import JobTemplate
from parsee.storage.interfaces import StorageManager
from parsee.storage.in_memory_storage import InMemoryStorageManager
from parsee.extraction.tasks.element_classification.element_classifier_llm import LLMLocationFeatureBuilder
from parsee.utils.enums import ElementType
from parsee.extraction.models.model_loader import element_classifiers_from_schema, meta_classifiers_from_items, mapping_classifiers_from_schema, question_classifiers_from_schema, ModelLoader
from parsee.extraction.tasks.meta_info_structuring.features import LLMMetaFeatureBuilder
from parsee.extraction.final_structuring import get_structured_tables_from_locations, final_tables_from_columns
from parsee.extraction.tasks.mappings.features import LLMMappingFeatureBuilder
from parsee.extraction.tasks.questions.features import GeneralQueriesPromptBuilder
from parsee.extraction.tasks.mappings.mapping_classifier import ParseeBucket
from parsee.extraction.models.llm_models.llm_base_model import truncate_prompt
from parsee.datasets.dataset_dataclasses import DatasetRow
from parsee.extraction.extractor_dataclasses import AssignedAnswer


PARTIAL_TABLE_APPEND = "_partial"


def create_dataset_entries(source_identifier: str, template: JobTemplate, document: StandardDocumentFormat, writer: DatasetWriter, storage: StorageManager):

    template = storage.db_values_template(template, False)
    storage.disable_db_updates = True

    # LOCATIONS

    indices = [v.source.element_index for k, v in enumerate(document.elements) if v.el_type == ElementType.TABLE]

    location_classifiers = element_classifiers_from_schema(template.detection, storage, template.detection.settings)
    if len(location_classifiers) > 0:
        location_classifier = location_classifiers[0]
        features = location_classifier.feature_builder.make_features(source_identifier, template.id, indices, document.elements)
        locations = location_classifier.classify_elements(document)

        for feature_entry in features:
            new_entry = {}
            for detection_item in template.detection.items:
                filtered = [x for x in locations if x.detected_class == detection_item.id and x.source.element_index == feature_entry.element_identifier]
                new_entry[detection_item.id] = 0 if len(filtered) == 0 else 1
                new_entry[detection_item.id+PARTIAL_TABLE_APPEND] = 0 if len(filtered) == 0 else round(filtered[0].partial_prob)
            feature_entry.assign_truth_values(new_entry)

        writer.write_rows(features, "locations")

        structured_table_cols = get_structured_tables_from_locations(template, document, locations)

        # META
        meta_features = []
        all_meta_ids = list(set(reduce(lambda acc, x: acc + x.metaInfoIds, template.detection.items, [])))
        meta_ids_by_main_class = reduce(lambda acc, x: {**acc, x.id: x.metaInfoIds}, template.detection.items, {})
        if len(all_meta_ids) > 0:
            meta_classifiers = meta_classifiers_from_items([x for x in template.meta if x.id in all_meta_ids], storage, template.detection.settings)
            for meta_classifier in meta_classifiers:
                values_predicted = meta_classifier.predict_meta(structured_table_cols, document.elements)
                for k, output_col in enumerate(structured_table_cols):
                    meta_values = [x for x in values_predicted[k] if x.class_id in meta_ids_by_main_class[output_col.detected_class]]
                    features = meta_classifier.feature_builder.make_features(source_identifier, template.id, output_col, document.elements, None, None)
                    dict_values = {x.class_id: x.class_value for x in meta_values}
                    features.assign_truth_values(dict_values)
                    meta_features.append(features)

            writer.write_rows(meta_features, "meta")
        
        structured_tables = final_tables_from_columns(structured_table_cols)

        # MAPPINGS
        mapping_features = []
        li_added = set()
        for output_table in structured_tables:
            detection_item = [x for x in template.detection.items if output_table.detected_class == x.id][0]
            if detection_item.mapRows is not None and output_table.li_identifier not in li_added:
                mapping_classifier = mapping_classifiers_from_schema(template.detection, storage, template.detection.settings)[0]
                mappings, schema = mapping_classifier.classify_elements(output_table)
                if schema is not None:
                    li_added.add(output_table.li_identifier)
                    for kv_index, _ in enumerate(output_table.line_items):
                        features = mapping_classifier.feature_builder.make_features(source_identifier, template.id, output_table, detection_item.mapRows.id, kv_index)
                        bucket_choice = [x for x in mappings if x.kv_index == kv_index][0] if len([x for x in mappings if x.kv_index == kv_index]) > 0 else ParseeBucket(schema.id, ID_NOT_AVAILABLE, output_table.li_identifier, "manual", 1, kv_index, 0, Decimal(1))
                        features.assign_truth_values({"bucket_id": bucket_choice.bucket_id, "definition_idx": bucket_choice.definition_idx, "multiplier": bucket_choice.multiplier})
                        mapping_features.append(features)
        writer.write_rows(mapping_features, "mapping")


def create_llm_dataset_entries(source_identifier: str, template: JobTemplate, document: StandardDocumentFormat, writer: DatasetWriter, storage: StorageManager, model_loader: ModelLoader, max_tokens_prompt=4000):

    template = storage.db_values_template(template, False)
    storage.disable_db_updates = True
    encoding = tiktoken.get_encoding("cl100k_base")

    # LOCATIONS
    location_prompt_builder = LLMLocationFeatureBuilder()
    location_classifiers = element_classifiers_from_schema(template.detection, model_loader, template.detection.settings)
    rows = []
    if len(location_classifiers) > 0:
        location_classifier = location_classifiers[0]
        locations = location_classifier.classify_elements(document)

        if len(locations) > 0:

            for detection_item in template.detection.items:
                filtered = [x for x in locations if x.detected_class == detection_item.id]

                prompt = location_prompt_builder.make_prompt(detection_item, document, storage)
                real_prompt = truncate_prompt(str(prompt), encoding, max_tokens_prompt)
                answer = location_prompt_builder.build_raw_answer(filtered)

                row = DatasetRow(source_identifier, template.id, detection_item.id, {"prompt": real_prompt})
                row.assign_truth_values({"answer": answer})

                rows.append(row)
            writer.write_rows(rows, "locations")

            structured_table_cols = get_structured_tables_from_locations(template, document, locations)

            # META
            meta_prompt_builder = LLMMetaFeatureBuilder()
            meta_rows = []
            all_meta_ids = list(set(reduce(lambda acc, x: acc + x.metaInfoIds, template.detection.items, [])))
            meta_ids_by_main_class = reduce(lambda acc, x: {**acc, x.id: x.metaInfoIds}, template.detection.items, {})
            if len(all_meta_ids) > 0:
                meta_classifiers = meta_classifiers_from_items([x for x in template.meta if x.id in all_meta_ids], model_loader, template.detection.settings)
                for meta_classifier in meta_classifiers:
                    meta_values_all = meta_classifier.predict_meta(structured_table_cols, document.elements)
                    for k, output_col in enumerate(structured_table_cols):
                        meta_items_filtered = [x for x in template.meta if x.id in meta_ids_by_main_class[output_col.detected_class]]
                        prompt = meta_prompt_builder.make_prompt(output_col, document.elements, meta_items_filtered)
                        real_prompt = truncate_prompt(str(prompt), encoding, max_tokens_prompt)
                        meta_values = meta_values_all[k]
                        row = DatasetRow(source_identifier, template.id, output_col.detected_class, {"prompt": real_prompt})
                        prompt_answer = meta_prompt_builder.build_raw_answer(meta_items_filtered, meta_values)
                        row.assign_truth_values({"answer": prompt_answer})
                        meta_rows.append(row)

            writer.write_rows(meta_rows, "meta")

            structured_tables = final_tables_from_columns(structured_table_cols)

            # MAPPINGS
            mapping_prompt_builder = LLMMappingFeatureBuilder()
            mapping_rows = []
            li_added = set()
            for output_table in structured_tables:
                detection_item = [x for x in template.detection.items if output_table.detected_class == x.id][0]
                if detection_item.mapRows is not None and output_table.li_identifier not in li_added:
                    mapping_classifier = mapping_classifiers_from_schema(template.detection, model_loader, template.detection.settings)[0]
                    mappings, schema = mapping_classifier.classify_elements(output_table)
                    if schema is not None:
                        li_added.add(output_table.li_identifier)
                        for kv_index, _ in enumerate(output_table.line_items):
                            prompt = mapping_prompt_builder.make_prompt(output_table, schema, kv_index)
                            real_prompt = truncate_prompt(str(prompt), encoding, max_tokens_prompt)
                            bucket_choice = [x for x in mappings if x.kv_index == kv_index][0] if len([x for x in mappings if x.kv_index == kv_index]) > 0 else ParseeBucket(schema.id, ID_NOT_AVAILABLE, output_table.li_identifier, "manual", 1, kv_index, 0, Decimal(1))
                            prompt_answer = mapping_prompt_builder.build_raw_answer(bucket_choice, schema)
                            row = DatasetRow(source_identifier, template.id, detection_item.id, {"prompt": real_prompt})
                            row.assign_truth_values({"answer": prompt_answer})
                            mapping_rows.append(row)

            writer.write_rows(mapping_rows, "mapping")

    # GENERAL QUERIES
    question_feature_builder = GeneralQueriesPromptBuilder(storage)
    question_classifiers = question_classifiers_from_schema(template.questions, template.meta, model_loader, {})
    if len(question_classifiers) > 0:
        question_rows = []
        question_classifier = question_classifiers[0]
        answers = question_classifier.predict_answers(document)
        if len(answers) > 0:
            for item in question_classifier.items:
                meta_items_filtered = [x for x in template.meta if x.id in item.metaInfoIds]
                prompt = question_feature_builder.build_prompt(item, meta_items_filtered, document)
                real_prompt = truncate_prompt(str(prompt), encoding, max_tokens_prompt)
                answers_filtered = [x for x in answers if x.class_id == item.id]
                if len(answers_filtered) > 0:
                    # join together in case the answer has multiple blocks
                    full_answer = "\n\n".join([answer.raw_value for answer in answers_filtered])
                    row = DatasetRow(source_identifier, template.id, item.id, {"full_prompt": real_prompt})
                    row.assign_truth_values({"answer": full_answer})
                    question_rows.append(row)
            writer.write_rows(question_rows, "questions")


def create_dataset_rows(template: JobTemplate, document: StandardDocumentFormat, assigned_answers: List[AssignedAnswer], storage: Optional[StorageManager] = None, max_tokens_prompt=4000, custom_model_loader: Optional[ModelLoader] = None) -> List[DatasetRow]:
    encoding = tiktoken.get_encoding("cl100k_base")
    storage = InMemoryStorageManager(None) if storage is None else storage
    model_loader = ModelLoader(storage) if custom_model_loader is None else custom_model_loader
    template = storage.db_values_template(template, False)
    question_feature_builder = GeneralQueriesPromptBuilder(storage)
    question_models = question_classifiers_from_schema(template.questions, template.meta, model_loader, {"truth_questions": assigned_answers})
    question_rows = []
    if len(question_models) > 0:
        question_model = question_models[0]
        answers = question_model.predict_answers(document)
        if len(answers) > 0:
            for item in question_model.items:
                meta_items_filtered = [x for x in template.meta if x.id in item.metaInfoIds]
                prompt = question_feature_builder.build_prompt(item, meta_items_filtered, document)
                real_prompt = truncate_prompt(str(prompt), encoding, max_tokens_prompt)
                answers_filtered = [x for x in answers if x.class_id == item.id]
                if len(answers_filtered) > 0:
                    # join together in case the answer has multiple blocks
                    full_answer = "\n\n".join([answer.raw_value for answer in answers_filtered])
                    row = DatasetRow(document.source_identifier, template.id, item.id, {"full_prompt": real_prompt})
                    row.assign_truth_values({"answer": full_answer})
                    question_rows.append(row)
    return question_rows
