import os.path
from typing import *

from parsee.extraction.extractor_dataclasses import ParseeAnswer
from parsee.datasets.readers.interfaces import DatasetReader
from parsee.datasets.writers.interfaces import DatasetWriter
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.templates.job_template import JobTemplate
from parsee.extraction.models.model_loader import ModelLoader, LLMQuestionModel
from parsee.storage.interfaces import StorageManager
from parsee.storage.in_memory_storage import InMemoryStorageManager
from parsee.extraction.models.llm_models.prompts import Prompt
from parsee.extraction.extractor_elements import StandardDocumentFormat, ExtractedSource
from parsee.utils.enums import DocumentType


class EvaluationResult:

    def __init__(self, custom_compare_func: Optional[Dict[str, Callable]], exclude_meta_keys: Optional[List[str]]):
        self.answers = {}
        self.custom_compare_func = custom_compare_func
        self.exclude_meta_keys = exclude_meta_keys

    def add_answers(self, source_identifier: str, answers: List[ParseeAnswer], is_assigned: bool):
        if len(answers) == 0:
            return
        if source_identifier not in self.answers:
            self.answers[source_identifier] = {}
        key = "assigned" if is_assigned else answers[0].model
        if key not in self.answers[source_identifier]:
            self.answers[source_identifier][key] = {}
        obj = self.answers[source_identifier][key]
        for answer in answers:
            if answer.class_id not in obj:
                obj[answer.class_id] = {}
            if answer.unique_id() not in obj[answer.class_id]:
                obj[answer.class_id][answer.unique_id(self.exclude_meta_keys)] = answer

    def _values_match(self, val1: ParseeAnswer, val2: ParseeAnswer) -> bool:
        if val1.class_id in self.custom_compare_func:
            return self.custom_compare_func[val1.class_id](val1, val2)
        return val1.class_value == val2.class_value

    def evaluate(self) -> Dict:
        scores_by_source = {}
        for source, sc_dict in self.answers.items():
            scores_by_source[source] = {}
            reference_item = sc_dict["assigned"]
            for model, by_class in sc_dict.items():
                scores_by_source[source][model] = {"error_log": [], "missing_answers": 0}
                # check that all questions have been answered
                scores_by_source[source][model]["completion"] = len(by_class.keys()) / len(reference_item.keys())
                # check each question
                score = 0
                score_meta = 0
                for class_id, assigned_values in reference_item.items():
                    if class_id not in by_class:
                        continue
                    predicted_values = by_class[class_id]
                    # check if some answers are missing from the result
                    scores_by_source[source][model]["missing_answers"] = max(len(assigned_values.keys()) - len(predicted_values.keys()), 0)
                    for meta_key, correct_value in assigned_values.items():
                        if meta_key in predicted_values:
                            if meta_key != "" and len(correct_value.meta) > 0:
                                score_meta += 1
                            predicted_value = predicted_values[meta_key]
                            # check if values match
                            if self._values_match(correct_value, predicted_value):
                                score += 1
                            else:
                                scores_by_source[source][model]["error_log"].append({"doc": source, "class_id": class_id, "type": "main question", "expected": correct_value.class_value, "actual": predicted_value.class_value})
                        else:
                            if len(assigned_values.keys()) == 1 and len(predicted_values.keys()) == 1:
                                predicted_value = predicted_values[list(predicted_values.keys())[0]]
                            else:
                                # try to find main answer
                                filtered_answers = [x for x in predicted_values.values() if x.class_value == correct_value.class_value]
                                if len(filtered_answers) > 0:
                                    predicted_value = filtered_answers[0]
                                else:
                                    continue
                            # meta mismatch
                            scores_by_source[source][model]["error_log"].append({"doc": source, "class_id": class_id, "type": "meta", "expected": correct_value.meta_key(), "actual": predicted_value.meta_key()})
                            # check if values match
                            if correct_value.class_value == predicted_value.class_value:
                                score += 1
                            else:
                                scores_by_source[source][model]["error_log"].append({"doc": source, "class_id": class_id, "type": "main question", "expected": correct_value.class_value, "actual": predicted_value.class_value})

                scores_by_source[source][model]["total_correct"] = score
                scores_by_source[source][model]["total_correct_meta_found"] = score_meta

        total_scores = {}
        for source, by_source in scores_by_source.items():
            for model, scores in by_source.items():
                if model not in total_scores:
                    total_scores[model] = {"completion": 0, "total_correct": 0, "total_correct_meta_found": 0, "missing_answers": 0, "error_log": []}
                for score_key, score_value in scores.items():
                    total_scores[model][score_key] += score_value
        reference_scores = total_scores["assigned"]
        for model, scores_dict in total_scores.items():
            scores_dict_final = {**scores_dict}
            scores_dict_final["completion"] = scores_dict["completion"] / len(scores_by_source.keys())
            # calculate completeness: how many items are not "missing" entirely
            scores_dict_final["completeness"] = (reference_scores["total_correct"] - scores_dict_final["missing_answers"])/reference_scores["total_correct"]
            # scores INCLUDING missing answers
            scores_dict_final["total_correct_percent"] = scores_dict_final["total_correct"] / reference_scores["total_correct"]
            scores_dict_final["total_correct_meta_found_percent"] = (scores_dict_final["total_correct_meta_found"] / reference_scores["total_correct_meta_found"]) if reference_scores["total_correct_meta_found"] > 0 else None
            # scores EXCLUDING missing answers
            scores_dict_final["total_correct_percent_ex_missing"] = scores_dict_final["total_correct"] / (reference_scores["total_correct"]-scores_dict_final["missing_answers"])
            scores_dict_final["total_correct_meta_found_percent_ex_missing"] = (scores_dict_final["total_correct_meta_found"] / (reference_scores["total_correct_meta_found"]-scores_dict_final["missing_answers"])) if reference_scores["total_correct_meta_found"] > 0 else None
            total_scores[model] = scores_dict_final
        return total_scores


def evaluate_llm_performance(template: JobTemplate, reader: DatasetReader, models: List[MlModelSpecification], storage: Optional[StorageManager] = None, writer_for_model_answers: Optional[DatasetWriter] = None, use_saved_model_answers: bool = False, new_dataset_name: Optional[str] = None, custom_compare_func: Optional[Dict[str, Callable]] = None, exclude_meta_keys: Optional[List[str]] = None, retry_on_na: bool = False) -> Dict:

    storage = InMemoryStorageManager(models) if storage is None else storage
    loader = ModelLoader(storage)
    ev = EvaluationResult(custom_compare_func, exclude_meta_keys)

    if len(models) == 0:
        raise Exception("please provide some model specifications")

    for spec in models:
        if spec.model_type == "custom":
            raise Exception("custom models not allowed here")

    for row, _ in reader.row_generator():

        for k, model_spec in enumerate(models):
            model: LLMQuestionModel = loader.get_question_model(model_spec.model_id, template.questions.items, template.meta, {})
            if model is None:
                raise Exception("model not found")
            schema_items = [x for x in template.questions.items if x.id == row.element_identifier]
            if len(schema_items) == 0:
                raise Exception("item not found in schema")
            schema_item = schema_items[0]
            if use_saved_model_answers and row.get_value(model_spec.model_id, False) is not None and str(row.get_value(model_spec.model_id, False)).strip() != "" and not (retry_on_na and str(row.get_value(model_spec.model_id, False)) == "n/a"):
                prompt_answer = row.get_value(model_spec.model_id, False)
                answers_model = model.parse_prompt_answer(schema_item, prompt_answer, None, None)
            else:
                # for text based datasets, the data is already in the prompt
                available_data = None
                if model_spec.multimodal:
                    page_indexes = row.get_feature("page_indexes").split("|")
                    sources = [ExtractedSource(DocumentType.PDF, None, None, 0, {"page_idx": x}) for x in page_indexes]
                    available_data = storage.image_creator.get_images(StandardDocumentFormat(DocumentType.PDF, row.source_identifier, [], None), sources, len(page_indexes), model_spec.max_image_pixels)
                prompt = Prompt(None, row.get_feature("full_prompt"), None, None, available_data)
                answers_model = model.predict_for_prompt(prompt, schema_item, None, None)
                raw_answer = answers_model[0].raw_value if len(answers_model) > 0 else "n/a"
                row.assign_truth_values({model_spec.model_id: raw_answer})
            ev.add_answers(row.source_identifier, answers_model, False)
            if k == 0:
                answers_assigned = model.parse_prompt_answer(schema_item, row.get_value("assigned", False), None, None)
                ev.add_answers(row.source_identifier, answers_assigned, True)
        if writer_for_model_answers is not None:
            writer_for_model_answers.write_rows([row], "dataset_with_answers" if new_dataset_name is None else new_dataset_name)

    return ev.evaluate()
