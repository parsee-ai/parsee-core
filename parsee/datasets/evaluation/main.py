from typing import *

from parsee.extraction.extractor_dataclasses import ParseeAnswer
from parsee.datasets.readers.interfaces import DatasetReader
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.templates.job_template import JobTemplate
from parsee.extraction.models.model_loader import ModelLoader, LLMQuestionModel
from parsee.storage.interfaces import StorageManager
from parsee.storage.in_memory_storage import InMemoryStorageManager


class EvaluationResult:

    def __init__(self):
        self.answers = {}

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
                obj[answer.class_id][answer.unique_id()] = answer

    def evaluate(self) -> Dict:
        scores_by_source = {}
        for source, sc_dict in self.answers.items():
            scores_by_source[source] = {}
            reference_item = sc_dict["assigned"]
            for model, by_class in sc_dict.items():
                scores_by_source[source][model] = {"error_log": []}
                # check that all questions have been answered
                scores_by_source[source][model]["completion"] = len(by_class.keys()) / len(reference_item.keys())
                # check each question
                score = 0
                score_meta = 0
                for class_id, assigned_values in reference_item.items():
                    if class_id not in by_class:
                        continue
                    predicted_values = by_class[class_id]
                    for meta_key, correct_value in assigned_values.items():
                        if meta_key in predicted_values:
                            if meta_key != "":
                                score_meta += 1
                            predicted_value = predicted_values[meta_key]
                            # check if values match
                            if correct_value.class_value == predicted_value.class_value:
                                score += 1
                            else:
                                scores_by_source[source][model]["error_log"].append({"doc": source, "class_id": class_id, "type": "main question", "expected": correct_value.class_value, "actual": predicted_value.class_value})
                        elif len(assigned_values.keys()) == 1 and len(predicted_values.keys()) == 1:
                            predicted_value = predicted_values[list(predicted_values.keys())[0]]
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
                    total_scores[model] = {"completion": 0, "total_correct": 0, "total_correct_meta_found": 0, "error_log": []}
                for score_key, score_value in scores.items():
                    total_scores[model][score_key] += score_value
        reference_scores = total_scores["assigned"]
        for model, scores_dict in total_scores.items():
            scores_dict_final = {**scores_dict}
            for score_key, score_value in scores_dict.items():
                if score_key == "completion":
                    scores_dict_final[score_key] = scores_dict[score_key] / len(scores_by_source.keys())
                elif score_key == "error_log":
                    pass
                else:
                    rel_key = score_key+"_percent"
                    scores_dict_final[rel_key] = score_value / reference_scores[score_key]
            total_scores[model] = scores_dict_final
        return total_scores


def evaluate_llm_performance(template: JobTemplate, reader: DatasetReader, models: List[MlModelSpecification], storage: Optional[StorageManager] = None) -> Dict:

    storage = InMemoryStorageManager(models) if storage is None else storage
    loader = ModelLoader(storage)
    ev = EvaluationResult()

    for spec in models:
        if spec.model_type == "custom":
            raise Exception("custom models not allowed here")

    for row, _ in reader.row_generator():

        for k, model_spec in enumerate(models):
            model: LLMQuestionModel = loader.get_question_model(model_spec.internal_name, template.questions.items, template.meta, {})
            if model is None:
                raise Exception("model not found")
            schema_items = [x for x in template.questions.items if x.id == row.element_identifier]
            if len(schema_items) == 0:
                raise Exception("item not found in schema")
            schema_item = schema_items[0]
            answers_model = model.predict_for_prompt(row.get_feature("full_prompt"), schema_item, None, None)
            ev.add_answers(row.source_identifier, answers_model, False)
            if k == 0:
                answers_assigned = model.parse_prompt_answer(schema_item, row.get_value("answer", False), None, None)
                ev.add_answers(row.source_identifier, answers_assigned, True)

    return ev.evaluate()
