from typing import *
from functools import reduce

from parsee.templates.job_template import JobTemplate
from parsee.templates.helpers import create_template, StructuringItem, OutputType
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.storage.interfaces import StorageManager
from parsee.extraction.extractor_elements import StandardDocumentFormat, FinalOutputTableColumn
from parsee.extraction.extractor_dataclasses import ParseeAnswer, ParseeBucket
from parsee.storage.in_memory_storage import InMemoryStorageManager
from parsee.converters.image_creation import ImageCreator
from parsee.extraction.final_structuring import get_structured_tables_from_locations, final_tables_from_columns
from parsee.extraction.models.model_loader import element_models_from_schema, meta_models_from_items, question_models_from_schema, mapping_models_from_schema, ModelLoader


def _model_loader(models: List[MlModelSpecification], custom_model_loader: Optional[ModelLoader] = None, custom_image_creator: Optional[ImageCreator] = None) -> ModelLoader:
    model_loader = custom_model_loader
    if model_loader is None:
        storage = InMemoryStorageManager(models, custom_image_creator)
        model_loader = ModelLoader(storage)
    return model_loader


def run_custom_prompt(doc: StandardDocumentFormat, prompt: str, model: MlModelSpecification, custom_model_loader: Optional[ModelLoader] = None, custom_image_creator: Optional[ImageCreator] = None) -> str:
    """
    Runs a custom prompt and returns the unmodified response from the model.
    Will insert data (either text or images, depending on model type selcted) from your document the same way the extraction via templates works.
    """
    template = create_template([StructuringItem(prompt, OutputType.TEXT)])
    # update the models
    template.set_default_model(model)
    _, _, answers_text = structure_data(doc, template, _model_loader([model], custom_model_loader, custom_image_creator), {})
    if len(answers_text) < 1:
        return ""
    return answers_text[0].raw_value


def run_job_with_single_model(doc: StandardDocumentFormat, job_template: JobTemplate, model: MlModelSpecification, custom_model_loader: Optional[ModelLoader] = None, custom_image_creator: Optional[ImageCreator] = None) -> Tuple[List[ParseeBucket], List[FinalOutputTableColumn], List[ParseeAnswer]]:
    model_loader = _model_loader([model], custom_model_loader, custom_image_creator)
    # update the models
    job_template.set_default_model(model)
    return structure_data(doc, job_template, model_loader, {})


def structure_data(doc: StandardDocumentFormat, job_template: JobTemplate, model_loader: ModelLoader, params: Dict[str, Any]) -> Tuple[List[ParseeBucket], List[FinalOutputTableColumn], List[ParseeAnswer]]:

    # add manual answers to params
    params = {**params, **job_template.detection.settings, **job_template.questions.settings}

    question_models = question_models_from_schema(job_template.questions, job_template.meta, model_loader, params)

    answers: List[ParseeAnswer] = []
    for question_model in question_models:
        answers += question_model.predict_answers(doc)

    models_loc = element_models_from_schema(job_template.detection, model_loader, params)

    locations = []
    for model_loc in models_loc:
        locations += model_loc.classify_elements(doc)

    output_values = get_structured_tables_from_locations(job_template, doc, locations)

    # add meta values
    all_meta_ids = list(set(reduce(lambda acc, x: acc+x.metaInfoIds, job_template.detection.items, [])))
    meta_ids_by_main_class = reduce(lambda acc, x: {**acc, x.id: x.metaInfoIds}, job_template.detection.items, {})
    if len(all_meta_ids) > 0:
        meta_models = meta_models_from_items([x for x in job_template.meta if x.id in all_meta_ids], model_loader, params)
        for meta_model in meta_models:
            meta_predictions_list = meta_model.predict_meta(output_values, doc.elements)
            for k, meta_predictions in enumerate(meta_predictions_list):
                output_values[k].meta += [x for x in meta_predictions if x.class_id in meta_ids_by_main_class[output_values[k].detected_class]]

    # run mapping
    all_mappings: List[ParseeBucket] = []
    tables = final_tables_from_columns(output_values)
    mapping_models = mapping_models_from_schema(job_template.detection, model_loader, params)
    for model_mapping in mapping_models:
        for table in tables:
            mappings, mapping_schema = model_mapping.classify_elements(table)
            all_mappings += mappings

    return all_mappings, output_values, answers