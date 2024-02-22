from typing import *

from parsee.templates.element_schema import ElementDetectionSchema, ElementSchema
from parsee.templates.general_structuring_schema import StructuringItemSchema, GeneralQuerySchema, GeneralQueryItemSchema
from parsee.extraction.tasks.element_classification.element_classifier import ElementClassifier, AssignedElementClassifier
from parsee.extraction.tasks.questions.question_model import QuestionModel, AssignedQuestionModel
from parsee.extraction.models.llm_models.chatgpt_model import ChatGPTModel
from parsee.extraction.models.llm_models.replicate_model import ReplicateModel
from parsee.extraction.tasks.questions.question_model_llm import LLMQuestionModel
from parsee.extraction.tasks.meta_info_structuring.meta_info import MetaInfoClassifier
from parsee.extraction.tasks.meta_info_structuring.meta_info_llm import MetaLLMClassifier
from parsee.extraction.tasks.element_classification.element_classifier_llm import ElementClassifierLLM
from parsee.extraction.tasks.mappings.mapping_classifier import MappingClassifier
from parsee.extraction.tasks.mappings.mapping_classifier_llm import MappingClassifierLLM
from parsee.storage.interfaces import StorageManager
from parsee.utils.enums import ModelType
from parsee.extraction.models.model_dataclasses import MlModelSpecification
from parsee.extraction.models.llm_models.llm_base_model import LLMBaseModel


def get_llm_base_model(spec: MlModelSpecification) -> LLMBaseModel:
    if spec.model_type == ModelType.GPT:
        return ChatGPTModel(spec)
    elif spec.model_type == ModelType.REPLICATE:
        return ReplicateModel(spec)
    else:
        raise Exception("llm base model not found")


class ModelLoader:

    def __init__(self, storage: StorageManager):
        self.storage = storage

    def get_model_spec(self, model_name: Optional[str]) -> Union[MlModelSpecification, None]:
        filtered = [x for x in self.storage.get_available_models() if x.internal_name == model_name]
        if len(filtered) == 0:
            return None
        return filtered[0]

    def get_question_model(self, model_name: Optional[str], items: List[GeneralQueryItemSchema], all_meta_items: List[StructuringItemSchema], params: Dict[str, Any]) -> Union[QuestionModel, None]:
        if model_name is None:
            return None
        elif model_name == "assigned":
            return AssignedQuestionModel(items, all_meta_items, **params)
        else:
            return LLMQuestionModel(items, all_meta_items, self.storage, get_llm_base_model(self.get_model_spec(model_name)), **params)
    
    def get_element_model(self, model_name: Optional[str], items: List[ElementSchema], params: Dict[str, Any]) -> Union[ElementClassifier, None]:
        if model_name is None:
            return None
        elif model_name == "assigned":
            return AssignedElementClassifier(items, **params)
        else:
            return ElementClassifierLLM(items, self.storage, get_llm_base_model(self.get_model_spec(model_name)), **params)
    
    def get_meta_model(self, model_name: Optional[str], items: List[StructuringItemSchema], params: Dict[str, Any]) -> Union[MetaInfoClassifier, None]:
        if model_name is None:
            return None
        elif model_name == "assigned":
            # for assigned, no model is needed
            return None
        else:
            return MetaLLMClassifier(items, get_llm_base_model(self.get_model_spec(model_name)), self.storage, **params)
    
    def get_mapping_model(self, model_name: Optional[str], items: List[ElementSchema], params: Dict[str, Any]) -> Union[MappingClassifier, None]:
        if model_name is None:
            return None
        elif model_name == "assigned":
            return None # TODO
        else:
            return MappingClassifierLLM(items, self.storage, get_llm_base_model(self.get_model_spec(model_name)), **params)


def question_models_from_schema(schema: GeneralQuerySchema, all_meta_items: List[StructuringItemSchema], model_loader: ModelLoader, params: Dict[str, Any]) -> List[QuestionModel]:

    # group items by model
    items_by_model: Dict[str, List[GeneralQueryItemSchema]] = {}
    for item in schema.items:
        if item.model not in items_by_model:
            items_by_model[item.model] = []
        items_by_model[item.model].append(item)

    models: List[QuestionModel] = []
    for model_name, items in items_by_model.items():
        model = model_loader.get_question_model(model_name, items, all_meta_items, params)
        if model is None and model_name != "assigned":
            raise Exception(f"following model was not found: {model_name}")
        if model is not None:
            models.append(model)
    return models


def element_classifiers_from_schema(schema: ElementDetectionSchema, model_loader: ModelLoader, params: Dict[str, any]) -> List[ElementClassifier]:

    # group items by model
    items_by_classifier: Dict[str, List[ElementSchema]] = {}
    for item in schema.items:
        if item.model not in items_by_classifier:
            items_by_classifier[item.model] = []
        items_by_classifier[item.model].append(item)

    models: List[ElementClassifier] = []
    for model_name, items in items_by_classifier.items():
        model = model_loader.get_element_model(model_name, items, params)
        if model is None and model_name != "assigned":
            raise Exception(f"following model was not found: {model_name}")
        if model is not None:
            models.append(model)
    return models


def meta_models_from_items(meta_items: List[StructuringItemSchema], model_loader: ModelLoader, params: Dict[str, str]) -> List[MetaInfoClassifier]:

    # group items by model
    items_by_model: Dict[str, List[StructuringItemSchema]] = {}
    for item in meta_items:
        if item.model not in items_by_model:
            items_by_model[item.model] = []
        items_by_model[item.model].append(item)

    models: List[MetaInfoClassifier] = []
    for model_name, items in items_by_model.items():
        model = model_loader.get_meta_model(model_name, items, params)
        if model is None and model_name != "assigned":
            raise Exception(f"following model was not found: {model_name}")
        if model is not None:
            models.append(model)
    return models


def mapping_classifiers_from_schema(schema: ElementDetectionSchema, model_loader: ModelLoader, params: Dict[str, str]) -> List[MappingClassifier]:

    # group items by model
    items_by_classifier: Dict[str, List[ElementSchema]] = {}
    for item in schema.items:
        if item.mapRows is not None:
            if item.mappingModel not in items_by_classifier:
                items_by_classifier[item.mappingModel] = []
            items_by_classifier[item.mappingModel].append(item)

    models: List[MappingClassifier] = []
    for model_name, items in items_by_classifier.items():
        model = model_loader.get_mapping_model(model_name, items, params)
        if model is None and model_name != "assigned":
            raise Exception(f"following model was not found: {model_name}")
        if model is not None:
            models.append(model)
    return models
