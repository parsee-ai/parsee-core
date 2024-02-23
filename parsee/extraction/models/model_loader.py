from typing import *

from parsee.templates.element_schema import ElementDetectionSchema, ElementSchema
from parsee.templates.general_structuring_schema import StructuringItemSchema, GeneralQuerySchema, GeneralQueryItemSchema
from parsee.extraction.tasks.element_classification.element_model import ElementModel, AssignedElementModel
from parsee.extraction.tasks.questions.question_model import QuestionModel, AssignedQuestionModel
from parsee.extraction.models.llm_models.chatgpt_model import ChatGPTModel
from parsee.extraction.models.llm_models.replicate_model import ReplicateModel
from parsee.extraction.tasks.questions.question_model_llm import LLMQuestionModel
from parsee.extraction.tasks.meta_info_structuring.meta_info import MetaInfoModel
from parsee.extraction.tasks.meta_info_structuring.meta_info_llm import MetaLLMModel
from parsee.extraction.tasks.element_classification.element_model_llm import ElementModelLLM
from parsee.extraction.tasks.mappings.mapping_model import MappingModel
from parsee.extraction.tasks.mappings.mapping_model_llm import MappingModelLLM
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
            spec = self.get_model_spec(model_name)
            if spec is None or spec.model_type == ModelType.CUSTOM:
                # custom models have to be handled with a custom ModelLoader class
                return None
            return LLMQuestionModel(items, all_meta_items, self.storage, get_llm_base_model(self.get_model_spec(model_name)), **params)
    
    def get_element_model(self, model_name: Optional[str], items: List[ElementSchema], params: Dict[str, Any]) -> Union[ElementModel, None]:
        if model_name is None:
            return None
        elif model_name == "assigned":
            return AssignedElementModel(items, **params)
        else:
            spec = self.get_model_spec(model_name)
            if spec is None or spec.model_type == ModelType.CUSTOM:
                # custom models have to be handled with a custom ModelLoader class
                return None
            return ElementModelLLM(items, self.storage, get_llm_base_model(spec), **params)
    
    def get_meta_model(self, model_name: Optional[str], items: List[StructuringItemSchema], params: Dict[str, Any]) -> Union[MetaInfoModel, None]:
        if model_name is None:
            return None
        elif model_name == "assigned":
            # for assigned, no model is needed
            return None
        else:
            spec = self.get_model_spec(model_name)
            if spec is None or spec.model_type == ModelType.CUSTOM:
                # custom models have to be handled with a custom ModelLoader class
                return None
            return MetaLLMModel(items, get_llm_base_model(spec), self.storage, **params)
    
    def get_mapping_model(self, model_name: Optional[str], items: List[ElementSchema], params: Dict[str, Any]) -> Union[MappingModel, None]:
        if model_name is None:
            return None
        elif model_name == "assigned":
            return None # TODO
        else:
            spec = self.get_model_spec(model_name)
            if spec is None or spec.model_type == ModelType.CUSTOM:
                # custom models have to be handled with a custom ModelLoader class
                return None
            return MappingModelLLM(items, self.storage, get_llm_base_model(spec), **params)


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


def element_models_from_schema(schema: ElementDetectionSchema, model_loader: ModelLoader, params: Dict[str, any]) -> List[ElementModel]:

    # group items by model
    items_by_model: Dict[str, List[ElementSchema]] = {}
    for item in schema.items:
        if item.model not in items_by_model:
            items_by_model[item.model] = []
        items_by_model[item.model].append(item)

    models: List[ElementModel] = []
    for model_name, items in items_by_model.items():
        model = model_loader.get_element_model(model_name, items, params)
        if model is None and model_name != "assigned":
            raise Exception(f"following model was not found: {model_name}")
        if model is not None:
            models.append(model)
    return models


def meta_models_from_items(meta_items: List[StructuringItemSchema], model_loader: ModelLoader, params: Dict[str, str]) -> List[MetaInfoModel]:

    # group items by model
    items_by_model: Dict[str, List[StructuringItemSchema]] = {}
    for item in meta_items:
        if item.model not in items_by_model:
            items_by_model[item.model] = []
        items_by_model[item.model].append(item)

    models: List[MetaInfoModel] = []
    for model_name, items in items_by_model.items():
        model = model_loader.get_meta_model(model_name, items, params)
        if model is None and model_name != "assigned":
            raise Exception(f"following model was not found: {model_name}")
        if model is not None:
            models.append(model)
    return models


def mapping_models_from_schema(schema: ElementDetectionSchema, model_loader: ModelLoader, params: Dict[str, str]) -> List[MappingModel]:

    # group items by model
    items_by_model: Dict[str, List[ElementSchema]] = {}
    for item in schema.items:
        if item.mapRows is not None:
            if item.mappingModel not in items_by_model:
                items_by_model[item.mappingModel] = []
            items_by_model[item.mappingModel].append(item)

    models: List[MappingModel] = []
    for model_name, items in items_by_model.items():
        model = model_loader.get_mapping_model(model_name, items, params)
        if model is None and model_name != "assigned":
            raise Exception(f"following model was not found: {model_name}")
        if model is not None:
            models.append(model)
    return models
