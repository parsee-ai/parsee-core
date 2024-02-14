from typing import Dict, List, Union, Any

from parsee.templates.element_schema import ElementDetectionSchema, ElementSchema
from parsee.templates.general_structuring_schema import StructuringItemSchema, GeneralQuerySchema, GeneralQueryItemSchema
from parsee.extraction.tasks.element_classification.element_classifier import ElementClassifier, AssignedElementClassifier
from parsee.extraction.tasks.questions.question_classifier import QuestionModel, SimpleQuestionModel, AssignedQuestionModel
from parsee.extraction.models.llm_models.chatgpt_model import ChatGPTModel
from parsee.extraction.models.llm_models.replicate_model import ReplicateModel
from parsee.extraction.tasks.questions.question_classifier_llm import LLMQuestionModel
from parsee.extraction.tasks.meta_info_structuring.meta_info import MetaInfoClassifier
from parsee.extraction.tasks.meta_info_structuring.meta_info_llm import MetaLLMClassifier
from parsee.extraction.tasks.element_classification.element_classifier_llm import ElementClassifierLLM
from parsee.extraction.tasks.mappings.mapping_classifier import MappingClassifier
from parsee.extraction.tasks.mappings.mapping_classifier_llm import MappingClassifierLLM
from parsee.storage.interfaces import StorageManager
from parsee.utils.enums import ModelType
from parsee.extraction.models.model_dataclasses import MlModelSpecification


# TODO: fix custom models


def get_question_model_from_spec(model: MlModelSpecification, items: List[GeneralQueryItemSchema], all_meta_items: List[StructuringItemSchema], storage: StorageManager, settings: Dict[str, Any]) -> LLMQuestionModel:
    if model.model_type == ModelType.GPT:
        return LLMQuestionModel(items, all_meta_items, storage, ChatGPTModel(model), **settings)
    elif model.model_type == ModelType.REPLICATE:
        return LLMQuestionModel(items, all_meta_items, storage, ReplicateModel(model), **settings)
    else:
        # only LLMs suited for general queries
        raise Exception("unsupported model")


def question_classifiers_from_schema(schema: GeneralQuerySchema, all_meta_items: List[StructuringItemSchema], storage: StorageManager, params: Dict[str, Any]) -> List[QuestionModel]:

    available_models = storage.get_available_models()

    # group items by classifier
    items_by_classifier: Dict[str, List[GeneralQueryItemSchema]] = {}
    for item in schema.items:
        if item.classifier not in items_by_classifier:
            items_by_classifier[item.classifier] = []
        items_by_classifier[item.classifier].append(item)

    models: List[QuestionModel] = []
    for key, items in items_by_classifier.items():
        if key is None:
            models.append(SimpleQuestionModel(items, all_meta_items, **params))
        elif key == "assigned":
            models.append(AssignedQuestionModel(items, all_meta_items, **params))
        else:
            # search model
            filtered = [x for x in available_models if x.internal_name == key]
            if len(filtered) == 0:
                raise Exception("unsupported model")
            model = filtered[0]
            if model.model_type == ModelType.XBRL:
                models.append(XbrlQuestionResponder(items, all_meta_items, storage, **params))
            else:
                models.append(get_question_model_from_spec(model, items, all_meta_items, storage, params))
    return models


def element_classifiers_from_schema(schema: ElementDetectionSchema, storage: StorageManager, params: Dict[str, any]) -> List[ElementClassifier]:

    available_models = storage.get_available_models()

    # group items by classifier
    items_by_classifier: Dict[str, List[ElementSchema]] = {}
    for item in schema.items:
        if item.classifier not in items_by_classifier:
            items_by_classifier[item.classifier] = []
        items_by_classifier[item.classifier].append(item)

    classifiers: List[ElementClassifier] = []
    model_readers = {}
    for key, items in items_by_classifier.items():
        if key is None:
            classifiers.append(SimpleElementClassifier(items, **params))
        elif key == "assigned":
            classifiers.append(AssignedElementClassifier(items, **params))
        else:
            # search model
            filtered = [x for x in available_models if x.internal_name == key]
            if len(filtered) == 0:
                raise Exception("unsupported model")
            model = filtered[0]
            if model.model_type == ModelType.GPT:
                classifiers.append(ElementClassifierLLM(items, storage, ChatGPTModel(model), **params))
            elif model.model_type == ModelType.REPLICATE:
                classifiers.append(ElementClassifierLLM(items, storage, ReplicateModel(model), **params))
            elif model.model_type == ModelType.CUSTOM:
                if key not in model_readers:
                    model_readers[key] = storage.get_model_reader(model)
                classifiers.append(CustomElementClassifier(items, model_readers[key], model))
            else:
                raise Exception("unsupported model")
    return classifiers


def meta_classifiers_from_items(meta_items: List[StructuringItemSchema], storage: StorageManager, params: Dict[str, str]) -> List[MetaInfoClassifier]:

    available_models = storage.get_available_models()

    # group items by classifier
    items_by_classifier: Dict[str, List[StructuringItemSchema]] = {}
    for item in meta_items:
        if item.classifier not in items_by_classifier:
            items_by_classifier[item.classifier] = []
        items_by_classifier[item.classifier].append(item)

    classifiers: List[MetaInfoClassifier] = []
    model_readers = {}
    for key, items in items_by_classifier.items():
        if key is None:
            classifiers.append(SimpleMetaInfoClassifier(items, **params))
        elif key == "assigned":
            # for assigned meta values, no model is needed
            pass
        else:
            # search model
            filtered = [x for x in available_models if x.internal_name == key]
            if len(filtered) == 0:
                raise Exception("unsupported model")
            model = filtered[0]
            if model.model_type == ModelType.GPT:
                classifiers.append(MetaLLMClassifier(items, ChatGPTModel(model), storage, **params))
            elif model.model_type == ModelType.REPLICATE:
                classifiers.append(MetaLLMClassifier(items, ReplicateModel(model), storage, **params))
            elif model.model_type == ModelType.CUSTOM:
                if key not in model_readers:
                    model_readers[key] = storage.get_model_reader(model)
                classifiers.append(CustomMetaInfoClassifier(items, model_readers[key], model))
            else:
                raise Exception("unsupported model")
    return classifiers


def mapping_classifiers_from_schema(schema: ElementDetectionSchema, storage: StorageManager, params: Dict[str, str]) -> List[MappingClassifier]:

    available_models = storage.get_available_models()

    # group items by classifier
    items_by_classifier: Dict[str, List[ElementSchema]] = {}
    for item in schema.items:
        if item.mapRows is not None:
            if item.mappingModel not in items_by_classifier:
                items_by_classifier[item.mappingModel] = []
            items_by_classifier[item.mappingModel].append(item)

    classifiers: List[MappingClassifier] = []
    model_readers = {}
    for key, items in items_by_classifier.items():
        if key is None:
            classifiers.append(SimpleMappingClassifier(items, **params))
        elif key == "assigned":
            pass # TODO
        else:
            # search model
            filtered = [x for x in available_models if x.internal_name == key]
            if len(filtered) == 0:
                raise Exception("unsupported model")
            model = filtered[0]
            if model.model_type == ModelType.GPT:
                classifiers.append(MappingClassifierLLM(items, storage, ChatGPTModel(model), **params))
            elif model.model_type == ModelType.REPLICATE:
                classifiers.append(MappingClassifierLLM(items, storage, ReplicateModel(model), **params))
            elif model.model_type == ModelType.CUSTOM:
                if key not in model_readers:
                    model_readers[key] = storage.get_model_reader(model)
                classifiers.append(MappingClassifierCustom(items, model_readers[key], model))
            else:
                raise Exception("unsupported model")
    return classifiers
