from typing import *
from functools import reduce

from parsee.templates.job_template import JobTemplate
from parsee.templates.general_structuring_schema import GeneralQueryItemSchema, GeneralQuerySchema, StructuringItemSchema
from parsee.templates.element_schema import ElementDetectionSchema, ElementSchema
from parsee.utils.enums import OutputType, ContextType, SearchStrategy


id_store = {}


# creates some simple IDs for meta items etc (shorter IDs work in general better for LLM models than some long hashes for example)
def simple_id(category: str):
    if category not in id_store:
        id_store[category] = 0
    id_store[category] += 1
    return f"{category}{id_store[category]}"


class MetaItem(StructuringItemSchema):

    def __init__(self, question: str, output_type: OutputType, list_values: Optional[List[str]] = None, assigned_id: Optional[str] = None, example: Optional[str] = None, additional_info: Optional[str] = None):

        self.model = None
        self.type = output_type
        self.context = ContextType.QUESTIONS
        self.id = simple_id("meta") if assigned_id is None else assigned_id
        self.title = question
        self.additionalInfo = "" if additional_info is None else additional_info
        self.searchStrategy = SearchStrategy.VECTOR
        self.valuesList = list_values
        self.example = example
        self.keywords = None
        self.defaultValue = None
        self.customArgsJson = None


class StructuringItem(GeneralQueryItemSchema):

    def __init__(self, question: str, output_type: OutputType, list_values: Optional[List[str]] = None, meta_info: Optional[List[MetaItem]] = None, assigned_id: Optional[str] = None, example: Optional[str] = None, additional_info: Optional[str] = None, keywords: Optional[str] = None):

        self.model = None
        self.type = output_type
        self.context = ContextType.QUESTIONS
        self.id = simple_id("question") if assigned_id is None else assigned_id
        self.title = question
        self.additionalInfo = "" if additional_info is None else additional_info
        self.searchStrategy = SearchStrategy.VECTOR
        self.valuesList = list_values
        self.metaInfoIds = [] if meta_info is None else [x.id for x in meta_info]
        self.meta_info = meta_info
        self.example = example
        self.keywords = keywords
        self.defaultValue = None
        self.customArgsJson = None


class RowMapping:

    def __init__(self):
        pass # TODO


class TableItem(ElementSchema):

    def __init__(self, item_title: str, keywords: str, meta_info: Optional[List[MetaItem]] = None, row_mapping: Optional[RowMapping] = None, item_description: str = "", assigned_id: Optional[str] = None):

        self.id = simple_id("table") if assigned_id is None else assigned_id
        self.title = item_title
        self.additionalInfo = item_description
        self.keywords = keywords
        self.takeBestInProximity = False
        self.collapseColumns = False
        self.model = None
        self.searchStrategy = SearchStrategy.VECTOR
        self.mapRows = row_mapping
        self.mappingModel = None
        self.metaInfoIds = [] if meta_info is None else [x.id for x in meta_info]
        self.meta_info = meta_info


def create_template(structuring_items: Optional[List[StructuringItem]], table_items: Optional[List[TableItem]] = None) -> JobTemplate:

    structuring_items = [] if structuring_items is None else structuring_items
    table_items = [] if table_items is None else table_items
    meta_items_general = reduce(lambda acc, x: acc+x, [x.meta_info for x in structuring_items if x.meta_info is not None], [])
    meta_items_tables = reduce(lambda acc, x: acc + x, [x.meta_info for x in table_items if x.meta_info is not None], [])
    all_meta_items = list(set(meta_items_tables + meta_items_general))

    return JobTemplate(None, simple_id("template"), "", GeneralQuerySchema(structuring_items, {}), ElementDetectionSchema(table_items, {}), all_meta_items, None)
