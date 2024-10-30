from enum import Enum


class DocumentType(Enum):
    HTML = "html"
    PDF = "pdf"
    TEXT = "text"
    TABULAR = "tabular"
    OTHER = "other"


class ElementType(Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class OutputType(Enum):
    INTEGER = "int"
    LIST = "list"
    MULTI = "multi"
    TEXT = "text"
    NUMERIC = "numeric"
    ENTITY = "entity"
    DATE = "date"
    PERCENTAGE = "percentage"


class ContextType(Enum):
    QUESTIONS = "questions"
    ITEMS = "items"


class SearchStrategy(Enum):
    VECTOR = "vector"
    START = "start"


class RunMethod(Enum):
    WORKER = "worker"
    API = "api"


class AggregationMethod(Enum):
    SUM = "SUM"
    MAX = "MAX"


class ModelType(Enum):
    GPT = "gpt"
    CUSTOM = "custom"
    REPLICATE = "replicate"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    TOGETHER = "together"
    COHERE = "cohere"
    MISTRAL = "mistral"
