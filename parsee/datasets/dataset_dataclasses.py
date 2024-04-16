from typing import *
from dataclasses import dataclass
from enum import Enum

from parsee.utils.constants import ID_NOT_CONFIGURED


TRUTH_PREFIX = "ANSWER"
FEATURE_PREFIX = "FEATURE"


@dataclass(frozen=True)
class DatasetColumn:
    col_idx: int
    col_name: str
    is_feature: bool


class TransformationType(Enum):
    MIN_MAX = "min_max"
    CATEGORICAL = "categorical"
    WORD_VECTORS = "word_vectors"


@dataclass
class MinMaxTransformation:
    min: Optional[float]
    max: Optional[float]


@dataclass
class CategoricalTransformation:
    categories: List[str]


@dataclass
class WordVectorTransformation:
    tokenizer: any
    max_words: int
    col_index: int
    vocab_size: Optional[int]


@dataclass
class Transformation:
    type: TransformationType
    min_max: Optional[MinMaxTransformation]
    categorical: Optional[CategoricalTransformation]
    word_vectors: Optional[WordVectorTransformation]


@dataclass
class ColumnSettings:
    col: DatasetColumn
    transformation: Optional[Transformation]


class ColumnData:
    settings: ColumnSettings
    _values: List[any]

    def __init__(self, settings: ColumnSettings, values: List[any]):
        self.settings = settings
        self._values = [self.clean_value(x) for x in values]

    def clean_value(self, value: any) -> any:
        if self.settings.transformation is not None:
            if self.settings.transformation.type == TransformationType.MIN_MAX:
                value = float(value)
            if self.settings.transformation.type == TransformationType.CATEGORICAL:
                value = str(value)
            if self.settings.transformation.type == TransformationType.WORD_VECTORS:
                value = str(value)
        else:
            # no transformation -> make float
            value = float(value)
        return value

    def add_value(self, value: any, load_in_batches: bool):
        value = self.clean_value(value)
        # handle transformations, tokenizer is handled separately
        if self.settings.transformation is not None:
            if self.settings.transformation.type == TransformationType.MIN_MAX:
                if self.settings.transformation.min_max.min is None or self.settings.transformation.min_max.min > value:
                    self.settings.transformation.min_max.min = value
                if self.settings.transformation.min_max.max is None or self.settings.transformation.min_max.max < value:
                    self.settings.transformation.min_max.max = value
            if self.settings.transformation.type == TransformationType.CATEGORICAL:
                if value not in self.settings.transformation.categorical.categories:
                    self.settings.transformation.categorical.categories.append(value)

        if not load_in_batches:
            self._values.append(value)

    def add_value_simple(self, value: any):
        value = self.clean_value(value)
        self._values.append(value)

    def reset_values(self):
        self._values = []


@dataclass(frozen=True)
class MappingUniqueIdentifier:
    schema_id: str
    li_identifier: str
    kv_index: int


@dataclass(frozen=True)
class MetaUniqueIdentifier:
    detected_class: str
    col_idx: int
    kv_identifier: str


class BaseDatasetRow:
    _dict_values: Dict[str, any]

    def __init__(self, dict_values: Dict[str, any]):
        self._dict_values = {FEATURE_PREFIX + "_" + key: value for key, value in dict_values.items()}
        # check that keys are unique
        if len(self._dict_values.keys()) != len(set(self._dict_values.keys())):
            raise Exception("ids not unique")
        self.base_key_len = 0

    def column_names(self) -> List[str]:

        feature_keys = sorted(self._dict_values.keys())

        return feature_keys

    def to_list(self) -> List[str]:

        values = []
        for col_name in self.column_names():
            if col_name in self._dict_values.keys():
                col_value = str(self._dict_values[col_name]).replace('\x00', " ") # remove unreadable chars
                values.append(col_value)
        return values

    def assign_truth_values(self, values: Dict[str, any]):

        truth_keys = {TRUTH_PREFIX+"_"+key: value for key, value in values.items()}
        self._dict_values = {**self._dict_values, **truth_keys}

    def get_feature(self, feature_name: str) -> Union[any, None]:
        key = FEATURE_PREFIX+"_"+feature_name
        return self._dict_values[key] if key in self._dict_values else None

    def get_value(self, col_name: str, is_feature: bool) -> Union[any, None]:
        key = (FEATURE_PREFIX if is_feature else TRUTH_PREFIX)+"_"+col_name
        return self._dict_values[key] if key in self._dict_values else None

    def add_missing_columns(self, columns: List[str]):
        for col in columns:
            self._dict_values[col] = ID_NOT_CONFIGURED

    def get_dataset_columns(self) -> List[DatasetColumn]:
        output = []
        for k, col_name in enumerate(self.column_names()):
            if col_name.startswith(FEATURE_PREFIX):
                output.append(DatasetColumn(k+self.base_key_len, col_name[len(FEATURE_PREFIX)+1:], True))
            elif col_name.startswith(TRUTH_PREFIX):
                output.append(DatasetColumn(k + self.base_key_len, col_name[len(TRUTH_PREFIX)+1:], False))
        return output

    def get_column_by_name(self, column_name: str) -> Union[None, DatasetColumn]:

        is_feature = FEATURE_PREFIX+"_"+column_name in self.column_names()
        key = FEATURE_PREFIX+"_"+column_name if is_feature else TRUTH_PREFIX+"_"+column_name

        if key in self._dict_values:
            idx = self.column_names().index(key)
            return DatasetColumn(idx, column_name, is_feature)
        return None


class DatasetRow(BaseDatasetRow):
    source_identifier: str
    template_id: str
    element_identifier: any

    def __init__(self, source_identifier: str, template_id: str, element_identifier: any, dict_values: Dict[str, str]):
        super().__init__(dict_values)
        self.source_identifier = source_identifier
        self.template_id = template_id
        self.element_identifier = element_identifier
        self.base_columns = ["source_identifier", "template_id", "element_identifier"]
        self.base_key_len = len(self.base_columns)

    def __str__(self):
        return str(self._dict_values)

    def __repr__(self):
        return str(self)

    def column_names(self) -> List[str]:

        return self.base_columns + super().column_names()

    def to_list(self) -> List[str]:

        return [self.source_identifier, self.template_id, self.element_identifier] + super().to_list()

