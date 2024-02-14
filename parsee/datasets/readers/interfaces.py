from typing import *

from parsee.datasets.dataset_dataclasses import DatasetRow, DatasetColumn, Transformation, ColumnData, ColumnSettings
from parsee.extraction.models.model_dataclasses import MlModelSpecification


class DatasetReader:

    preprocessing_fun: Optional[Callable] = None
    indices: List[int]
    total_rows: int

    def __init__(self, indices: List[int]):
        self.reassign_indices(indices)

    def reassign_indices(self, indices: List[int]):
        self.indices = list(sorted(indices))
        self.total_rows = len(self.indices)

    def row_generator(self) -> Generator[Tuple[DatasetRow, int], None, None]:
        raise NotImplemented

    def infinite_final_data_generator(self, column_data: List[ColumnData], batch_size: int) -> Generator[any, None, None]:
        raise NotImplemented

    def apply_skip_condition_fun(self, fun: Callable, args: Dict):
        raise NotImplemented

    def set_preprocessing_fun(self, fun: Callable):
        self.preprocessing_fun = fun

    def get_columns(self) -> List[DatasetColumn]:
        raise NotImplemented

    def stream_column(self, col_name: str) -> Generator[any, None, None]:
        raise NotImplemented

    def undersample(self, fun: Callable):
        raise NotImplemented


class ModelReader:

    def load_model(self, model_specification: MlModelSpecification) -> Tuple[any, Dict[str, ColumnSettings]]:
        raise NotImplemented