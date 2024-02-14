from typing import *
from parsee.datasets.dataset_dataclasses import BaseDatasetRow, Transformation, ColumnSettings


class DatasetWriter:

    def write_rows(self, dataset_rows: List[BaseDatasetRow], dataset: str):
        self.check_consistency(dataset_rows)
        self._write_rows(dataset_rows, dataset)

    def _write_rows(self, dataset_rows: List[BaseDatasetRow], dataset: str):
        raise NotImplemented

    def check_consistency(self, rows: List[BaseDatasetRow]):
        all_keys = set()
        for row in rows:
            all_keys.update(row.column_names())
        for row in rows:
            current_column_names = set(row.column_names())
            diff = all_keys.difference(current_column_names)
            if len(diff) > 0:
                row.add_missing_columns(list(diff))

    def finish_writing(self):
        return


class ModelWriter:

    local_write_dir: str

    def set_sub_directory(self, sub_directory_name: str):
        raise NotImplemented

    def save_model_settings(self, columns: Dict[str, ColumnSettings]):
        raise NotImplemented