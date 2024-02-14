import os
import csv
from typing import *
import pickle
from collections import Counter

if int(os.getenv("LOAD_TENSORFLOW")) == 1:
    from keras.models import load_model
    from imblearn.under_sampling import RandomUnderSampler

import numpy as np
import pandas as pd

from src.datasets.readers.interfaces import DatasetReader, ModelReader
from src.datasets.dataset_dataclasses import DatasetColumn, TRUTH_PREFIX, FEATURE_PREFIX, DatasetRow, Transformation, ColumnData, ColumnSettings
from src.extraction.ml.models.model_dataclasses import MlModelSpecification
from src.datasets.features import make_real_features

csv.field_size_limit(2147483647)


def num_rows_in_file(dataset_path: str) -> int:
    f = open(dataset_path, encoding="utf8")
    datareader = csv.reader(f, delimiter=',', quotechar='"')
    num_rows = sum(1 for row in datareader)
    f.close()
    return num_rows


def replace_in_file(dataset_path: str, output_path: str, row_func: Callable):
    with open(dataset_path, "r", encoding="utf8") as rf:
        datareader = csv.reader(rf, delimiter=',', quotechar='"')
        with open(output_path, "w", encoding="utf8", newline='') as wf:
            writer = csv.writer(wf, delimiter=',', quotechar='"')
            for k, row in enumerate(datareader):
                if k % 100000 == 0:
                    print(k)
                row_transformed = row_func(row)
                writer.writerow(row_transformed)


def determine_columns(dataset_path: str) -> List[DatasetColumn]:
    f = open(dataset_path, encoding="utf8")
    datareader = csv.reader(f, delimiter=',', quotechar='"')
    row = datareader.__next__()
    f.close()
    output = []
    for col_idx, col_name in enumerate(row):
        if col_name.startswith(FEATURE_PREFIX):
            col_name = col_name[len(FEATURE_PREFIX)+1:]
            is_feature = True
            output.append(DatasetColumn(col_idx, col_name, is_feature))
        elif col_name.startswith(TRUTH_PREFIX):
            col_name = col_name[len(TRUTH_PREFIX)+1:]
            is_feature = False
            output.append(DatasetColumn(col_idx, col_name, is_feature))
    return output


def make_single_csv_from_parquet_files(parquet_dir: str, output_file_name: str):
    files = [x for x in os.listdir(parquet_dir) if x.endswith(".parquet")]
    output_path = os.path.join(parquet_dir, output_file_name)
    for k, file_name in enumerate(files):
        file_path = os.path.join(parquet_dir, file_name)
        df = pd.read_parquet(file_path)
        if k == 0:
            df.to_csv(output_path, mode='w', header=True, index=False)
        else:
            df.to_csv(output_path, mode='a', header=False, index=False)


class CsvDiskReader(DatasetReader):

    def __init__(self, dataset_path: str, indices_filter: Optional[List[int]], identifier_func: Optional[Callable]):
        self.rows_in_file = num_rows_in_file(dataset_path)
        indices = [x for x in range(1, self.rows_in_file)] if indices_filter is None else indices_filter
        super().__init__(indices)
        self.dataset_path = dataset_path
        self.current_index = 0
        self.current_index_pos = 0
        self.entry_counter = 0
        self.columns = determine_columns(dataset_path)
        self.col_map = {x.col_idx: x for x in self.columns}
        self.identifier_func = identifier_func if identifier_func is not None else lambda x: x

    def apply_skip_condition_fun(self, fun: Callable, args: Dict):

        # go once through all rows to check skip conditions and add indices if necessary
        to_skip = []
        for row, idx in self.row_generator():

            # check if row should be skipped
            if fun(row, idx, **args) is True:
                to_skip.append(idx)
        indices = [x for x in self.indices if x not in to_skip]
        self.reassign_indices(indices)

    def reassign_indices(self, indices: List[int]):
        super().reassign_indices(indices)
        self.init_index()

    def init_index(self):
        self.current_index_pos = 0
        self.current_index = self.indices[self.current_index_pos]

    def get_columns(self) -> List[DatasetColumn]:
        return self.columns

    def increment_index(self):
        self.current_index_pos += 1
        if self.current_index_pos <= self.total_rows - 1:
            self.current_index = self.indices[self.current_index_pos]

    def row_generator(self) -> Generator[Tuple[DatasetRow, int], None, None]:

        with open(self.dataset_path, encoding="utf8") as csvfile:
            datareader = csv.reader(csvfile, delimiter=',', quotechar='"')
            # initialise current_index
            self.init_index()
            for k, row in enumerate(datareader):
                # filter condition, also exclude header
                if k > 0 and k == self.current_index:

                    # increment current index
                    self.increment_index()

                    source_identifier = row[0]
                    template_id = row[1]
                    item_identifier = self.identifier_func(row[2])

                    features = {}
                    truth_values = {}
                    for col in self.columns:
                        if col.is_feature:
                            features[col.col_name] = row[col.col_idx]
                        else:
                            truth_values[col.col_name] = row[col.col_idx]

                    dr = DatasetRow(source_identifier, template_id, item_identifier, features)
                    dr.assign_truth_values(truth_values)

                    if self.preprocessing_fun is not None:
                        dr = self.preprocessing_fun(dr)

                    self.entry_counter += 1

                    yield dr, self.current_index-1

    def infinite_final_data_generator(self, column_data: List[ColumnData], batch_size: int) -> Generator[any, None, None]:

        def reset_batch_data():
            for col in column_data:
                col.reset_values()

        current_batch_len = 0
        reset_batch_data()
        while True:
            for k, (row, idx) in enumerate(self.row_generator()):
                for column in column_data:
                    column.add_value_simple(row.get_value(column.settings.col.col_name, column.settings.col.is_feature))

                current_batch_len += 1

                if current_batch_len >= batch_size or k >= self.total_rows-1:
                    # get final features
                    final_features = make_real_features([x for x in column_data if x.settings.col.is_feature])
                    # get final truth values
                    truth_combined = make_real_features([x for x in column_data if not x.settings.col.is_feature])[0]

                    yield final_features, truth_combined

                    # reset variables
                    reset_batch_data()
                    current_batch_len = 0

    def stream_column(self, col_name: str) -> Generator[any, None, None]:
        for row, _ in self.row_generator():
            col_value = row.get_feature(col_name)
            yield col_value

    def undersample(self, fun: Callable):
        truth_values = []
        indices = []

        for row, idx in self.row_generator():
            truth_values.append(fun(row))
            indices.append(idx)

        dist = sorted(Counter(truth_values).items(), key=lambda x: -x[1])
        print("pre-undersampling", dist)

        # perform under-sampling
        rus = RandomUnderSampler(sampling_strategy="majority")
        indices_resampled, y_resampled = rus.fit_resample(np.asarray(indices).reshape(len(truth_values), 1), truth_values)

        dist_post_resampling = Counter(y_resampled).items()
        print("post-undersampling", dist_post_resampling)

        indices_resampled = set(list(indices_resampled.reshape(len(indices_resampled)).tolist()))

        # create new file
        new_dataset_path = os.path.join(os.path.dirname(self.dataset_path), os.path.basename(self.dataset_path[0:-4])+"_resampled.csv")
        with open(new_dataset_path, "w", encoding="utf8", newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            with open(self.dataset_path, encoding="utf8", newline='') as csvfile:
                datareader = csv.reader(csvfile, delimiter=',', quotechar='"')
                for idx, row in enumerate(datareader):
                    if idx % 1000000 == 0:
                        print(idx)
                    if idx == 0 or idx in indices_resampled:
                        writer.writerow(row)
        # set new file path
        self.__init__(new_dataset_path, None, self.identifier_func)


class DiskModelReader(ModelReader):

    def __init__(self):
        self.loaded = {}

    def load_transformations(self):
        pass

    def load_model(self, model_specification: MlModelSpecification) -> Tuple[any, Dict[str, ColumnSettings]]:
        if model_specification.internal_name in self.loaded:
            return self.loaded[model_specification.internal_name]
        keras_model = load_model(model_specification.model_path())
        with open(model_specification.settings_path(), "rb") as t:
            transformations = pickle.load(t)

            self.loaded[model_specification.internal_name] = keras_model, transformations
            return self.loaded[model_specification.internal_name]
