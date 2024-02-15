import os
import csv
from typing import *
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from parsee.datasets.readers.interfaces import DatasetReader
from parsee.datasets.dataset_dataclasses import DatasetColumn, TRUTH_PREFIX, FEATURE_PREFIX, DatasetRow

csv.field_size_limit(2147483647)


def num_rows_in_file(dataset_path: str) -> int:
    f = open(dataset_path, encoding="utf8")
    datareader = csv.reader(f, delimiter=',', quotechar='"')
    num_rows = sum(1 for row in datareader)
    f.close()
    return num_rows


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


class SimpleCsvDiskReader(DatasetReader):

    def __init__(self, dataset_path: str):
        self.rows_in_file = num_rows_in_file(dataset_path)
        indices = [x for x in range(1, self.rows_in_file)]
        super().__init__(indices)
        self.dataset_path = dataset_path
        self.current_index = 0
        self.current_index_pos = 0
        self.entry_counter = 0
        self.columns = determine_columns(dataset_path)
        self.col_map = {x.col_idx: x for x in self.columns}

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
                    item_identifier = row[2]

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