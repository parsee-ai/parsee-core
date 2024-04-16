import shutil
from typing import *
import os
import csv
import uuid
import pickle
from shutil import rmtree

from parsee.datasets.dataset_dataclasses import BaseDatasetRow, Transformation, ColumnSettings
from parsee.datasets.writers.interfaces import DatasetWriter, ModelWriter


class CsvDiskWriter(DatasetWriter):

    def __init__(self, write_location: str, create_sub_dir: bool = True):

        # create folder
        folder_name = "dataset_"+str(uuid.uuid4())
        final_path = os.path.join(write_location, folder_name) if create_sub_dir else write_location
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        self.base_location = write_location
        self.write_location = final_path
        self.file = {}
        self.writer = {}
        self.rows_written = {}
        self.full_paths = {}

    def file_name(self, dataset_name: str):
        return dataset_name+".csv"

    def create_dataset_file(self, dataset: str):

        full_file_path = os.path.join(self.write_location, self.file_name(dataset))
        self.full_paths[dataset] = full_file_path
        self.file[dataset] = open(full_file_path, "w", newline='', encoding="utf8")
        self.writer[dataset] = csv.writer(self.file[dataset], delimiter=',', quotechar='"')
        self.rows_written[dataset] = 0

    def _write_rows(self, dataset_rows: List[BaseDatasetRow], dataset: str):

        if len(dataset_rows) == 0:
            return

        if dataset not in self.file:
            self.create_dataset_file(dataset)

        if self.rows_written[dataset] == 0:
            self.writer[dataset].writerow(dataset_rows[0].column_names())

        self.writer[dataset].writerows([x.to_list() for k, x in enumerate(dataset_rows)])
        self.rows_written[dataset] += len(dataset_rows)

    def finish_writing(self):
        # close files
        for _, file in self.file.items():
            file.close()

    def delete_all_files(self):
        shutil.rmtree(self.write_location)

