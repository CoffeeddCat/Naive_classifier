"""
Source Name                                             ------->  0
Material Type                                           ------->  1
Characteristics [BioSourceType]                         ------->  2
Characteristics [CellLine]                              ------->  3
Characteristics [CellType]                              ------->  4
Characteristics [DevelopmentalStage]                    ------->  5
Characteristics [DiseaseStage]                          ------->  6
Characteristics [DiseaseState]                          ------->  7
Characteristics [OrganismPart]                          ------->  8
Characteristics [Organism]                              ------->  9
Characteristics [Sex]                                   -------> 10
Labeled Extract Name                                    -------> 11
Material Type                                           -------> 12
Label                                                   -------> 13
Hybridization Name                                      -------> 14
Array Design REF                                        -------> 15
Scan Name                                               -------> 16
Array Data File                                         -------> 17
Comment [ArrayExpress FTP file]                         -------> 18
Protocol REF                                            -------> 19
Normalization Name                                      -------> 20
Comment[Normalization Type]                             -------> 21
Derived Array Data Matrix File                          -------> 22
Comment [Derived ArrayExpress FTP file]                 -------> 23
Factor Value [CELLLINE]                                 -------> 24
Factor Value [CELLTYPE]                                 -------> 25
Factor Value [DEVELOPMENTALSTAGE]                       -------> 26
Factor Value [DISEASESTAGE]                             -------> 27
Factor Value [DISEASESTATE]                             -------> 28
Factor Value [ORGANISMPART]                             -------> 29
"""

from config import *
import numpy as np
from sklearn.model_selection import train_test_split

class Loader:
    def __init__(self, label_file_path, data_file_path, label_choose, training_set_percent):
        self.label_file_path = label_file_path
        self.data_file_path = data_file_path
        self.label_choose = label_choose
        self.training_set_percent = training_set_percent

        self.initialize_label()
        self.initialize_data()

        self.mix_and_separate()

    def initialize_label(self):
        print("now initializing label...")
        self.label_file = open(self.label_file_path, 'r')
        line = self.label_file.readline()
        item_number = 0
        while True:
            line = self.label_file.readline()
            if not line:
                break
            item_number += 1

        self.label_data = np.zeros((item_number, 1))
        self.label_dict = {}
        self.number_to_label = []

        self.label_file.seek(0)
        line = self.label_file.readline()
        line_index = 0
        self.class_number = 0
        while True:
            line = self.label_file.readline()
            if not line:
                break
            line_list = line.split('\t')
            try:
                self.label_data[line_index][0] = self.label_dict[line_list[self.label_choose]]
            except KeyError:
                self.label_dict[line_list[self.label_choose]] = self.class_number
                self.label_data[line_index][0] = self.label_dict[line_list[self.label_choose]]
                self.number_to_label.append(line_list[self.label_choose])
                self.class_number += 1
            line_index += 1

        # print(self.label_dict)
        # print(self.number_to_label)
        # print(self.label_data)
        self.label_file.close()
        print("initializing label done.")

    def initialize_data(self):
        print("now initializing data...")
        self.data_file = open(self.data_file_path, 'r')

        head = self.data_file.readline()
        self.item_number = len(head.split('\t')) - 1

        self.feature_size_raw = 0
        while True:
            line = self.data_file.readline()
            if not line:
                break
            self.feature_size_raw += 1
        print(self.feature_size_raw)

        self.data = np.zeros((self.item_number, self.feature_size_raw))

        self.data_file.seek(0)
        self.data_file.readline()
        feature_index = 0
        while True:
            line = self.data_file.readline()
            if not line:
                break
            line_list = line.split()
            for i in range(1, len(line_list)):
                self.data[i-1][feature_index] = float(line_list[i])
            feature_index += 1

        self.data_file.close()
        print("initializing data done.")

    def mix_and_separate(self):
        # print(self.label_data)
        # print(self.data)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data, self.label_data, test_size = 1 - self.training_set_percent)
        # print(self.x_test)
        # print(self.y_test)

# for test
if __name__ == "__main__":
    loader = Loader(LABEL_FILE_PATH, DATA_FILE_PATH, 10, TRAINING_SET_PERCENT)