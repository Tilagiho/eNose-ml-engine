import csv_loader

import pandas as pd
import numpy as np
import torch
from torch.utils import data
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import os
import time
import math

from typing import List

import plot_helpers

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch.nn as nn
from dataclasses import dataclass

from itertools import chain, combinations

# constants
ts = 0.33   # size of test set for train/ test split

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

"""
Heuristic for calculating more stable functionalisation vectors:
Takes the n_median_values median values of func_values and calculates their average.
Takes half of func_values median values if n_median_values == -1.
"""
def medianFuncHeuristic(func_values: list, n_median_values: int=-1) :
    # get default n_median_values
    if n_median_values==-1:
        n_median_values = int(np.ceil(len(func_values) / 2.))

    # find n_median_values medians and average
    # sort func_values
    sorted_func_values = sorted(func_values)

    remove_last = False

    while len(sorted_func_values) > n_median_values:
        if remove_last:
            sorted_func_values.pop(-1)
        else:
            sorted_func_values.pop(0)

        remove_last = not remove_last

    median_average = np.average(sorted_func_values)

    return median_average

class DirectoryFuncData(data.Dataset):
    """"
    Stores data of all files in one directory.
    Stores the original data of each file as a pandas DataFrame and the converted data as a numpy array.

    """
    def __init__(self, dir, convertToRelativeVectors=False, calculateFuncVectors=True):
        self.dir = dir
        # load raw data
        self.file_data_list, self.functionalisation = csv_loader.load_data(dir)

        # extract labels
        self.label_list = []
        for dataframe in self.get_dataframes():
            # labels:
            # replace nan by "" -> all elements are strings
            dataframe.loc[pd.isna(dataframe["user class"]), "user class"] = ""
            dataframe.loc[pd.isna(dataframe["detected class"]), "detected class"] = ""

            # extract
            labels = dataframe.loc[:, "user class"]
            self.label_list.append(labels.to_numpy())

        # extract sensor data
        vector_arrays = []
        for dataframe in self.get_dataframes():
            vector_arrays.append(dataframe.iloc[:, 1:65].to_numpy())

        # convert sensor data to relative
        if convertToRelativeVectors:
            vector_arrays = self.__get_relative_vector_arrays(vector_arrays)

        # calculate functionalisation data from sensor data
        self.func_data = []

        if calculateFuncVectors:
            for i, vector_array in enumerate(vector_arrays):
                self.func_data.append(self.__get_functionalisation_data(vector_array, self.file_data_list[i].sensor_failures))
        else:
            self.func_data = vector_arrays

    def get_filenames(self):
        filenames = []
        for file_data in self.file_data_list:
            filenames.append(file_data.filename)
        return filenames

    def get_dataframes(self):
        dataframes = []
        for file_data in self.file_data_list:
            dataframes.append(file_data.dataframe)
        return dataframes

    """ Returns concatenated np.array of the data of all files """
    def get_data(self, exclude:list):
        n_funcs = self.func_data[0].shape[1]

        # prepare empty array
        data = np.zeros((0, n_funcs))

        for filename, func_data in zip(self.get_filenames(), self.func_data):
            # check exclude list
            file_excluded = False
            for sequence in exclude:
                if sequence in filename:
                    file_excluded = True

            # append func_data to data if file is not excluded
            if not file_excluded:
                data = np.append(data, func_data).reshape((-1, n_funcs))
        return data

    """ Returns concatenated np.array of the labels of all files """
    def get_labels(self, exclude):
        # prepare empty array
        full_labels = np.zeros((0))

        for filename, labels in zip(self.get_filenames(), self.label_list):
            # check exclude list
            file_excluded = False
            for sequence in exclude:
                if sequence in filename:
                    file_excluded = True

            # append func_data to data if file is not excluded
            if not file_excluded:
                full_labels = np.append(full_labels, labels)
        return full_labels

    """ Returns concatenated np.array of the data of all files """
    def get_classified_data(self, exclude:list):
        full_data = self.get_data(exclude)
        full_labels = self.get_labels(exclude)

        if full_data.size == 0:
            return full_data

        return full_data[full_labels != ""]

    """ Returns concatenated np.array of the labels of all files """
    def get_classified_labels(self, exclude):
        full_labels = self.get_labels(exclude)

        if full_labels.size == 0:
            return full_labels

        return full_labels[full_labels != ""]

    def __get_relative_vector_arrays(self, vector_arrays):
        relative_arrays = []

        for (i, vector_array) in enumerate(vector_arrays):
            relative_array = np.zeros_like(vector_array)
            base_vectors = self.file_data_list[i].base_vector_list

            for row in range(vector_array.shape[0]):
                timestamp = self.file_data_list[i].dataframe.iloc[row, 0]

                relative_array[row] = self.get_relative_vector(vector_array[row], base_vectors, timestamp)

            relative_arrays.append(relative_array)

        return relative_arrays

    def rename_class(self, old_name: str, new_name: str):
        for labels in self.label_list:
            labels[labels == old_name] = new_name

    def drop_func(self, func: int):
        for i in range(len(self.func_data)):
            self.func_data[i] = np.delete(self.func_data[i], func, axis=1)

    @staticmethod
    def get_relative_vector(vector, base_vector_list, timestamp):
        format = '%d.%m.%Y - %H:%M:%S'

        # find base_vector of vector
        base_vector = vector
        vector_time = time.strptime(timestamp, format)[0:6]
        for bv in base_vector_list:
            base_vector_time = time.strptime(bv[0], format)[0:6]

            if base_vector_time > vector_time:
                break
            base_vector = bv[1:]

        # calculate relative channel values in %
        relative_vector = np.zeros_like(vector)
        for i in range(vector.shape[0]):
            relative_vector[i] = 100 * ((vector[i] / base_vector[i]) - 1.0)

        return relative_vector

    # set detected classes in self.dataframes to detected_class_array
    # label_array is expected to be one-dimensional & contain numeric labels (encoded classes)
    # the number of elements in label_array is expected to be equal to the sum of the elements in each element of self.dataframes
    def set_detected_classes(self, label_array, class_list):
        # go through dataframes
        # i: position in detected_class_array
        # j: position in dataframe
        i = 0
        for file_data in self.file_data_list:
            dataframe = file_data.dataframe

            for j in range(dataframe.shape[0]):
                dataframe.loc[j , 'detected class'] = class_list[label_array[i]]
                i += 1

    def set_detected_probs(self, prob_array, class_list):
        # go through dataframes
        # i: position in detected_class_array
        # j: position in dataframe
        i = 0
        for file_data in self.file_data_list:
            dataframe = file_data.dataframe

            for j in range(dataframe.shape[0]):
                # get prob string for each classes
                string_list = []
                for class_index in range(len(class_list)):
                    string_list.append("{}:{:.3f}".format(class_list[class_index], prob_array[i, class_index]))
                # join & store in dataframe
                dataframe.loc[j, 'detected class'] = ",".join(string_list)

                i += 1

    def detect_probs(self, model, class_list):
        # get probs
        probs = model.get_scores(torch.from_numpy(self.func_data).float()).data.float().numpy()

        # save in dataframes
        i=0
        for file_data in self.file_data_list:
            for j in range(file_data.dataframe.shape[0]):
                string_list = []
                for class_index in range(len(class_list)):
                    string_list.append("{}:{:.3f}".format(class_list[class_index], probs[i, class_index]))
                # join & store in dataframe
                file_data.dataframe.loc[j, 'detected class'] = ",".join(string_list)

                i += 1

    def save(self, extension):
        csv_loader.save_data(self.get_filenames(), self.get_dataframes(), extension)

    # get_functionalisation_data:
    # takes np.array of sensor measuremnts as input, outputs np.array of average value for each functionalisation
    def __get_functionalisation_data(self, meas_data, failures):
        if (failures == [True in range (len(failures))]):
            raise ValueError("Data with a full sensor failure is not permitted!")

        # generate dict {func id: #of active channels} of functionalisations
        self.funcMap = {}

        for channel in range(0, len(self.functionalisation)):
            fid = self.functionalisation[channel]

            # fid not in funcMap: zero init
            if fid not in self.funcMap.keys():
                self.funcMap[fid] = 0

            # increment # of active channels
            self.funcMap[fid] += 1

        #                                     #
        #   convert meas_data to func_data    #
        #                                     #
        func_data = np.zeros([meas_data.shape[0], len(self.funcMap)])
        func_keys = sorted(list(self.funcMap.keys()))

        for row in range (0, func_data.shape[0]):
            # create value list fo each functionalisation
            # valueMap: dict {func id: list of values}
            valueMap = {}
            for channel in range(0, meas_data.shape[1]):
                # ignore channels with sensor failures
                if failures[channel]:
                    continue

                # column: index of channels func in funcMap.keys()
                funcIndex = func_keys.index(self.functionalisation[channel])

                # add channel value
                if not funcIndex in valueMap:
                    valueMap[funcIndex] = [meas_data[row, channel]]
                else:
                    valueMap[funcIndex].append(meas_data[row, channel])

            # use heuristic to determine functionalisation values
            for funcIndex in valueMap:
                func_data[row, funcIndex] = medianFuncHeuristic(valueMap[funcIndex])

        return func_data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.get_classified_labels())

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = str(index)

        # Load data and get label
        if self.item is None:
            # Load data and get label
            X = torch.from_numpy(self.dataset.valid_set[index, :]).float()
            y = self.dataset.valid_classes[index].float()

            return X, y
        else:
            item = self.item
            self.item = None

            return item, None

        return X, y

    def __enter__(self):
        return self

@dataclass
class MetaContainer:
    classes: list

class FuncDataset(data.Dataset):
    """
    Container for multiple measurements. Used to create normalised training and test dataframes based on functionalisation of the sensor.
    Uses dataLoader to load measurements.
    """
    # load training data from data_dir
    # if test_dir != "": load test data from test_dir
    # calculate functionalisation averages based on training data
    # determine sensor failures based on traing (+ test data if test_dir specified)
    # test_dir == "": split loaded data into training & test set
    # -> if random_tt_directory_split is set to True, the subdirectories in data_dir will be randomly split into test & train set
    # normalise both sets based on functionalisation training set
    def __init__(self, data_dir="data", convertToRelativeVectors=True, calculateFuncVectors=True, convertToMultiLabels=False, n_out_vecs=1, out_vec_steps=[], n_averaged_vecs=1):
        self.data_dir = data_dir
        self.train_dirs = []
        self.test_dirs = []

        self.is_relative = convertToRelativeVectors
        self.convertToMultiLabels = convertToMultiLabels

        self.item = None

        # configurations for time series models:
        self.n_output_vectors = n_out_vecs  # number of vectors in output for each index i
        if not out_vec_steps or len(out_vec_steps) != len( self.n_output_vectors):   # if default out_vec_steps or invalid steps
            self.output_vectors_steps = range(n_out_vecs)
        else:
            self.output_vectors_steps = out_vec_steps

        self.n_averaged_vecs = n_averaged_vecs  # number of vectors averaged for each vector in output

        # init functionalisation
        self.func_vector_header = []

        # load data from all subdirs in dir and store in directory_data_dict
        self.directory_data_dict = {}

        for element in os.listdir(data_dir):
            # ignore hidden directories
            if element.startswith("."):
                continue

            path = data_dir + "/" + element

            # path is sub-folder: recurse load_data with sub-folder
            if os.path.isdir(path):
                self.directory_data_dict[path] = DirectoryFuncData(path, convertToRelativeVectors, calculateFuncVectors)

                # ignore empty directories
                if not self.directory_data_dict[path].func_data:
                    continue

                # check functionalisation
                if calculateFuncVectors:
                    dir_func_vector_header = sorted(list(self.directory_data_dict[path].funcMap.keys()))
                else:
                    dir_func_vector_header = [i for i in range(self.directory_data_dict[path].get_data()).shape[1]]
                if not self.func_vector_header:
                    self.func_vector_header = dir_func_vector_header
                else:
                    assert dir_func_vector_header == self.func_vector_header, "Inconsistant funcVectorHeader:\n{}\nvs\n{}".format(self.func_vector_header, dir_func_vector_header)


        # zero init train & test set + labels
        self.train_set = np.zeros(0)
        self.valid_set = np.zeros(0)
        self.train_classes = torch.zeros(0)
        self.valid_classes = torch.zeros(0)

        # init full data
        data_list = []
        for directory_data in self.directory_data_dict.values():
            if directory_data.func_data:
                data_list.append(directory_data.get_data([]))
        self.full_data = np.concatenate(data_list)

        # scaler for normalisation
        self.scaler = preprocessing.StandardScaler(with_mean=True)

        # loss function
        if not convertToMultiLabels:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.BCEWithLogitsLoss()

        # fastai attributes:
        self.classes = list(self.get_classes().keys())

        # if multi-hot encoding is used:
        # remove "No Smell" class from classes
        if (convertToMultiLabels):
            self.classes.remove("No Smell")
        self.c = len(self.classes)
        self.y = MetaContainer(self.classes)

        # convert classes into labels
        if convertToMultiLabels:
            self.label_encoder = preprocessing.MultiLabelBinarizer()
            self.label_encoder.fit([self.classes])
        else:
            self.label_encoder = preprocessing.LabelEncoder()
            self.label_encoder.fit(self.classes)

        # init fastai attributes & other in self.directory_data_dict
        for directory_data in self.directory_data_dict.values():
            # fastai
            directory_data.classes = self.classes
            directory_data.c = self.c
            directory_data.y = self.y

            # other
            directory_data.label_encoder = self.label_encoder
            directory_data.scaler = self.scaler


    def get_classes(self, dataset_type:str='original'):
        class_dict = {}

        # 'original': get classes from the unaltered dataset
        if dataset_type is 'original':
            for train_dir in self.directory_data_dict.values():
                # get classes & their counts
                dir_classes = np.unique(train_dir.get_classified_labels(exclude=[]), return_counts=True)

                # add to class_dict
                for i in range(len(dir_classes[0])):
                    class_label = dir_classes[0][i]
                    class_count = dir_classes[1][i]

                    if not class_label in class_dict:
                        class_dict[class_label] = class_count
                    else:
                        class_dict[class_label] += class_count
        # 'valid': get classes of the validation dataset
        elif dataset_type is 'valid':
            classes = np.unique(self.valid_classes, return_counts=True)

            for i in range(len(classes[0])):
                class_dict[classes[0][i]] = classes[1][i]
        # 'train': get classes of the training dataset
        elif dataset_type is 'train':
            classes = np.unique(self.train_classes, return_counts=True)

            for i in range(len(classes[0])):
                class_dict[classes[0][i]] = classes[1][i]
        else:
            raise NotImplementedError

        return class_dict

    """ 
    Set test_set to the classified data of the element at position index in the key list of the directory data dict.
    All other entries are added to the train_set.
    """
    def setLooSplit(self, index, generateData=False, normaliseData=True, balanceDatasets=True):
        train_dirs = sorted(self.directory_data_dict.keys())
        train_dirs.remove(train_dirs[index])
        test_dirs = [sorted(self.directory_data_dict.keys())[index]]

        self.setDirSplit(train_dirs, test_dirs, generateData, normaliseData, balanceDatasets)

    """ 
    Split all classified vectors & their correspnding label into train & test set based on train_dirs & test_dirs.
    All dir names in train_dirs and test_dirs should be contained in self.directory_data_dict.
    """
    def setDirSplit(self, train_dirs: List[str], test_dirs: List[str], generateData=False, normaliseData=True, balanceDatasets=True, exclude:list=[]):
        self.train_dirs = train_dirs
        self.test_dirs = test_dirs

        # extract train dataset & labels from train_dirs
        train_data_list = []
        train_label_list = []
        for train_dir in train_dirs:
            if not self.directory_data_dict.__contains__(train_dir):
                train_dir = self.data_dir + "/" + train_dir
                if not self.directory_data_dict.__contains__(train_dir):
                    raise ValueError(train_dir + " is not in the directory data dict!")

            train_data_list.append(self.directory_data_dict[train_dir].get_classified_data(exclude))
            train_label_list.append(self.directory_data_dict[train_dir].get_classified_labels(exclude))

        self.train_set = np.concatenate(train_data_list)
        self.train_classes = np.concatenate(train_label_list)

        # extract test dataset & labels from test_dirs
        if test_dirs:
            test_data_list = []
            test_label_list = []
            for test_dir in test_dirs:
                if not self.directory_data_dict.__contains__(test_dir):
                    test_dir = self.data_dir + "/" + test_dir
                    if not self.directory_data_dict.__contains__(test_dir):
                        raise ValueError(test_dir + " is not in the directory data dict!")

                test_data_list.append(self.directory_data_dict[test_dir].get_classified_data(exclude))
                test_label_list.append(self.directory_data_dict[test_dir].get_classified_labels(exclude))

            self.valid_set = np.concatenate(test_data_list)
            self.valid_classes = np.concatenate(test_label_list)
        else:
            self.valid_set = np.ndarray((0, self.train_set.shape[1]))
            self.valid_classes = np.ndarray((0, self.train_set.shape[1]))

        self.convert_class_labels()

        if generateData:
            self.generate_data()

        if balanceDatasets:
            self.balance_datasets()

        if normaliseData:
            self.normalise_data()

    """ 
    Prepare kfold train test split
    """
    def prepareTTSplit(self, n_splits=10):
        self.train_dirs = []
        self.test_dirs = []

        # get list of all data & labels
        data_list = []
        label_list = []
        for directory_data in self.directory_data_dict.values():
            data_list.append(directory_data.get_classified_data())
            label_list.append(directory_data.get_classified_labels())

        # only keep classified labels and data
        self.labels = np.concatenate(label_list)
        self.data = np.concatenate(data_list)

        # prepare stratified kfold split
        self.skf = StratifiedKFold(n_splits=n_splits, random_state=123, shuffle=True)

    def setTTSplit(self, index, n_splits=10, normaliseData=True, balanceDatasets=True):
        # check if split was prepared
        # prepare if not
        try:
            if self.skf.n_splits != n_splits:
                self.prepareTTSplit(n_splits)
        except NameError:
            self.prepareTTSplit(n_splits)

        # get split indexes for index
        i=0
        for (train_index, test_index) in self.skf.split(self.data, self.labels):
            if i == index:
                break

            i += 1

        # set train & test set based on indexes
        self.train_set = self.data[train_index]
        self.train_classes = self.labels[train_index]
        self.valid_set = self.data[test_index]
        self.valid_classes = self.labels[test_index]

        self.convert_class_labels()

        if balanceDatasets:
            self.balance_datasets()

        if normaliseData:
            self.normalise_data()

    # generate data:
    # average < 5 %: add quadrupled vector
    # average < 20%: add doubled vector
    # average > 5%: add halved vector
    # average > 20% add quartered vector
    # ignore samples labeled with no smell
    def generate_data(self):
        print("Generating additional data based on the data loaded...")
        no_smell_label = self.label_encoder.transform([('No Smell',)])[0]

        d_additions = []
        l_additions = []
        for dataset, labels in [(self.train_set, self.train_classes), (self.valid_set, self.valid_classes)]:
            # ignore empty datasets
            if dataset.shape[0] == 0:
                continue

            dataset_addition = np.ndarray((0, dataset.shape[1]))
            if self.convertToMultiLabels:
                labels_addition = torch.zeros((0, labels.shape[1]))
            else:
                labels_addition = torch.zeros(0)

            original_size = dataset.shape[0]
            for i in range(original_size):
                if self.convertToMultiLabels:
                    if (labels[i].numpy() == no_smell_label).all():
                        continue
                else:
                    if labels[i] == no_smell_label:
                        continue

                if self.convertToMultiLabels:
                    label = labels[i].reshape(1, -1)
                else:
                    label = labels[i]
                labels_addition = torch.cat((labels_addition, torch.cat((label, label), dim=0)), dim=0)
                if np.average(dataset[i]) < 5.:
                    dataset_addition = np.vstack((dataset_addition, dataset[i] * 4.))
                else:
                    dataset_addition = np.vstack((dataset_addition, dataset[i] / 2.))
                if np.average(dataset[i]) < 20.:
                    dataset_addition = np.vstack((dataset_addition, dataset[i] * 2.))
                else:
                    dataset_addition = np.vstack((dataset_addition, dataset[i] / 4.))

            d_additions.append(dataset_addition)
            l_additions.append(labels_addition)

        # add additions to datasets and classes
        self.train_set = np.vstack((self.train_set, d_additions[0]))
        if len(d_additions) > 1:
            self.valid_set = np.vstack((self.valid_set, d_additions[1]))

        self.train_classes = torch.cat((self.train_classes, l_additions[0]), dim=0)
        if len(l_additions) > 1:
            self.valid_classes = torch.cat((self.valid_classes, l_additions[1]), dim=0)

    # normalise data:
    # calculate mean & variance based on train set
    # apply normalisation to full, train and test set
    def normalise_data(self):
        print("Normalising datasets...")
        # calc mean & variance of train set
        self.scaler.fit(self.train_set)

        # apply normalisation to training, test & full data
        self.train_set = self.scaler.transform(self.train_set)
        if self.valid_set.shape[0] > 0:
            self.valid_set = self.scaler.transform(self.valid_set)

        # full data
        data_list = []
        for directory_data in self.directory_data_dict.values():
            if directory_data.func_data:
                data_list.append(directory_data.get_data(exclude=[]))
        self.full_data = np.concatenate(data_list)
        self.full_data = self.scaler.transform(self.full_data)

    def balance_datasets(self):
        print("Balancing datasets...")
        # balance training & test dataset:
        datasets = []

        training_class_sets = []
        upsampled_training_class_sets = []
        train_labels = []

        # single class label:
        # get set of vectors for each class
        # all vectors only have one label
        if not self.convertToMultiLabels:
            # create list of samples for each class
            for label in range(self.c):
                training_class_sets.append(self.train_set[self.train_classes == label])
                train_labels.append(label)
        # multi class labels:
        # get set of vectors for each class
        # vectors with multiple labels are added to multiple sets
        else:
            # class "No Smell": empty vector
            na_class_vector = torch.zeros(1, self.c)
            training_class_sets.append(self.train_set[(self.train_classes == na_class_vector).all(1)])
            train_labels.append(na_class_vector)

            # all other classes
            for i in range(self.c):
                label = torch.tensor([i])
                label.unsqueeze_(0)
                class_vector = torch.zeros(1, self.c).scatter(1, label, 1)

                training_class_sets.append(self.train_set[torch.eq(self.train_classes, class_vector).numpy()[:, i]])
                train_labels.append(class_vector)

        # get max number of samples
        # n_training_samples = max([class_set.shape[0] for class_set in training_class_sets])
        n_training_samples = int(np.median([class_set.shape[0] for class_set in training_class_sets]))

        # upsample samples for each class to n_training_samples
        for (i, label) in enumerate(train_labels):
            resampled_set = resample(training_class_sets[i],
                                 #replace=True,     # sample with replacement
                                 n_samples=n_training_samples,    # to match majority class
                                 random_state=123) # reproducible results
            upsampled_training_class_sets.append(resampled_set)

        # update training_data & training_classes with upsampled sets
        self.train_set = np.concatenate(upsampled_training_class_sets)
        self.train_classes = torch.from_numpy(np.concatenate([np.repeat(label, n_training_samples, axis=0) for label in train_labels]))

        # balance training dataset:
        test_class_sets = []
        resampled_test_class_sets = []
        test_labels = []

        # single class label:
        # get set of vectors for each class
        # all vectors only have one label
        if not self.convertToMultiLabels:
            # create list of samples for each class
            for label in range(len(self.label_encoder.classes_)):
                test_class_sets.append(self.valid_set[self.valid_classes == label])
                test_labels.append(label)
                # multi class labels:
                # get set of vectors for each class
                # vectors with multiple labels are added to multiple sets
        else:
            # class "No Smell": empty vector
            na_class_vector = torch.zeros(1, self.c)

            test_class_sets.append(self.valid_set[(self.valid_classes == na_class_vector).all(1)])
            test_labels.append(na_class_vector)

            # all other classes
            for i in range(self.c):
                label = torch.tensor([i])
                label.unsqueeze_(0)
                class_vector = torch.zeros(1, self.c).scatter(1, label, 1)

                test_class_sets.append(self.valid_set[torch.eq(self.valid_classes, class_vector).numpy()[:, i]])
                test_labels.append(class_vector)

        # get max number of samples
        n_test_samples = int(np.median([class_set.shape[0] for class_set in test_class_sets]))
        # n_test_samples = max([class_set.shape[0] for class_set in test_class_sets])

        # resample samples for each class to n_training_samples
        label_list = []
        for (i, label) in enumerate(test_labels):
            if test_class_sets[i].shape[0] > 0:
                resampled_set = resample(test_class_sets[i],
#                                    replace=True,     # sample with replacement
                                     n_samples=n_test_samples,    # to match median class
                                     random_state=123) # reproducible results
                resampled_test_class_sets.append(resampled_set)
                label_list.append(np.repeat(label, n_test_samples, axis=0))

        # update training_data & training_classes with upsampled sets
        if self.valid_set.shape[0] > 0:
            self.valid_set = np.concatenate(resampled_test_class_sets)
            self.valid_classes = torch.from_numpy(np.concatenate(label_list)).float()

    def delete_class(self, classname):
        self.rename_class(classname, "")

    def rename_class(self, old_name: str, new_name: str):
        # rename labels in directory_datas
        for directory_data in self.directory_data_dict.values():
            directory_data.rename_class(old_name, new_name)

        # update fastai attributes
        self.classes = list(self.get_classes().keys())

        # if multi-hot encoding is used:
        # remove "No Smell" class from classes
        if self.convertToMultiLabels:
            self.classes.remove("No Smell")
        self.c = len(self.classes)
        self.y = MetaContainer(self.classes)

        # convert classes into numeric labels
        if self.convertToMultiLabels:
            self.label_encoder.fit([self.classes])
        else:
            self.label_encoder.fit(self.classes)

        print("Renamed class \"" + old_name + "\" to \"" + new_name + "\"")

    def drop_func(self, func: int):
        '''drop func from the dataset'''
        assert func in self.func_vector_header, "Func {} not in func-header {}".format(func, self.func_vector_header)

        index = self.func_vector_header.index(func)
        # drop funcs in directory_datas
        for directory_data in self.directory_data_dict.values():
            directory_data.drop_func(index)

        # drop funcs from current dataset
        if self.train_set.shape[0] > 0:
            np.delete(self.train_set, index, axis=1)

        if self.valid_set.shape[0] > 0:
            np.delete(self.valid_set, index, axis=1)

    def convert_class_labels(self):
        '''convert self.train_classes & self.test_classes into tensors usable for training'''
        # convertToMultiLabels not set:
        # create class labels
        # don't expect multiple classes
        if not self.convertToMultiLabels:
            # convert class names into numeric labels
            self.train_classes = torch.from_numpy(self.label_encoder.transform(self.train_classes)).float()
            if self.valid_classes.shape[0] > 0:
                self.valid_classes = torch.from_numpy(self.label_encoder.transform(self.valid_classes)).float()
            else:
                self.valid_classes = torch.zeros(0)
        # convertToMultiLabels not set:
        # create multi-hot encoded labels
        else:
            # prepare label tuples
            for i in range(self.train_classes.shape[0]):
                self.train_classes[i] = [self.train_classes[i]]
            for i in range(self.valid_classes.shape[0]):
                self.valid_classes[i] = [self.valid_classes[i]]

            # init multiHotEncoded labels
            train_labels = self.label_encoder.transform(self.train_classes)
            self.train_classes = torch.from_numpy(train_labels).float()
            valid_labels = self.label_encoder.transform(self.valid_classes)
            self.valid_classes = torch.from_numpy(valid_labels).float()

            # tensors = []
            # for class_set in [self.train_classes, self.valid_classes]:
            #     tensor = torch.zeros(class_set.shape[0], self.c)
            #     tensors.append(tensor)
            #     for i in range(class_set.shape[0]):
            #         # split into class list from AnnotationString & convert into tensor with list of classlabels
            #         class_labels = class_set[i].split(",")
            #         # remove "No Smell" labels
            #         if "No Smell" in class_labels:
            #             class_labels.remove("No Smell")
            #         class_label_tensor = torch.LongTensor(self.label_encoder.transform(class_labels))
            #
            #         # create multi-hot encoded tensor
            #         class_label_tensor.unsqueeze_(0)
            #         tensor[i] = torch.zeros(class_label_tensor.size(0), self.c).scatter(1, class_label_tensor, 1)
            #
            # self.train_classes = tensors[0].float()
            # self.valid_classes = tensors[1].float()


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.train_set)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = str(index)

        # Load data and get label
        X = torch.from_numpy(self.train_set[index, :]).float()
        y = self.train_classes[index].float()

        return X, y

    def __enter__(self):
        return self

    def set_item(self, item):
        self.item = item

    def get_eval_dataset(self):
        return EvaluationDataset(self)

    def measDays(self):
        return len(self.directory_data_dict.keys())

    def getPCA(self, useTrainset=True, nComponents=2):
        if useTrainset:
            data = self.train_set
        else:
            data = self.full_data

        scaler = preprocessing.StandardScaler(with_mean=True)
        data = scaler.fit_transform(data)


        pca = PCA(n_components=nComponents)
        transformed = pca.fit_transform(data)

        max = np.amax(transformed, 0)
        min = np.amin(transformed, 0)

        return pca, scaler, max, min

    def plotPCA(self, svm=None, learn=None, axises=(0, 1), mark_valid_set=True, classList=[], whiten=False):
        print("Plotting pca...")
        data_list = [self.train_set, self.valid_set]
        label_list = [self.train_classes.numpy(), self.valid_classes.numpy()]
        X = np.concatenate(data_list)

        try:
            X = self.scaler.transform(X)
        except: # ignore if scaler was not fitted
            pass

        labels = np.concatenate(label_list).astype(dtype=int)
        y = np.array(self.label_encoder.inverse_transform(labels)).reshape([-1,1])
        if self.convertToMultiLabels:
            for i in range(y.shape[0]):
                if not y[i]:
                    y[i, 0] = tuple(('No Smell',))

        # get indexes of validation set
        valid_index = np.zeros(y.shape[0], dtype=np.bool)
        valid_index[self.train_classes.shape[0]:] = True
        train_index = ~valid_index

        # filter classes
        if classList:
            filtered_class_set = []
            filtered_class_labels = []
            filtered_valid_index = []

            if self.convertToMultiLabels:
                for i in range(X.shape[0]):
                    for classname in y[i, 0]:
                        index_added = False # make sure every index is only added once
                        if classname in classList and not index_added:
                            filtered_class_set.append(X[i].reshape((1, X.shape[1])))
                            filtered_class_labels.append(y[i].reshape((1, y.shape[1])))
                            filtered_valid_index.append(valid_index[i].reshape((1, y.shape[1])))
                            index_added = True
            else:
                for classname in classList:
                    filtered_index = y == classname
                    filtered_class_set.append(X[filtered_index])
                    filtered_class_labels.append(y[filtered_index])
                    filtered_valid_index.append(valid_index[filtered_index])

            X = np.concatenate(filtered_class_set)
            y = np.concatenate(filtered_class_labels)
            valid_index = np.concatenate(filtered_valid_index).reshape((-1))
            train_index = ~valid_index

        lda = LinearDiscriminantAnalysis()
        pca = PCA(whiten=whiten)

        # color_map = plt.cm.get_cmap('viridis')
        color_map = plt.cm.get_cmap('gist_rainbow')

#        color_list = plt.get_cmap(color_map).colors

        # methods used
        # dim_reduction_methods = [('Linear Discriminant Analysis', lda), ('Principal Component Analysis', pca)]
        dim_reduction_methods = [('Principal Component Analysis', pca)]


        # plt.figure()
        for i, (name, model) in enumerate(dim_reduction_methods):
            fig = plt.figure()
            scatterAx = fig.add_subplot(1, 1, 1)
            # cumulatedAx = fig.add_subplot(1, 2, 2)
            # plt.subplot(1, 3, i + 1, aspect=1)

            # Fit the method's model
            model.fit(X[train_index], y[train_index])
            print(name)
            print("Explained variance: " + str(model.explained_variance_ratio_))
            if name == 'Linear Discriminant Analysis':
                components = model.coef_
            else:
                print("___Principal Component Analysis___")
                print("Explained Variance\tEigenvector")
                for explained_variance, eigenvector in zip(pca.explained_variance_ratio_, pca.components_):
                    print("{:18.4f}".format(explained_variance) + "\t" + str(eigenvector) +"\n")

            # Embed the data set in 2 dimensions using the fitted model
            X_embedded = model.transform(X)

#            if svm is not None:
#                X_support = model.transform(self.train_set)[svm.support_]
#                y_support = self.label_encoder.inverse_transform(self.train_classes[svm.support_])

            if not classList:
                classList = self.label_encoder.classes_
                if self.convertToMultiLabels:
                    classList = np.append(classList, ['No Smell'])

            # Plot the projected points and show the evaluation score
            # Z = np.zeros((y.shape[0], 1), dtype=np.int)
            # Z[y == 'No Smell'] = np.full((y[y == 'No Smell'].shape[0], 1), fill_value=classes.size-1, dtype=np.int)
            # Z[y != 'No Smell'] = self.label_encoder.transform(y[y != 'No Smell']).reshape((y[y != 'No Smell'].shape[0], 1))

            #for l in np.unique(y):
            #    scatterAx.plot(X_embedded[y==l, axises[0]], X_embedded[y==l, axises[1]],  c=Z[y==l], cmap=color_map, label=l)
            for i, label in enumerate(classList):
                # construct class index
                if self.convertToMultiLabels:
                    classindex = np.zeros((y.shape[0],), dtype=np.bool)
                    for index in range(y.shape[0]):
                        classindex[index] = label in y[index, 0]
                else:
                    classindex = (y == label).ravel()

                color_index = (i+1)*color_map.N//(len(classList))
                scatterAx.scatter(X_embedded[classindex & train_index, axises[0]], X_embedded[classindex & train_index, axises[1]],
                                  label=label, c=np.array([list(color_map(color_index))]), s=10, edgecolors='none', zorder=10)

                # add markers for validation set
                if mark_valid_set:
                    scatterAx.scatter(X_embedded[classindex & valid_index, axises[0]], X_embedded[classindex & valid_index, axises[1]],
                                     s=10, c=np.array([list(color_map(color_index))]),edgecolors= (0., 0., 0., 0.5), zorder=20)

            # create decision surface
            if svm is not None:
                plot_helpers.plot_contours(scatterAx, svm=svm, pca=model, cmap=color_map, alpha=0.15, axises=axises)
            elif learn is not None:
                plot_helpers.plot_contours(scatterAx, learner=learn, pca=model, cmap=color_map, alpha=0.3, axises=axises)


            legend = scatterAx.legend(loc='best', shadow=False, scatterpoints=1)
            for handle in legend.legendHandles:
                handle._sizes = [70]
            # scatterAx.set_title("{}".format(name))

            scatterAx.set_xlabel('principal component {}'.format(axises[0]))
            scatterAx.set_ylabel('principal component {}'.format(axises[1]))


            # cumulatedAx.plot(np.cumsum(model.explained_variance_ratio_))
            # cumulatedAx.set_xlabel('Number of Components')
            # cumulatedAx.set_ylabel('Variance (%)')  # for each component

        plt.show()

        return dim_reduction_methods

    def plotLDA(self, svm=None, learn=None, axises=(0, 1), mark_valid_set=True, classList=[], whiten=False):
        print("Plotting lda...")
        data_list = [self.train_set, self.valid_set]
        label_list = [self.train_classes.numpy(), self.valid_classes.numpy()]


        X = np.concatenate(data_list)

        try:
            X = self.scaler.transform(X)
        except:  # ignore if scaler was not fitted
            pass

        labels = np.concatenate(label_list).astype(dtype=int)
        y = np.array(self.label_encoder.inverse_transform(labels)).reshape([-1, 1])

        if self.convertToMultiLabels:
            for i in range(y.shape[0]):
                if not y[i]:
                    y[i, 0] = tuple(('No Smell',))

        # get indexes of validation set
        valid_index = np.zeros(y.shape[0], dtype=np.bool)
        valid_index[self.train_classes.shape[0]:] = True
        train_index = ~valid_index

        # filter classes
        if classList:
            filtered_class_set = []
            filtered_class_labels = []
            filtered_valid_index = []

            if self.convertToMultiLabels:
                for i in range(X.shape[0]):
                    for classname in y[i, 0]:
                        index_added = False  # make sure every index is only added once
                        if classname in classList and not index_added:
                            filtered_class_set.append(X[i].reshape((1, X.shape[1])))
                            filtered_class_labels.append(y[i].reshape((1, y.shape[1])))
                            filtered_valid_index.append(valid_index[i].reshape((1, y.shape[1])))
                            index_added = True
            else:
                for classname in classList:
                    filtered_index = y == classname
                    filtered_class_set.append(X[filtered_index])
                    filtered_class_labels.append(y[filtered_index])
                    filtered_valid_index.append(valid_index[filtered_index])

            X = np.concatenate(filtered_class_set)
            y = np.concatenate(filtered_class_labels)
            valid_index = np.concatenate(filtered_valid_index).reshape((-1))
            train_index = ~valid_index

        lda = LinearDiscriminantAnalysis()
        # pca = PCA(whiten=whiten)

        # color_map = plt.cm.get_cmap('viridis')
        color_map = plt.cm.get_cmap('gist_rainbow')

        #        color_list = plt.get_cmap(color_map).colors

        # methods used
        dim_reduction_methods = [('Linear Discriminant Analysis', lda)]
        # dim_reduction_methods = [('Principal Component Analysis', pca)]

        # plt.figure()
        for i, (name, model) in enumerate(dim_reduction_methods):
            fig = plt.figure()
            scatterAx = fig.add_subplot(1, 1, 1)
            # cumulatedAx = fig.add_subplot(1, 2, 2)
            # plt.subplot(1, 3, i + 1, aspect=1)

            # Fit the method's model
            model.fit(X[train_index], y[train_index])
            print(name)
            print("Explained variance: " + str(model.explained_variance_ratio_))
            if name == 'Linear Discriminant Analysis':
                components = model.coef_

            # Embed the data set in 2 dimensions using the fitted model
            X_embedded = model.transform(X)

            #            if svm is not None:
            #                X_support = model.transform(self.train_set)[svm.support_]
            #                y_support = self.label_encoder.inverse_transform(self.train_classes[svm.support_])

            if not classList:
                classList = self.label_encoder.classes_
                if self.convertToMultiLabels:
                    classList = np.append(classList, ['No Smell'])

            # Plot the projected points and show the evaluation score
            # Z = np.zeros((y.shape[0], 1), dtype=np.int)
            # Z[y == 'No Smell'] = np.full((y[y == 'No Smell'].shape[0], 1), fill_value=classes.size-1, dtype=np.int)
            # Z[y != 'No Smell'] = self.label_encoder.transform(y[y != 'No Smell']).reshape((y[y != 'No Smell'].shape[0], 1))

            # for l in np.unique(y):
            #    scatterAx.plot(X_embedded[y==l, axises[0]], X_embedded[y==l, axises[1]],  c=Z[y==l], cmap=color_map, label=l)
            for i, label in enumerate(classList):
                # construct class index
                if self.convertToMultiLabels:
                    classindex = np.zeros((y.shape[0],), dtype=np.bool)
                    for index in range(y.shape[0]):
                        classindex[index] = label in y[index, 0]
                else:
                    classindex = (y == label).ravel()

                color_index = (i + 1) * color_map.N // (len(classList))
                scatterAx.scatter(X_embedded[classindex & train_index, axises[0]],
                                  X_embedded[classindex & train_index, axises[1]],
                                  label=label, c=np.array([list(color_map(color_index))]), s=10, edgecolors='none',
                                  zorder=10)

                # add markers for validation set
                if mark_valid_set:
                    scatterAx.scatter(X_embedded[classindex & valid_index, axises[0]],
                                      X_embedded[classindex & valid_index, axises[1]],
                                      s=10, c=np.array([list(color_map(color_index))]),
                                      edgecolors=(0., 0., 0., 0.5), zorder=20)

            # create decision surface
            if svm is not None:
                plot_helpers.plot_contours(scatterAx, svm=svm, pca=model, cmap=color_map, alpha=0.15, axises=axises)
            elif learn is not None:
                plot_helpers.plot_contours(scatterAx, learner=learn, pca=model, cmap=color_map, alpha=0.3,
                                           axises=axises)

            legend = scatterAx.legend(loc='best', shadow=False, scatterpoints=1)
            for handle in legend.legendHandles:
                handle._sizes = [70]
            # scatterAx.set_title("{}".format(name))

            scatterAx.set_xlabel('principal component {}'.format(axises[0]))
            scatterAx.set_ylabel('principal component {}'.format(axises[1]))

            # cumulatedAx.plot(np.cumsum(model.explained_variance_ratio_))
            # cumulatedAx.set_xlabel('Number of Components')
            # cumulatedAx.set_ylabel('Variance (%)')  # for each component

        plt.show()

        return dim_reduction_methods

    def getPCA(self):
        data_list = []
        label_list = []
        for directory_data in self.directory_data_dict.values():
            data_list.append(directory_data.get_classified_data())
            label_list.append(directory_data.get_classified_labels())
        X = np.concatenate(data_list)
        y = np.concatenate(label_list)

        pca = PCA()
        pca.fit(X, y)

        return pca


    def plot3DAnalysis(self):
        data_list = []
        label_list = []
        for directory_data in self.directory_data_dict.values():
            data_list.append(directory_data.get_classified_data())
            label_list.append(directory_data.get_classified_labels())

        y = np.concatenate(label_list)

        X = np.concatenate(data_list)
        y = np.concatenate(label_list)
        lda = LinearDiscriminantAnalysis()
        pca = PCA()

        # methods used
        dim_reduction_methods = [('Linear Discriminant Analysis', lda), ('Pricipal Component Analysis', pca)]

        # plt.figure()
        for i, (name, model) in enumerate(dim_reduction_methods):
            if len(set(y)) == 3 and name == 'Linear Discriminant Analysis':
                continue

            fig = plt.figure()
            scatterAx = fig.add_subplot(1, 1, 1, projection='3d')
            # cumulatedAx = fig.add_subplot(1,2,2)
            # plt.subplot(1, 3, i + 1, aspect=1)

            # Fit the method's model
            model.fit(X, y)
            print(name)
            print(model.explained_variance_ratio_)

            # Embed the data set in 2 dimensions using the fitted model
            X_embedded = model.transform(X)

            # Plot the projected points and show the evaluation score
            for c in self.label_encoder.classes_:
                scatterAx.scatter(X_embedded[y == c, 0], X_embedded[y == c, 1], X_embedded[y == c, 2],
                                  label=c, s=3)

            scatterAx.legend(loc='best', shadow=False, scatterpoints=1)
            scatterAx.set_title("{}".format(name))

        #            cumulatedAx.plot(np.cumsum(model.explained_variance_ratio_))
        #            cumulatedAx.set_xlabel('Number of Components')
        #            cumulatedAx.set_ylabel('Variance (%)')  # for each component

    plt.show()

    # set detected classes in self.dataframes to detected_class_array
    # label_array is expected to be one-dimensional & contain numeric labels (encoded classes)
    # the number of elements in label_array is expected to be equal to the sum of the elements in each element of self.dataframes
    def set_detected_classes(self, label_array):
        start_index = 0
        for func_data_dir in self.directory_data_dict.values():
            n_vectors = func_data_dir.get_labels().size
            func_data_dir.set_detected_classes(label_array[start_index : start_index+n_vectors], self.label_encoder.classes_)
            start_index += n_vectors

    def set_detected_probs(self, prob_array):
        start_index = 0
        for func_data_dir in self.directory_data_dict.values():
            n_vectors = func_data_dir.get_labels().size
            func_data_dir.set_detected_probs(prob_array[start_index : start_index+n_vectors], self.label_encoder.classes_)
            start_index += n_vectors

    def save(self, extension):
        for func_data_dir in self.directory_data_dict.values():
            func_data_dir.save(extension)

    def get_filenames(self):
        filenames = []

        for func_data_dir in self.directory_data_dict.values():
            filenames.extend(func_data_dir.get_filenames())

        return filenames

    def detect_probs(self, model):
        for func_data_dir in self.directory_data_dict.values():
            func_data_dir.detect_probs(model, self.label_encoder.classes_)

    def get_correlation_matrix(self, classlist: list = None):
        # no classlist specified:
        # use all classes
        if classlist is None:
            classlist = self.classes

        datalist = []
        for classindex in self.label_encoder.transform(classlist):
            if self.convertToMultiLabels:
                datalist.append(self.train_set[self.train_classes[:, classindex].bool()])
                datalist.append(self.valid_set[self.valid_classes[:, classindex].bool()])
            else:
                datalist.append(self.train_set[self.train_classes == classindex])
                datalist.append(self.valid_set[self.valid_classes == classindex])
        dataset = np.concatenate(datalist)

        df = pd.DataFrame(data=dataset)
        return df.corr()

    def get_r2_matrix(self, classlist: list = []):
        corr = self.get_correlation_matrix(classlist)
        return corr**2

    def plot_correlation_matrix(self, classlist: list = None):
        # get correlation matrix
        corr = self.get_correlation_matrix(classlist)

        if classlist is None:
            classlist = self.classes

        # plot correlation matrix as a heatmap
        fig, ax = plt.subplots()

        cmap_name = "YlGn"

        im, cbar = plot_helpers.heatmap(corr.to_numpy(), [str(i) for i in range(corr.shape[0])], [str(i) for i in range(corr.shape[0])], ax=ax,
                           cmap=cmap_name, cbarlabel="Correlation coefficient")
        if corr.shape[0] < 16:
            texts = plot_helpers.annotate_heatmap(im, valfmt="{x:.3f}")

        fig.suptitle("Correlation Matrix of " + ", ".join(classlist))
        # fig.tight_layout()
        plt.show()

    def plot_r2_matrix(self, classlist: list = []):
        # get correlation matrix
        r2 = self.get_r2_matrix(classlist)

        if classlist == []:
            classlist = self.classes

        # plot correlation matrix as a heatmap
        fig, ax = plt.subplots()

        cmap_name = "YlGn"

        im, cbar = plot_helpers.heatmap(r2.to_numpy(), [str(i) for i in range(r2.shape[0])],
                           [str(i) for i in range(r2.shape[0])], ax=ax,
                           cmap=cmap_name, cbarlabel="R2 value")
        if r2.shape[0] < 16:
            texts = plot_helpers.annotate_heatmap(im, valfmt="{x:.3f}")

        fig.suptitle("R2 Matrix of " + ", ".join(classlist))
        # fig.tight_layout()
        plt.show()

    def plot_func_relationships(self, fixed_func: int, classlist: list = None):
        print ("Plotting functionalisation crossplot...")
        if classlist is None:
            classlist = self.classes

            if self.convertToMultiLabels:
                classlist.append('No Smell')


        # prepare data
        X = np.concatenate([self.train_set, self.valid_set.reshape((-1, self.train_set.shape[1]))])
        # y = np.concatenate([self.train_classes, self.valid_classes.reshape((-1))])
        y = np.concatenate([self.train_classes, self.valid_classes])

        # start plotting
        # determine number of subplots
        n_subplots = self.train_set.shape[1]

        plots_per_row = math.ceil(n_subplots/2)

        fig, axes = plt.subplots(nrows=2, ncols=plots_per_row)
        fig.suptitle("Crossplots for func " + str(fixed_func))

        for func in range(self.train_set.shape[1]):
            ax = axes[math.floor(func / plots_per_row), func % plots_per_row]
            ax.set_title(str(fixed_func) + " vs " + str(func))

            for classname in classlist:
                if self.convertToMultiLabels:
                    label = self.label_encoder.transform([[classname]])
                    indexes = (y == label).all(axis=1)
                else:
                    label = self.label_encoder.transform(classname)
                    indexes = y == label

                ax.scatter(X[indexes, fixed_func], X[indexes, func], label=classname)

        # add legend for the whole plot
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='right', shadow=False, scatterpoints=1)
        plt.show()

    def plot_func_relationship(self, xfunc: int, yfunc: int,classlist: list = None):
        if classlist is None:
            classlist = self.classes

        # prepare data
        X = np.concatenate([self.train_set, self.valid_set])
        y = np.concatenate([self.train_classes, self.valid_classes])

        # start plotting
        fig, ax = plt.subplots()
        fig.suptitle("Crossplot for func " + str(xfunc) + " vs " + str(yfunc))

        for classname in classlist:
            classlabel = self.label_encoder.transform([classname])[0]

            if self.convertToMultiLabels:
                indexes = np.array(y[:, classlabel], dtype=bool)
            else:
                indexes = y == classname

            ax.scatter(X[indexes, xfunc], X[indexes, yfunc], label=classname)

        # add legend for the whole plot
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='right', shadow=False, scatterpoints=1)
        plt.show()

    def get_linear_fit(self, xfunc: int, yfunc: int, classlist: list = None):
        if classlist is None:
            classlist = self.classes

        datalist = []
        for classindex in self.label_encoder.transform(classlist):
            if self.convertToMultiLabels:
                datalist.append(self.train_set[self.train_classes[:, classindex].bool()])
                datalist.append(self.valid_set[self.valid_classes[:, classindex].bool()])
            else:
                datalist.append(self.train_set[self.train_classes == classindex])
                datalist.append(self.valid_set[self.valid_classes == classindex])
        dataset = np.concatenate(datalist)

        coefficients, residuals, _, _, _ = np.polyfit(range(dataset.shape[0]),dataset[[xfunc, yfunc]],1,full=True)

        return coefficients


class EvaluationDataset:
    def __init__(self, dataset: FuncDataset):
        self.dataset = dataset
        # fastai attributes
        self.classes = dataset.classes
        self.c = dataset.c
        self.loss_func = dataset.loss_func

        self.item = None

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset.valid_set)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = str(index)

        if self.item is None:
            # Load data and get label
            X = torch.from_numpy(self.dataset.valid_set[index, :]).float()
            y = self.dataset.valid_classes[index].float()

            return X, y
        else:
            item = self.item
            self.item = None

            return item, None

    def __enter__(self):
        return self

    def set_item(self, item):
        self.item = item

class MeasIterator:
    """
    Iterates through measurements returning a touple of the current annotated class and a list of the
    current + n-1 previous vectors. Measurements are mirrored at the margins.
    """
    def __init__(self, n: int, funcdataset: FuncDataset):
        self.n = n

        # get iterator for DirectoryDataSets
        self.dirIterator = iter(funcdataset.directory_data_dict.values())

        # prepare state variables
        try:
            # retrieve next dirData & reset positions
            self.dirData = next(self.dirIterator)
            self.i_file = 0  # position of file in current dirData
            self.i_vector = 0  # position of vector in current file
            self.annotation_dict = {}

        except StopIteration:
            raise StopIteration

    def __next__(self):
        # get annotation
        self.annotation = self.dirData.label_list[self.i_file][self.i_vector]
        if self.annotation == "":
            self.annotation = "Not_Classified"

        # update annotaton_dict
        if self.annotation_dict.__contains__(self.annotation):
            self.annotation_dict[self.annotation] += 1
        else:
            self.annotation_dict[self.annotation] = 0

        # build vector_list
        self.vector_list = []
        for i in range(self.i_vector - self.n + 1, self.i_vector + 1):
            # mirror edges
            if i < 0:
                mirrored_i = -i
            else:
                mirrored_i = i

            # get vector
            self.vector_list.append(self.dirData.func_data[self.i_file][mirrored_i])

        # increment i_vector for next iteration
        self.i_vector += 1

        # check if i_file has to be increased
        if self.i_vector >= len(self.dirData.func_data[self.i_file]):
            self.i_file += 1
            self.i_vector = 0

        # check if next dirData has to be retrieved
        if self.i_file >= len(self.dirData.func_data):
            # retrieve next dirData & reset positions
            try:
                # retrieve next dirData & reset positions
                self.dirData = next(self.dirIterator)
                self.i_file = 0  # position of file in current dirData
                self.i_vector = 0  # position of vector in current file
            except StopIteration as e:
                raise e

        self.timestamp = self.get_timestamp()

        return self.vector_list, self.timestamp, self.annotation, self.annotation_dict[self.annotation], self.i_file, self.i_vector

    def __iter__(self):
        return self

    def get_timestamp(self):
        dataframe = self.dirData.get_dataframes()[self.i_file]
        timestamp = dataframe.iloc[self.i_vector, 0]
        return timestamp


if __name__ == "__main__":
    merge_eth_ipa_ac = False
    drop_funcs = False
    generate = False
    balance = False
    normalise = True
    dataset = FuncDataset(data_dir='data/eNose-dt-showcase',
                          convertToRelativeVectors=True, calculateFuncVectors=True,
                          convertToMultiLabels=True)
    if merge_eth_ipa_ac:
        dataset.rename_class("Aceton", "Eth IPA Ac")
        dataset.rename_class("Ethanol", "Eth IPA Ac")
        dataset.rename_class("Isopropanol", "Eth IPA Ac")
    if drop_funcs:
        dataset.drop_func(7)
        dataset.drop_func(4)
        dataset.drop_func(3)

    dataset.setDirSplit(["train"], [], normaliseData=normalise, balanceDatasets=balance)
    # dataset.setDirSplit(["train"], ["validate"], normaliseData=normalise, balanceDatasets=balance)
    # dataset.setDirSplit(["reference"], [], generateData=generate, normaliseData=normalise, balanceDatasets=balance)
    # dataset.setDirSplit(["4_1_7-4_1_10"], [], generateData=generate, normaliseData=normalise, balanceDatasets=balance)

    from sklearn.multiclass import OneVsRestClassifier
    from sklearn import svm
    svc = OneVsRestClassifier(svm.SVC(C=1000000, kernel='rbf', probability=False))
    svc.fit(dataset.train_set, dataset.train_classes)

    from sklearn.metrics import fbeta_score, precision_score, recall_score, hamming_loss, roc_auc_score, accuracy_score
    #print(fbeta_score(dataset.valid_classes.numpy(), svc.predict(dataset.valid_set), average='macro', beta=3.))

    y_true = dataset.valid_classes.numpy()
    if dataset.valid_set.shape[0] > 0:
        y_pred = svc.predict(dataset.valid_set)
        preds = svc.predict_proba(dataset.valid_set)
        print('fbeta_score: {}'.format(fbeta_score(y_true, y_pred, average='macro', beta=3.)))
        print('precision_score: {}'.format(precision_score(y_true, y_pred, average='macro')))
        print('recall_score: {}'.format(recall_score(y_true, y_pred, average='macro')))
        print('hamming_loss: {}'.format(hamming_loss(y_true, y_pred)))
        print('auc: {}'.format(roc_auc_score(y_true, preds)))
        print('Accuracy: {}'.format(accuracy_score(y_true, y_pred)))

    dataset.plot_func_relationships(0)
