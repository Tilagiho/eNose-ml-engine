from __future__ import print_function, division
import os, re, errno
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
#
# plt.ion()   # interactive mode

class FileData:
    def __init__(self, filename, dataframe, base_vector_list, sensor_failures):
        self.filename = filename
        self.dataframe = dataframe
        self.base_vector_list = base_vector_list
        self.sensor_failures = sensor_failures


# load_data:
# load DataFrames from all .csv files in dir & all of its subfolders
def load_data(dir="data"):
    file_data_list = []
    functionalisation = []

    for element in os.listdir(dir):
        path = dir + "/" + element

        # # path is sub-folder: recurse load_data with sub-folder
        # if os.path.isdir(path):
        #     (names, data, func, failures) = load_data(path)
        #     if len(functionalisation) == 0:
        #         functionalisation = func
        #     elif func != functionalisation:
        #         raise UserWarning("Error loading " + ", ".join(names) + ": Measurement with different functionalisation loaded! All measurements should be taken by sensors with the same functionalisation! ")
        #
        #     dataframes.extend(data)
        #     base_vector_list.extend(base_vectors)
        #     filenames.extend(names)
        #     sensor_failure_list.extend(failures)

        # path is csv file: load dataframe
        if not os.path.isdir(path) and path.endswith(".csv"):
            dataframe, func, base_vectors, failures = read_csv(path)

            if len(functionalisation) == 0:
                functionalisation = func
            else:
                if func != functionalisation:
                    raise UserWarning("Error loading " + ", "+ path + ": Measurement with different functionalisation loaded! All measurements should be taken by sensors with the same functionalisation! ")
            file_data_list.append(FileData(path, dataframe, base_vectors, failures))

    return file_data_list, functionalisation


# save_data:
# save dataframes as csv files in directory export
# filenames contains the path of the original measurement data
# the new path is original path with 'data/*.csv' replaced by 'export/output/*.csv'
# comments are copied from the original file
# overwrites existing files
def save_data(filenames, dataframes, output):
    for i in range(0, len(filenames)):
        original_file = filenames[i]
        dataframe = dataframes[i]
        new_file = re.sub(r'data/', r'predicted/' + output + '/', original_file)

        print("Saving " + new_file)

        # create subfolders
        if not os.path.exists(os.path.dirname(new_file)):
            try:
                os.makedirs(os.path.dirname(new_file))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # open original and new file
        with open(original_file, 'r') as of:
            with open(new_file, 'w') as nf:
                # copy comments
                line = of.readline()

                while line:
                    if line.startswith("#"):
                        nf.writelines([line])

                    line = of.readline()

                # write dataframe
                dataframe.to_csv(nf, sep=';', index=False, header=False)


# read_csv:
# load DataFrame from filename
def read_csv(filename):
    # prepare header names
    header = ["timestamp"]
    for i in range(0, 64):
        header.append("ch" + str(i))
    header.extend(["user class", "detected class"])

    # load DataFrame
    print ("Loading " + filename)
    measurement = pd.read_csv(filename, delimiter=';', comment='#', names=header)

    # get base vectors, functionalisation
    with open(filename, 'r', encoding='latin1') as f:
        line = f.readline()

        functionalisation = [0 for i in range(64)]
        base_vectors = []
        sensor_failures = [False for i in range(64)]
        while line.startswith('#'):
            if line.startswith("#baseLevel:"):
                data = (line.split("baseLevel:")[1]).split(";")
                base_vector = [data[0]]
                base_vector.extend([float(value) for value in data[1:]])
                base_vectors.append(base_vector)
            elif line.startswith("#failures:"):
                failure_string = line.split(':')[1].split('\n')[0]
                for i, character in enumerate(failure_string):
                    sensor_failures[i] = character == '1'
            elif line.startswith("#functionalisation:"):
                functionalisation = [int(func) for func in line.split(':')[1].split(';') if func != '']
            line = f.readline()

        return measurement, functionalisation, base_vectors, sensor_failures




