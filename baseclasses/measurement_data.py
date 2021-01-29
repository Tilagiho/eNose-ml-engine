import pandas as pd
import numpy as np
from datetime import datetime
from scipy import signal, interpolate

import csv_loader

class MeasurementData:
    """
    Contains one measurement
    """
    def __init__(self, file_path: str, zero_set_failing_channels: bool = True):
        dataframe, functionalisation, base_vectors, sensor_failures = csv_loader.read_csv(file_path)

        #                  #
        #    meta data     #
        #                  #
        self.n_channels = len(functionalisation)
        self.n_samples = dataframe.shape[0]

        self.base_vectors = np.array([base_vectors[i][1:] for i in range(len(base_vectors))])
        self.base_vector_timestamps = np.array([pd.to_datetime(base_vectors[i][0], infer_datetime_format=True) for i in range(len(base_vectors))])

        self.sensor_failures = sensor_failures
        self.functionalisation = functionalisation  # functionalisation: func for each channel
        self.func_map = {}                          # func_map: dict with funcs present as keys and number of channels as values
        self.update_func_map()
        self.func_list = sorted(list(self.func_map.keys()))     # func_list: sorted list of funcs present



        #           #
        #   data    #
        #           #
        self.dataframe = dataframe
        self.timestamps = pd.to_datetime(self.dataframe['timestamp'], infer_datetime_format=True)
        self.absolute_data = self.dataframe[["ch{}".format(i) for i in range(self.n_channels)]].to_numpy()

        # set failing channels to zero
        if zero_set_failing_channels:
            for channel in range(self.n_channels):
                if self.sensor_failures[channel]:
                    self.absolute_data[:, channel] = np.zeros_like(self.absolute_data[:, channel])

    def update_func_map(self):
        # generate dict {func id: #of active channels} of functionalisations
        self.func_map = {}

        for channel in range(self.n_channels):
            fid = self.functionalisation[channel]

            # fid not in funcMap: zero init
            if fid not in self.func_map.keys():
                self.func_map[fid] = 0

            # increment # of active channels
            if not self.sensor_failures[channel]:
                self.func_map[fid] += 1

    """
    Calcultates the relative vectors for index
    """
    def get_relative_data(self, index=None):
        # no index specified:
        # calculate all relative vectors
        if index is None:
            index = range(self.n_samples)

        filtered_absolute_data = self.absolute_data[index].reshape(-1, self.n_channels)
        relative_data = np.zeros_like(filtered_absolute_data)

        for row in range(filtered_absolute_data.shape[0]):
            base_vector = self.get_base_vector(self.timestamps[row])
            relative_data[row] = 100. * (filtered_absolute_data[row] / base_vector - 1)

        # correct failing channels
        for channel in range(self.n_channels):
            if self.sensor_failures[channel]:
                relative_data[:, channel] = np.zeros_like(relative_data[:, channel])

        return relative_data

    """
    Calculate the functionalisation vectors for index
    func_vector_type can be "median" or average
    
    func_vector_type: 
    "median" : use the average of n_median median values for each functionalisation
               n_median = -1: use half of the active channels for each functionalisation 
    "average": use the average of all values for each functionalisation

    """
    def get_func_data(self, index=None, func_vector_type="median", n_median=-1):
        filtered_relative_data = self.get_relative_data(index)
        func_data = np.zeros((filtered_relative_data.shape[0], len(self.func_map)))

        if func_vector_type is "median":
            for func in self.func_list:
                func_index = self.func_list.index(func)
                # active channels:
                # all channels active (non-failing) for the current functionalisation
                active_channels = \
                    [i for i in range(self.n_channels) if self.functionalisation[i] == func and not self.sensor_failures[i]]

                for row in range(filtered_relative_data.shape[0]):
                    if n_median is -1:
                        n = int(np.ceil(self.func_map[func] / 2.))
                    else:
                        n = n_median

                    # active_channel_values:
                    # collect values of all active (non-failing) channels for the current func
                    active_channel_values = \
                        [filtered_relative_data[row, i] for i in active_channels]

                    # get n median values of active_channel_values and add to func_data
                    median_list = []
                    for i in range(n):
                        median_index = active_channel_values.index(np.percentile(active_channel_values,50,interpolation='nearest'))
                        median_list.append(active_channel_values[median_index])
                        del active_channel_values[median_index]

                    func_data[row, func_index] = np.average(median_list)

        elif func_vector_type is "average":
            for func in self.func_list:
                func_index = self.func_list.index(func)
                # active channels:
                # all channels active (non-failing) for the current functionalisation
                active_channels = \
                    [i for i in range(self.n_channels) if self.functionalisation[i] == func and not self.sensor_failures[i]]

                func_data[:, func_index] = np.average(filtered_relative_data[:, active_channels], 1)

        return func_data

    def get_base_vector(self, timestamp: datetime):
        base_vector = self.base_vectors[0]
        for i in range(1, self.base_vectors.shape[0]):
            if timestamp < self.base_vector_timestamps[i]:
                return base_vector

            base_vector = self.base_vectors[i]

        return base_vector

    def get_smoothened_relative_data(self, std: float=1.):
        # interpolate equally spaced vectors
        x = (self.timestamps - self.timestamps.iloc[0]) / np.timedelta64(1, 's')
        y = self.get_relative_data()
        f = interpolate.interp1d(x, y, axis=0)

        # x_new :
        # equally spaced t in s with step size of 2s
        x_new = np.arange(x.iloc[0], x.iloc[-1], 2)
        y_new = f(x_new)

        # apply gaussian filter
        window = signal.windows.gaussian(int(8*std+1), std).reshape(-1, 1)
        return signal.convolve2d(y_new, window, mode='same', boundary='symm') / sum(sum(window)), x_new


if __name__ == "__main__":
    m = MeasurementData("/home/pingu/eNose-ml-engine/data/eNose-base-dataset/train/5_Ammoniak_200206.csv")
    func_vector = m.get_func_data(0)
    func_data = m.get_func_data()
    average_func_data = m.get_func_data(func_vector_type="average")
    filtered_data, equally_spaced_t = m.get_smoothened_relative_data()