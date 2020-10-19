import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from typing import List

import matplotlib.pyplot as plt
import matplotlib
import math
from itertools import combinations, permutations

from funcdataset import FuncDataset
from plot_helpers import heatmap, annotate_heatmap


class FunctionalisationAnalyser:
    def __init__(self, dataset: FuncDataset):
        assert not dataset.convertToMultiLabels,  "The dataset classes cannot be one- or multi-hot encoded labels"

        self.dataset = dataset

        self.classes = dataset.classes.copy()
        self.classes.remove("No Smell")

        self.cmap_name = "Greens"

        # create normalised vectors for each class
        self.normalised_class_data = dict()
        for i, label in enumerate(self.dataset.label_encoder.transform(self.dataset.classes)):
            # filter for vectors with current label
            class_data = self.dataset.train_set[self.dataset.train_classes == label]

            # normalise vectors based on the distribution of the average vectors
            # -> directions of vectors are maintained
            average_class_vectors = np.average(class_data, axis=1).reshape(-1, 1)
            scaler = StandardScaler()
            scaler.fit(average_class_vectors)

            self.normalised_class_data[self.dataset.classes[i]] = np.zeros_like(class_data)

            for func_index in range(class_data.shape[1]):
                self.normalised_class_data[self.dataset.classes[i]][:,func_index] = scaler.transform(class_data[:, func_index].reshape(-1, 1)).reshape(-1)

    def linear_separability_score(self, classname_a: str, classname_b: str, functionalisations: List[int] = [], c: float = 1., return_clf=False):
        if not functionalisations:
            functionalisations = self.dataset.func_vector_header

        assert len(functionalisations) > 1, "linear_seperability_score can only be calculated for 2 or more functionalisations"
        #                   #
        #   prepare data    #
        #                   #
        # create index for functionalisations
        func_index = [self.dataset.func_vector_header.index(func) for func in functionalisations]

        # create filtered view on train set & classes
        data_a = self.normalised_class_data[classname_a][:, func_index]
        data_b = self.normalised_class_data[classname_b][:, func_index]
        data = np.vstack((data_a, data_b))
        binary_labels = np.concatenate([np.repeat(0, data_a.shape[0]), np.repeat(1, data_b.shape[0])])

        #                           #
        #   fit linear svm to data  #
        #                           #
        clf = svm.SVC(C=c, kernel='linear')
        clf.fit(data, binary_labels)

        #                                                   #
        #   return accuracy of the svm on the data fitted   #
        #                                                   #
        if return_clf:
            return clf.score(data, binary_labels), clf
        return clf.score(data, binary_labels)

    def class_separability_matrix(self, n_funcs: int = 2, c: float = 1.):
        cs_matrix = np.full((len(self.classes), len(self.classes)), 0.5)

        for class_pair in combinations(self.classes, 2):
            class_a, class_b = class_pair
            index_a = self.classes.index(class_a)
            index_b = self.classes.index(class_b)

            ls_score = self.func_separability_matrix(class_a, class_b, n_funcs=n_funcs, c=c).max()

            cs_matrix[index_a, index_b] = ls_score
            cs_matrix[index_b, index_a] = ls_score

        return cs_matrix

    def get_func_labels(self, n_funcs):
        func_labels_a = []
        func_labels_b = []

        funcs = [i for i in range(len(self.dataset.func_vector_header))]

        for func_combination in combinations(funcs, n_funcs):
            func_comb_a = func_combination[:math.floor(n_funcs/2.)]
            func_comb_b = func_combination[math.floor(n_funcs/2.):]

            if not func_comb_a in func_labels_a:
                func_labels_a.append(func_comb_a)

            if not func_comb_b in func_labels_b:
                func_labels_b.append(func_comb_b)

        return func_labels_a, func_labels_b

    def func_separability_matrix(self, classname_a: str, classname_b: str, n_funcs: int = 2, c: float = 1.):
        assert n_funcs > 1

        func_labels_a, func_labels_b = self.get_func_labels(n_funcs)
        fs_matrix = np.full((len(func_labels_a), len(func_labels_b)), 0.5)

        # go through possible combinations of size n_funcs
        for func_combination in combinations([i for i in range(len(self.dataset.func_vector_header))], n_funcs):
            # split combination into two parts (for x- and y-axis)
            func_comb_a = func_combination[:math.floor(n_funcs/2.)]
            func_comb_b = func_combination[math.floor(n_funcs/2.):]

            # calculate linear separability score for combination
            # func_index_a = func_labels_a.index(func_comb_a)
            # func_index_b = func_labels_b.index(func_comb_b)

            ls_score = self.linear_separability_score(classname_a, classname_b, functionalisations=func_combination,
                                                      c=c)
            # create permutations of combination
            # set ls_score for all permuatations contained in labels
            for permutation in permutations(func_combination, n_funcs):
                func_perm_a = permutation[:math.floor(n_funcs/2.)]
                func_perm_b = permutation[math.floor(n_funcs/2.):]

                if func_perm_a in func_labels_a and func_perm_b in func_labels_b:
                    index_a = func_labels_a.index(func_perm_a)
                    index_b = func_labels_b.index(func_perm_b)
                    index_a = func_labels_a.index(func_perm_a)
                    index_b = func_labels_b.index(func_perm_b)

                    fs_matrix[index_a, index_b] = ls_score

            # ls_score = self.linear_separability_score(classname_a, classname_b, functionalisations=func_combination, c=c)
            # fs_matrix[func_index_a, func_index_b] = ls_score

        return fs_matrix

    def plot_func_separability_matrix(self, classname_a: str, classname_b: str, n_funcs: int, c: float = 1.):
        print("Plotting func separability matrix...")
        # calculate matrix
        fs_matrix = self.func_separability_matrix(classname_a, classname_b, n_funcs, c=c)

        # generate labels
        labels_a, labels_b = self.get_func_labels(n_funcs)
        labels_a = [', '.join([str(elem) for elem in comb]) for comb in labels_a]
        labels_b = [', '.join([str(elem) for elem in comb]) for comb in labels_b]

        # plot heatmap
        fig, ax = plt.subplots()

        im, cbar = heatmap(fs_matrix, labels_a, labels_b, ax=ax,
                           cmap=self.cmap_name, cbarlabel="separability of \"{}\" and \"{}\"".format(classname_a, classname_b))

        texts = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.0)

    def plot_class_separability_matrix(self, n_funcs=2, c = 1.):
        print("Plotting class separabilitiy matrix...")
        cs_matrix = self.class_separability_matrix(n_funcs=n_funcs, c=c)

        fig, ax = plt.subplots()

        im, cbar = heatmap(cs_matrix, self.classes, self.classes, ax=ax,
                           cmap=self.cmap_name, cbarlabel="Class separability with {} funcs".format(n_funcs))

        texts = annotate_heatmap(im, valfmt="{x:.3f}", threshold=0.0)

    def plot_func_relationships(self, fixed_func: int, classlist: list = None):
        print ("Plotting functionalisation crossplot...")
        if classlist is None:
            classlist = self.classes

        # prepare data
        X = np.concatenate([self.normalised_class_data[classname] for classname in classlist])
        y = np.concatenate([np.repeat(self.dataset.label_encoder.transform([classname]), self.normalised_class_data[classname].shape[0]) for classname in classlist])

        # start plotting
        # determine number of subplots
        n_subplots = X.shape[1]

        plots_per_row = math.ceil(n_subplots/2)

        fig, axes = plt.subplots(nrows=2, ncols=plots_per_row)
        fig.suptitle("Crossplots for func " + str(fixed_func))

        for func in range(n_subplots):
            ax = axes[math.floor(func / plots_per_row), func % plots_per_row]
            ax.set_title(str(fixed_func) + " vs " + str(func))
            for classname in classlist:
                label = self.dataset.label_encoder.transform([classname])
                indexes = y == label

                ax.scatter(X[indexes, fixed_func], X[indexes, func], label=classname, marker='.')

        # add legend for the whole plot
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='right', shadow=False, scatterpoints=1)
        plt.show()


if __name__ == "__main__":
    merge_eth_ipa_ac = False
    generate = True
    balance = False
    normalise = True
    dataset = FuncDataset(data_dir='data/eNose-dt-showcase',
                          convertToRelativeVectors=True, calculateFuncVectors=True,
                          convertToMultiLabels=False)
    if merge_eth_ipa_ac:
        dataset.rename_class("Aceton", "Eth IPA Ac")
        dataset.rename_class("Ethanol", "Eth IPA Ac")
        dataset.rename_class("Isopropanol", "Eth IPA Ac")

    dataset.setDirSplit(["train"], [], normaliseData=normalise, balanceDatasets=balance)
    # dataset.setDirSplit(["train"], ["validate"], normaliseData=normalise, balanceDatasets=balance)
    # dataset.setDirSplit(["4_1_7-4_1_10"], [], generateData=generate, normaliseData=normalise, balanceDatasets=balance)
    # dataset.setDirSplit(["reference"], [], generateData=generate, normaliseData=normalise, balanceDatasets=balance)

    analyser = FunctionalisationAnalyser(dataset)
    # analyser.plot_func_relationships(0, classlist=['Ammoniak', 'Eth IPA Ac'])
    # analyser.plot_func_relationships(0, classlist=['Toluol', 'Eth IPA Ac'])
    # analyser.plot_func_relationships(0, classlist=['Toluol', 'Ammoniak'])
    #
    # print(analyser.linear_separability_score(functionalisations=dataset.func_vector_header, classname_a='Toluol', classname_b='Eth IPA Ac'))
    # print(analyser.linear_separability_score(functionalisations=dataset.func_vector_header, classname_a='Ammoniak', classname_b='Eth IPA Ac'))
    # print(analyser.linear_separability_score(functionalisations=dataset.func_vector_header, classname_a='Toluol', classname_b='Ammoniak'))

    #dataset.plot_func_relationships(0, ['Toluol', 'Eth IPA Ac'])
    #analyser.plot_func_relationships(0, ['Toluol', 'Eth IPA Ac'])
    #analyser.plot_class_separability_matrix(n_funcs=2)
    #analyser.plot_class_separability_matrix(n_funcs=3)
    #analyser.plot_class_separability_matrix(n_funcs=8)

    #analyser.plot_func_separability_matrix('Toluol', 'Ammoniak', n_funcs=2)

    #dataset.plotLDA()
    #dataset.plotPCA()