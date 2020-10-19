import train_svm
import fastai_additions

import pandas as pd
import numpy as np

from joblib import load

# experiment settings
iterations = 10

kernels = ['rbf', 'linear']
train_svm.kernel = 'rbf'

param_grid = [
  {'C': [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000], 'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']}
 ]

# load & prepare dataset
train_svm.load_dataset()

# train_svm.dataset.drop_func(7)
#train_svm.dataset.drop_func(4)
#train_svm.dataset.drop_func(3)

# experiment loop
for C in np.arange(1.0, 1.1, 0.2):
    train_svm.C = C

    metrics = ["multi_label_exact_match", "multi_label_precision", "multi_label_recall", "multi_label_fbeta40_macro",
               "auc"]
    experiment_df = pd.DataFrame(
        columns=metrics
    )

    for i in range(iterations):


        print("\n______________________________________________________")
        print("Experiment:\nreg: {}\niteration {}".format(train_svm.C, i))
        print("______________________________________________________\n")

        # iteration settings
        train_svm.savename = "svm_1vr_{}_reg{}_e{}".format(train_svm.kernel, train_svm.C, i)

        # resample data, prepare & train model
        train_svm.resample_dataset()
        train_svm.fit_svm()

        # load trained model & create interpretation of it
        trained_learn = load("models/{}.joblib".format(train_svm.savename))
        interp = fastai_additions.MultiLabelClassificationInterpretation(train_svm.dataset, svm=trained_learn)

        # add results to experiment dataframe
        results = dict()

        results["multi_label_exact_match"] = interp.exact_match_score()
        results["multi_label_precision"] = interp.precision_score()
        results["multi_label_recall"] = interp.recall_score()
        results["multi_label_fbeta40_macro"] = interp.fbeta_score(beta=4.)

        _, _, roc_auc = interp.roc()
        results["auc"] = roc_auc["micro"]

        experiment_df.loc[i] = list(results.values())

    # calculate median, average & std dev
    median = experiment_df.median(axis=0)
    average = experiment_df.mean(axis=0)
    stddev = experiment_df.std(axis=0)
    max = experiment_df.max(axis=0)

    experiment_df.loc[iterations] = median
    experiment_df.loc[iterations+1] = average
    experiment_df.loc[iterations+2] = stddev
    experiment_df.loc[iterations+3] = max

    # save df as csv
    # experiment_df.to_csv("results/svm_dropped34_{}_reg{}_it{}.csv".format(train_svm.kernel, train_svm.C, iterations))
