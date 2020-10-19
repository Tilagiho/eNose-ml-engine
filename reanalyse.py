import svm_loader
from joblib import load

import pandas as pd

#####################
#   model settings  #
#####################
# general settings
# model_type = 'perceptron'
model_type = 'svm'

iterations = 10
thresh = 0.2

# perceptron settings
nhidden = 0

if model_type is 'svm':
    svm_loader.load()

    metrics = ["multi_label_exact_match", "multi_label_precision", "multi_label_recall", "multi_label_fbeta40_macro",
               "auc"]
    experiment_df = pd.DataFrame(
        columns=metrics
    )

    for i in range(iterations):
        # add results to experiment dataframe
        results = dict()

        results["multi_label_exact_match"] = svm_loader.interps[i].exact_match_score(thresh=thresh)
        results["multi_label_precision"] = svm_loader.interps[i].precision_score(thresh=thresh)
        results["multi_label_recall"] = svm_loader.interps[i].recall_score(thresh=thresh)
        results["multi_label_fbeta40_macro"] = svm_loader.interps[i].fbeta_score(beta=4., thresh=thresh)

        _, _, roc_auc = svm_loader.interps[i].roc()
        results["auc"] = roc_auc["micro"]

        experiment_df.loc[i] = list(results.values())

    # calculate median, average & std dev
    median = experiment_df.median(axis=0)
    average = experiment_df.mean(axis=0)
    stddev = experiment_df.std(axis=0)
    max = experiment_df.max(axis=0)

    experiment_df.loc[iterations] = median
    experiment_df.loc[iterations + 1] = average
    experiment_df.loc[iterations + 2] = stddev
    experiment_df.loc[iterations + 3] = max

    # save df as csv
    experiment_df.to_csv("results/{}_thresh{}.csv".format(svm_loader.savename, thresh))
elif model_type is 'perceptron':
    pass # TODO

