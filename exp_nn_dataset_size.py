import train
import fastai_additions

import pandas as pd

from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np

# experiment settings
train.n_hidden_layers = 3
train.weight_decay = 0.05
iterations = 1

# load & prepare dataset
train.load_dataset()

train.dataset.drop_func(7)
train.dataset.drop_func(4)
train.dataset.drop_func(3)

# use n days as training set
n_meas_days = 3
all_days = set(['200130', '200206', '200207', '200210', '200211', '200212', '200214', '200217', '200218', '200221', '200224'])
n_exclude = len(all_days) - n_meas_days


# get all combinations
import itertools
excludes = list(itertools.combinations(all_days, n_exclude))

scores = []
prec_scores = []
rec_scores = []


# experiment loop
for exclude in excludes:
    for nhidden in [3]:
        for weight_decay in [0.05]:
            train.n_hidden_layers = nhidden
            train.weight_decay = weight_decay

            metrics = ["total_epochs", "best_epoch", "multi_label_exact_match", "multi_label_precision", "multi_label_recall", "multi_label_fbeta40_macro", "auc"]
            experiment_df = pd.DataFrame(
                columns=metrics
            )

            for i in range(iterations):
                print("\n______________________________________________________")
                print("Experiment:\n{}\niteration {}".format(exclude, i))
                print("______________________________________________________\n")

                # iteration settings
                train.train_name = "exp_size{}_ff{}_wd{}_e{}".format(n_meas_days, train.n_hidden_layers, train.weight_decay, i)

                # resample data, prepare & train model
                train.dataset.setDirSplit(["train"], ["validate"], normaliseData=True, balanceDatasets=True, exclude=list(exclude))
                train.create_learner()
                train.train_learner()

                # load trained model & create interpretation of it
                trained_learn = train.learn.load(train.train_name)

                preds, y_true = trained_learn.get_preds()
                scores.append(fbeta_score(y_true, preds>0.3, beta=3., average='macro', zero_division=1))
                prec_scores.append(precision_score(y_true, preds>0.3, average='macro', zero_division=1))
                rec_scores.append(recall_score(y_true, preds>0.3, average='macro', zero_division=1))

                print(scores[-1])
                # interp = fastai_additions.MultiLabelClassificationInterpretation(train.dataset, learn=trained_learn)

                # load training metrics table into dataframe
                # train_df = pd.read_csv("models/" + train.train_name + ".csv")

                # add results to experiment dataframe
                # results = dict()
                # results["total_epochs"] = train_df.shape[0]
                # results["best_epoch"] = train_df["multi_label_fbeta40_macro"].idxmax(axis=1)

                # results["multi_label_exact_match"] = train_df["multi_label_exact_match"][results["best_epoch"]]
                # results["multi_label_precision"] = train_df["multi_label_precision"][results["best_epoch"]]
                # results["multi_label_recall"] = train_df["multi_label_recall"][results["best_epoch"]]
                # results["multi_label_fbeta40_macro"] = train_df["multi_label_fbeta40_macro"][results["best_epoch"]]

                # _, _, roc_auc = interp.roc()
                # results["auc"] = roc_auc["micro"]

                # experiment_df.loc[i] = list(results.values())

            # calculate median, average & std dev
            # median = experiment_df.median(axis=0)
            # average = experiment_df.mean(axis=0)
            # stddev = experiment_df.std(axis=0)
            # max = experiment_df.max(axis=0)
            #
            # experiment_df.loc[iterations] = median
            # experiment_df.loc[iterations+1] = average
            # experiment_df.loc[iterations+2] = stddev
            # experiment_df.loc[iterations+3] = max

            # experiment_df.rename(index={iterations: "median", iterations+1: "average", iterations+2: "stddev", iterations+3: "max"})
            #
            # # save df as csv
            # experiment_df.to_csv("results/test2ff{}_wd{}it{}.csv".format(train.n_hidden_layers, train.weight_decay, iterations))

score_array = np.array(scores)
prec_score_array = np.array(prec_scores)
rec_score_array = np.array(rec_scores)


print('n_meas_days: {}\nfbeta:{}, {}\nprec:{}, {}\nrec:{}, {}\n'.format(n_meas_days, np.average(score_array), np.std(score_array), np.average(prec_score_array), np.std(prec_score_array), np.average(rec_score_array), np.std(rec_score_array)))