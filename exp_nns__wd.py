import train
import fastai_additions

import pandas as pd

# experiment settings
train.n_hidden_layers = 5
train.weight_decay = 0.15
iterations = 30

# load & prepare dataset
train.load_dataset()

train.dataset.drop_func(4)
train.dataset.drop_func(3)

# experiment loop
for nhidden in [2, 3, 4]:
    for weight_decay in [0.00, 0.025, 0.05, 0.10]:
        train.n_hidden_layers = nhidden
        train.weight_decay = weight_decay

        metrics = ["total_epochs", "best_epoch", "multi_label_exact_match", "multi_label_precision", "multi_label_recall", "multi_label_fbeta40_macro", "auc"]
        experiment_df = pd.DataFrame(
            columns=metrics
        )

        for i in range(iterations):
            print("\n______________________________________________________")
            print("Experiment:\nnhidden: {}\twd: {}\niteration {}".format(train.n_hidden_layers, train.weight_decay, i))
            print("______________________________________________________\n")

            # iteration settings
            train.train_name = "test2ff{}_wd{}_e{}".format(train.n_hidden_layers, train.weight_decay, i)

            # resample data, prepare & train model
            train.resample_dataset()
            train.create_learner()
            train.train_learner()

            # load trained model & create interpretation of it
            trained_learn = train.learn.load(train.train_name)
            interp = fastai_additions.MultiLabelClassificationInterpretation(train.dataset, learn=trained_learn)

            # load training metrics table into dataframe
            train_df = pd.read_csv("models/" + train.train_name + ".csv")

            # add results to experiment dataframe
            results = dict()
            results["total_epochs"] = train_df.shape[0]
            results["best_epoch"] = train_df["multi_label_fbeta40_macro"].idxmax(axis=1)

            results["multi_label_exact_match"] = train_df["multi_label_exact_match"][results["best_epoch"]]
            results["multi_label_precision"] = train_df["multi_label_precision"][results["best_epoch"]]
            results["multi_label_recall"] = train_df["multi_label_recall"][results["best_epoch"]]
            results["multi_label_fbeta40_macro"] = train_df["multi_label_fbeta40_macro"][results["best_epoch"]]

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

        experiment_df.rename(index={iterations: "median", iterations+1: "average", iterations+2: "stddev", iterations+3: "max"})

        # save df as csv
        experiment_df.to_csv("results/test2ff{}_wd{}it{}.csv".format(train.n_hidden_layers, train.weight_decay, iterations))