import train_svm
import fastai_additions
import joblib

import pandas as pd


# model settings
model_type = 'svm'
iterations = 1

# global variables
learners = []
interps = []
savename = None


def load():
    global model_type, iterations, learners, interps, savename

    # load & prepare dataset
    train_svm.load_dataset()

    # train_svm.dataset.drop_func(7)
    # train_svm.dataset.drop_func(4)
    # train_svm.dataset.drop_func(3)

    train_svm.resample_dataset()

    train_svm.saveResults = False


    # load pretrained weights
    print("Insert name of pretrained model:")
    savename = input()

    # load models
    for i in range(iterations):
        if model_type is 'svm':
            learners.append(joblib.load('models/{}_e{}.joblib'.format(savename, i)))
            interps.append(fastai_additions.MultiLabelClassificationInterpretation(train_svm.dataset, svm=learners[i]))