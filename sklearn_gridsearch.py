from funcdataset import FuncDataset
import numpy as np
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from skmultilearn.adapt import MLkNN
from sklearn import svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import fbeta_score, f1_score, make_scorer
from functools import partial

# load dataset
dataset = FuncDataset(data_dir='data/eNose-base-dataset', convertToRelativeVectors=True,
                         calculateFuncVectors=True, convertToMultiLabels=True)

# merge Ethanol, Isopropanol and Aceton
dataset.rename_class("Ethanol", "Eth IPA Ac")
dataset.rename_class("Isopropanol", "Eth IPA Ac")
dataset.rename_class("Aceton", "Eth IPA Ac")

# set train & validation set
dataset.setDirSplit(["train"], ["validate"], normaliseData=True, balanceDatasets=True)

# prepare parameters
parameters = [{'estimator__kernel':['rbf'], 'estimator__C': [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]}]
#              {'estimator__kernel': ['linear'], 'estimator__C': [1, 10, 100, 1000]},
#              {'estimator__kernel': ['poly'], 'estimator__C': [1, 10, 100, 1000], 'estimator__degree': [3, 4, 5]}]
# parameters = {'k': range(1,3), 's': [0.5, 0.7, 1.0]}



# prepare f3-score
f3_score = partial(fbeta_score, beta=3., average='macro', zero_division=1)

# train with fixed train-test-set split
train_valid_set = np.append(dataset.train_set, dataset.valid_set, axis=0)
train_valid_classes = np.append(dataset.train_classes, dataset.valid_classes, axis=0)
test_fold = np.append(np.repeat(-1, dataset.train_set.shape[0]), np.repeat(0, dataset.valid_set.shape[0]))
ps = PredefinedSplit(test_fold)

# create classifier & grid search
svc = OneVsOneClassifier(svm.SVC(probability=True),)
clf = GridSearchCV(svc, parameters, scoring=make_scorer(f3_score), cv=ps, verbose=3, n_jobs=3, refit=False)
# clf = GridSearchCV(MLkNN(), parameters, scoring=make_scorer(f3_score), cv=ps, verbose=3, n_jobs=3, refit=False)

clf.fit(train_valid_set, train_valid_classes)

# refit model with best parameter combination
# model has to be refitted, because GridSearchcv refits best model with train+validation set
