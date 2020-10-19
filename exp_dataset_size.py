from funcdataset import FuncDataset
import numpy as np

merge_eth_ipa_ac = True
drop_funcs = False
balance = True
normalise = True
dataset = FuncDataset(data_dir='data/eNose-base-dataset',
                      convertToRelativeVectors=True, calculateFuncVectors=True,
                      convertToMultiLabels=True)
if merge_eth_ipa_ac:
    dataset.rename_class("Aceton", "Eth IPA Ac")
    dataset.rename_class("Ethanol", "Eth IPA Ac")
    dataset.rename_class("Isopropanol", "Eth IPA Ac")
if drop_funcs:
    dataset.drop_func(3)
    dataset.drop_func(4)
    dataset.drop_func(7)

# use n days as training set
n_meas_days = 10
all_days = set(['200130', '200206', '200207', '200210', '200211', '200212', '200214', '200217', '200218', '200221', '200224'])
n_exclude = len(all_days) - n_meas_days


# get all combinations
import itertools
excludes = list(itertools.combinations(all_days, n_exclude))

scores = np.zeros((len(excludes)))
prec_scores = np.zeros((len(excludes)))
rec_scores = np.zeros((len(excludes)))

for i, exclude in enumerate(excludes):
    dataset.setDirSplit(["train"], ["validate"], normaliseData=normalise, balanceDatasets=balance, exclude=list(exclude))

    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

    from sklearn import svm
    svc = OneVsRestClassifier(svm.SVC(C=1000000, kernel='rbf'))

    svc.fit(dataset.train_set, dataset.train_classes)

    from sklearn.metrics import fbeta_score, precision_score, recall_score
    y_true = dataset.valid_classes.numpy()
    preds = svc.predict(dataset.valid_set)

    scores[i] = fbeta_score(y_true, preds, average='macro', beta=3.)
    prec_scores[i] = precision_score(y_true, preds, average='macro', zero_division=1)
    rec_scores[i] = recall_score(y_true, preds, average='macro', zero_division=1)

    print('{}|{}: {} -> {} : {}'.format(i, exclude, dataset.train_set.shape, svc.estimators_[0].n_support_+svc.estimators_[1].n_support_+svc.estimators_[2].n_support_, scores[i]))

print('n_meas_days: {}'.format(n_meas_days))
print('fbeta: {}, {}'.format(np.average(scores), np.std(scores)))
print('precision: {}, {}'.format(np.average(prec_scores), np.std(prec_scores)))
print('recall: {}, {}'.format(np.average(rec_scores), np.std(rec_scores)))

