import train
import fastai_additions
import numpy as np

# model settings
model_type = 'perceptron'

train.n_hidden_layers = 3
iterations = 30

# load & prepare dataset
train.load_dataset()

# train.dataset.drop_func(4)
# train.dataset.drop_func(3)

train.resample_dataset()

train.saveResults = False

# prepare model
if model_type is 'perceptron':
    train.create_learner()

# load pretrained weights
print("Insert name of pretrained model:")
model_name = input()

# load models
learners = []
interps = []
for i in range(iterations):
    if model_type is 'perceptron':
        learners.append(train.learn.load("{}_e{}".format(model_name, i)))
        interps.append(fastai_additions.MultiLabelClassificationInterpretation(train.dataset, learn=learners[i]))

from sklearn.metrics import fbeta_score, precision_score, recall_score, hamming_loss, coverage_error, label_ranking_loss, roc_auc_score, accuracy_score
#print(fbeta_score(dataset.valid_classes.numpy(), svc.predict(dataset.valid_set), average='macro', beta=3.))

fbeta_scores = []
precision_scores = []
recall_scores = []
hamming_losses = []
coverage_errors = []
label_ranking_losses = []
aucs = []
accuracy_scores = []

for learn in learners:
    preds, y_true = learn.get_preds()
    y_pred = preds > 0.3
    fbeta_scores.append(fbeta_score(y_true, y_pred, average='macro', beta=3.))
    precision_scores.append(precision_score(y_true, y_pred, average='macro'))
    recall_scores.append(recall_score(y_true, y_pred, average='macro'))
    hamming_losses.append(hamming_loss(y_true, y_pred))
    coverage_errors.append(coverage_error(y_true, y_pred))
    label_ranking_losses.append(label_ranking_loss(y_true, preds))
    aucs.append(roc_auc_score(y_true, preds))
    accuracy_scores.append(accuracy_score(y_true, y_pred))

fbeta_array = np.array(fbeta_scores)
print('fbeta_score: {}'.format(np.average(fbeta_array)))

precision_array = np.array(precision_scores)
print('precision_score: {}'.format(np.average(precision_array)))

recall_array = np.array(recall_scores)
print('recall_score: {}'.format(np.average(recall_array)))

hamming_array = np.array(hamming_losses)
print('hamming_loss: {}'.format(np.average(hamming_array)))

coverage_array = np.array(coverage_errors)
print('coverage_error: {}'.format(np.average(coverage_array)))

label_ranking_array = np.array(label_ranking_losses)
print('label_ranking_loss: {}'.format(np.average(label_ranking_array)))

auc_array = np.array(aucs)
print('auc: {}'.format(np.average(auc_array)))

accuracy_array = np.array(accuracy_scores)
print('Accuracy: {}'.format(np.average(accuracy_array)))
