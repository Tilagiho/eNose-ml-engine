# train.py:
# training script for data annotated by the eNoseAnnotator

from funcdataset import *
import linear_classifier
import oneHidden_reluAct_net

import natsort

from fastai.basic_data import DataBunch
import fastai
import fastai.basic_train as train
import fastai.metrics as metrics
import fastai.vision as vision
import fastai.callbacks as callbacks
from fastai.callbacks.tensorboard import *
import fastai.basic_train as basic_train

#               #
#   settings    #
#               #
# data settings
training_data = 'data/dataset_0403_full'  # string or list of strings
calculate_func_vectors = True
convert_to_relative_vectors = True
normalise_data = True
balance_datasets=True

# model settings
# model_type = 'linear'
model_type = 'simple_nn'
n_hidden_layers = 1
hidden_layer_width = 8

# training settings
max_epochs = 100
batch_size = 8


#
#
#
# models
def create_model(model_name, dataset, cv_index):
    switcher={
        'linear':linear_classifier.LinearNetwork(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= model_name + "_" + str(cv_index),
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative),
        'simple_nn':oneHidden_reluAct_net.OneHiddenNetwork(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= model_name + "_" + str(cv_index),
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative,
                                nHidden=hidden_layer_width)
    }

    return switcher.get(model_name)

#                       #
#   training script     #
#                       #
if False:
    # load dataset
    dataset = FuncDataset(data_dir=training_data, convertToRelativeVectors=convert_to_relative_vectors,
                          calculateFuncVectors=calculate_func_vectors, convertToMultiLabels=True)

    # prepare tensorboard dir
    project_id = training_data.split('/')[-1]
    tboard_path = 'data/tensorboard/'
    project_path = tboard_path + '/' + project_id
    if not os.path.exists(tboard_path):
        os.mkdir(tboard_path)

    if os.path.exists(project_path):
        try:
            last_experiment_id = \
            natsort.natsorted([dI for dI in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, dI))])[
                -1]
            experiment_index = int(last_experiment_id[len('experiment'):]) + 1
        except ValueError:
            experiment_index = 0
    else:
        os.mkdir(project_path)
        experiment_index = 0
    experiment_id = 'experiment' + str(experiment_index)

    # meas day cross validation loop
    learners = []       # list for trained learners
    interpreters = []   # list of interpreters
    for cv_index in range(dataset.measDays()):
        print ("\ncv with train set " + str(cv_index))
        # split train & eval set
        dataset.setLooSplit(cv_index, normalise_data, balance_datasets)

        # init model
        model = create_model(model_type, dataset, cv_index)
        # model.eval()

        # create fastai databunch
        train_dataLoader = DataLoader(dataset, batch_size=batch_size, num_workers=3, shuffle=True)
        valid_dataLoader = DataLoader(dataset.get_eval_dataset(), batch_size=batch_size, num_workers=3, shuffle=True)
        data = DataBunch(train_dataLoader, valid_dataLoader)

        # create fastai Learner:
        # prepare metrics
        # top2_acc = metrics.partial(metrics.top_k_accuracy, k=2)
        # top2_acc.__name__ = 'top2_accuracy'
        # top3_acc = metrics.partial(metrics.top_k_accuracy, k=3)
        # top3_acc.__name__ = 'top3_accuracy'

        f1_micro_score = metrics.MultiLabelFbeta(beta=1.0, thresh=0.3, average='micro')
        f1_macro_score = metrics.MultiLabelFbeta(beta=1.0, thresh=0.3, average='macro')
        f05_macro_score = metrics.MultiLabelFbeta(beta=0.5, thresh=0.3, average='macro')
        f2_macro_score = metrics.MultiLabelFbeta(beta=2.0, thresh=0.3, average='macro')

        metric_list = [
            f1_micro_score,
            f1_macro_score,
            f05_macro_score,
            f2_macro_score
        ]

        # create Learner
        learn = vision.Learner(data, model, metrics=metric_list, loss_func=nn.BCEWithLogitsLoss())
        # add additional callbacks
        extra_callbacks = [
            callbacks.EarlyStoppingCallback(learn, min_delta=1e-5, patience=6)
            #callbacks.SaveModelCallback(learn)
        ]
        learn.callbacks += extra_callbacks

        # tensorflow callback
        run_id = 'run' + str(cv_index)
        learn.callback_fns.append(metrics.partial(LearnerTensorboardWriter,
                                          base_dir=project_path+ "/" + experiment_id,
                                          name=run_id))

        # training loop
        #vision.lr_find(learn)
        #learn.recorder.plot()
        vision.fit_one_cycle(learn, 100, max_lr=3e-3)

        # store learner and interpreter
        learners.append(learn)
        interpreters.append(learn.interpret())

    # after training:
    # process results for easy analysis
    #max_accuracies = [np.max(np.array(x.recorder.metrics), 0)[x.recorder.metrics_names.index('accuracy')] for x in learners]
    #max_top2_accuracies = [np.max(np.array(x.recorder.metrics), 0)[x.recorder.metrics_names.index('top2_accuracy')] for x in learners]
    #max_top3_accuracies = [np.max(np.array(x.recorder.metrics), 0)[x.recorder.metrics_names.index('top3_accuracy')] for x in learners]
    #min_val_losses = [min(x.recorder.val_losses) for x in learners]

    # sorted_by_eval_loss = sorted(indexed_learners, key=lambda x: x[1].recorder.metrics[x[1].recorder.metrics_names.index('accuracy')])