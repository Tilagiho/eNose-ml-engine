#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

# # Imports

# In[2]:


# local
from funcdataset import *
import models

# pyTorch
from torch.utils.data import DataLoader

# fastai
from fastai.basic_data import DataBunch
import fastai.basic_train as basic_train
import fastai.metrics as metrics
import fastai.callbacks as callbacks
import fastai

from fastai_additions import *

# # Settings

# In[3]:


# data settings
training_data = 'data/eNose-base-dataset'  # string or list of strings
normalise_data = True
balance_datasets = True

# In[4]:


# model settings
model_type = 'multi_layer_perceptron'
n_hidden_layers = 3

# In[5]:


# training settings
max_epochs = 100
batch_size = 16
saveResults = True
weight_decay = 0.

# global variables
dataset = None
train_name = None   # save name of trained model
learn = None

def load_dataset():
    # # Load data

    # In[6]:
    # load dataset
    global dataset
    dataset = FuncDataset(data_dir=training_data, convertToRelativeVectors=True,
                          calculateFuncVectors=True, convertToMultiLabels=True)

    # merge Eth, IPA & Aceton
    dataset.rename_class("Ethanol", "Eth IPA Ac")
    dataset.rename_class("Isopropanol", "Eth IPA Ac")
    dataset.rename_class("Aceton", "Eth IPA Ac")

def resample_dataset():
    # split into train & validate set based on dir name
    #dataset.setDirSplit(["train"], ["validate"], normaliseData=True, balanceDatasets=True)
    dataset.setDirSplit(["train"], ["validate"], normaliseData=True, balanceDatasets=True)


# prepare metrics
class MultiLabelFbeta05Macro(metrics.MultiLabelFbeta): pass


class MultiLabelFbeta10Micro(metrics.MultiLabelFbeta): pass


class MultiLabelFbeta10Macro(metrics.MultiLabelFbeta): pass


class MultiLabelFbeta20Macro(metrics.MultiLabelFbeta): pass


class MultiLabelFbeta40Macro(metrics.MultiLabelFbeta): pass


def create_learner():
    # # Prepare fastai train loop

    # In[7]:

    # get model name
    global train_name
    if saveResults:
        if train_name is None:
            train_name = input()

    # In[8]:


    # create fastai databunch
    global dataset
    train_dataLoader = DataLoader(dataset, batch_size=batch_size, num_workers=3, shuffle=True)
    valid_dataLoader = DataLoader(dataset.get_eval_dataset(), batch_size=batch_size, num_workers=3, shuffle=True)
    data = DataBunch(train_dataLoader, valid_dataLoader)

    # create model
    model = models.create_model(model_type, dataset, nHiddenLayers=n_hidden_layers, loss_func=dataset.loss_func)

    # In[9]:

    # create fastai Learner:

    f1_micro_score = MultiLabelFbeta10Micro(beta=1.0, thresh=0.3, average='micro')
    f1_macro_score = MultiLabelFbeta10Macro(beta=1.0, thresh=0.3, average='macro')
    f05_macro_score = MultiLabelFbeta05Macro(beta=0.5, thresh=0.3, average='macro')
    f2_macro_score = MultiLabelFbeta20Macro(beta=2.0, thresh=0.3, average='macro')
    f4_macro_score = MultiLabelFbeta40Macro(beta=4.0, thresh=0.3, average='macro')

    precision = MultiLabelPrecision(thresh=0.3, average='micro')
    recall = MultiLabelRecall(thresh=0.3, average='micro')

    exact_match_score = MultiLabelExactMatch()

    metric_list = [
        exact_match_score,
        precision,
        recall,
        f4_macro_score
    ]

    # create Learner
    global learn
    learn = basic_train.Learner(data, model, metrics=metric_list, loss_func=dataset.loss_func)

    # add additional callbacks
    global callbacks
    extra_callbacks = [
        callbacks.EarlyStoppingCallback(learn, min_delta=1e-5, patience=10)
        # callbacks.SaveModelCallback(learn)
    ]

    learn.callbacks += extra_callbacks

    if saveResults:
        save_callbacks = [
            callbacks.SaveModelCallback(learn, name=train_name, monitor='multi_label_fbeta40_macro'),
            callbacks.CSVLogger(learn, 'models/' + train_name)
        ]

        learn.callbacks += save_callbacks


def train_learner():
    # # Train model

    # In[10]:


    # learn.lr_find()
    # learn.recorder.plot()


    # In[11]:


    learn.fit(max_epochs, wd=weight_decay)

    # # Analyse results

    # In[12]:


    # load_name = input()

    # In[ ]:


    # learn = learn.load(load_name)

    # In[ ]:

    # In[ ]:


    # interp = MultiLabelClassificationInterpretation(dataset, learn=learn)
    # interp.plot_confusion_matrix()
    # interp.plot_roc()

    # In[ ]:


if __name__ == "__main__":
    load_dataset()
    create_learner()
    train_learner()

