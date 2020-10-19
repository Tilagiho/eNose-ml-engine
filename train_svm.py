#!/usr/bin/env python
# coding: utf-8

# In[107]:

dataset = None
clf = None
savename = None

# # Imports

# In[108]:


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

# scikit-learn
import sklearn
from sklearn.svm import SVC

# custom fastai additions
from fastai_additions import *

from joblib import dump, load

# # Settings

# In[102]:


# data settings
training_data = 'data/eNose-base-dataset'  # string or list of strings
normalise_data = True
balance_datasets=True
merge_eth_ipa_ac = True
convertToMultiLabels = True


# In[103]:


# training settings
useMultiLabelClassification = True
saveResults = True


# global variables
C = 0.
kernel = 'rbf'


# # Load data
def load_dataset():
    # # Load data

    # In[6]:
    # load dataset
    global dataset, convertToMultiLabels
    dataset = FuncDataset(data_dir=training_data, convertToRelativeVectors=True,
                          calculateFuncVectors=True, convertToMultiLabels=convertToMultiLabels)

    # merge Eth, IPA & Aceton
    dataset.rename_class("Ethanol", "Eth IPA Ac")
    dataset.rename_class("Isopropanol", "Eth IPA Ac")
    dataset.rename_class("Aceton", "Eth IPA Ac")


def resample_dataset():
    # split into train & validate set based on dir name
    dataset.setDirSplit(["train"], ["validate"], normaliseData=True, balanceDatasets=True)


# # Create & fit support vector machine

# In[64]:


# get model name
# if saveResults:
#    train_name = input()
#    load_name = train_name


# In[104]:

def fit_svm():
    # create support vector machine
    global clf, dataset, C, savename, kernel

    clf =\
        sklearn.OneVsRestClassifier(SVC(gamma='auto', kernel=kernel, probability=True, C=C))
    clf.fit(dataset.train_set, dataset.train_classes)

    # In[50]:
    # save svm using joblibs pickle replacement
    dump(clf, 'models/{}.joblib'.format(savename))

