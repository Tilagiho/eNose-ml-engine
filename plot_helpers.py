from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
import fastai
import torch
from torch.utils.data import DataLoader
from fastai.torch_core import ItemBase
from scipy.sparse import issparse


import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

class gridData(ItemBase):
    def __init__(self, data:np.array):
        self.data, self.obj = torch.from_numpy(np.array(data, dtype=np.float32)), data

    def __enter__(self):
        return self

def make_meshgrid(x, y, h=.01):
    x_min, x_max = x.min() - 2, x.max() + 2
    y_min, y_max = y.min() - 5, y.max() + 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, cmap, svm=None, learner=None, pca: PCA=None, axises=(0,1), **params):
    xx, yy = make_meshgrid(np.array([ax.get_xlim()[0], ax.get_xlim()[1]]), np.array([ax.get_ylim()[0], ax.get_ylim()[1]]))

    if pca is None:
        coords = np.c_[xx.ravel(), yy.ravel()]
    else:
        # zero pad xx, yy
        pca_coords = np.zeros((xx.ravel().shape[0], pca.components_.shape[0]))
        pca_coords[:, axises[0]] = xx.ravel()
        pca_coords[:, axises[1]] = yy.ravel()
        coords = pca.inverse_transform(pca_coords)

    if svm is not None:
        Z = svm.predict(coords)

        if issparse(Z):
            Z = Z.todense()
    elif learner is not None:
        preds = torch.sigmoid(learner.model.forward(torch.from_numpy(coords).float()))
        pred_classes = preds > 0.3
        Z = pred_classes

    if Z.shape[1] > 1:
        # create discrete integer values from predicted labels

       Z = discretize(Z) * cmap.N // Z.shape[1]
    Z = Z.reshape(xx.shape)

    out = ax.contourf(xx, yy, Z, cmap=cmap, **params)
    return out

def discretize(Z):
    Z_discrete = np.full((Z.shape[0], 1), fill_value=Z.shape[1]+1, dtype=np.float)
    label_count = np.zeros_like(Z_discrete, dtype=np.int)

    for i in range(Z.shape[1]):
        # get index of elements labeled in i
        index = np.array(Z[:, i], np.bool).reshape(Z_discrete.shape)

        # get index of labels with preexisting labels
        prev_labeled = np.array(label_count>0, np.bool).reshape(Z_discrete.shape)

        # init or add label value
        Z_discrete[~prev_labeled & index] = i+1
        Z_discrete[prev_labeled & index] += i+1

        label_count[index] += 1

    # pick value for multilabeled examples:
    # value in middle of two classes for examples with two labels
    Z_discrete[label_count==2] /= 2
    if np.max(label_count) > 2:
        #Z_discrete[label_count>2] = (label_count[label_count>2]-3)/(np.max(label_count)-2)
        Z_discrete[label_count > 2] /= label_count[label_count>2]
    return Z_discrete