from fastai.torch_core import *
from fastai.callback import *
from fastai.callbacks import *
from fastai.basic_data import *
from fastai.basic_train import *

from sklearn.svm import SVC

from funcdataset import *

import pandas

#               #
class MultiLabelClassificationInterpretation:
    "Interpretation methods fpr multi-label classification models."
    def __init__(self, dataset:FuncDataset, learn:Learner=None, svm:SVC=None, thresh:float=0.3):
        # if interpretation of fastai Learner
        if learn is not None:
            preds, y, losses = learn.get_preds(with_loss=True)

            # # prepare predicted labels
            # apply threshold to predictions
            pred_classes = torch.zeros_like(preds, dtype=torch.bool)
            pred_classes[preds>thresh] = True

            # add column for no class being predicted
            no_class_predicted = ~pred_classes.any(dim=1)
            no_class_predicted.unsqueeze_(1)
            pred_classes = torch.cat([pred_classes, no_class_predicted], dim=1)

            # # prepare true classes
            y_true = y.bool()
            no_class_true = ~y_true.any(dim=1)
            no_class_true.unsqueeze_(1)
            y_true = torch.cat([y_true, no_class_true], dim=1)

        # if interpretation of support vector machine
        elif svm is not None:
            y_true = torch.zeros([dataset.test_classes.shape[0], dataset.c]).long()
            y_true.scatter_(1, dataset.test_classes.long().unsqueeze(dim=1), 1.)
            preds = torch.from_numpy(svm.predict_proba(dataset.test_set))
            pred_classes = torch.zeros_like(preds, dtype=torch.bool)
            pred_classes[preds>thresh] = True
            losses = None   # TODO
        else:
            raise ValueError("Either learn or svm has to be specified!")

        self.learn, self.y_true, self.losses, self.preds, self.pred_classses, self.dataset = learn, y_true, losses, preds, pred_classes, dataset

    def confusion_matrix(self, slice_size: int = None, target_type='single_class_labels'):
        "Confusion matrix as an `np.ndarray`. Only implemented for targets with exactly one label."
        if target_type != 'single_class_labels':
            # TODO: use sci-kit multilabel conf matrix
            raise NotImplementedError

        cm = torch.zeros([self.y_true.shape[1], self.y_true.shape[1]])

        if slice_size is None:
            for i in range(self.y_true.shape[0]):
                c = self.y_true[i].long().argmax(dim=0)

                for j in range(self.y_true.shape[1]):
                    if i == c and self.pred_classses[c, c]:
                        cm[c, c] += 1
                    elif self.pred_classses[i, j]:
                        cm[c, j] += 1
        else:
            raise NotImplementedError
        return to_np(cm)

    def plot_confusion_matrix(self, normalize: bool = False, title: str = 'Confusion matrix', cmap: Any = "Blues",
                              slice_size: int = None,
                              norm_dec: int = 2, plot_txt: bool = True, return_fig: bool = None, **kwargs) -> Optional[
        plt.Figure]:
        "Plot the confusion matrix, with `title` and using `cmap`."
        # This function is mainly copied from the sklearn docs
        cm = self.confusion_matrix(slice_size=slice_size)
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = plt.figure(**kwargs)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(self.dataset.c + 1)     # c + 1, because multilabeled datasets don't include 'No Smell'
        plt.xticks(tick_marks, self.dataset.y.classes, rotation=90)
        plt.yticks(tick_marks, self.dataset.y.classes, rotation=0)

        if plot_txt:
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
                plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        ax = fig.gca()
        ax.set_ylim(len(self.dataset.y.classes) + 1 - .5, -.5) # len + 1, because multilabeled datasets don't include 'No Smell'

        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)
        if ifnone(return_fig, defaults.return_fig): return fig


class MultiLabelExactMatch(Callback):
    "Wrap a `func` in a callback for metrics computation."
    def __init__(self, thresh=0.3, sigmoid=True):
        # If it's a partial, use func.func
        # self.name = 'exact_match'
        self.thresh, self.sigmoid = thresh, sigmoid

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.n_exact_matches, self.count = 0., 0.

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        pred, targ = ((last_output.sigmoid() if self.sigmoid else last_output) > self.thresh).byte(), last_target.byte()

        for i in range(targ.shape[0]):
            if (pred[i] == targ[i]).all():
                self.n_exact_matches += 1

        self.count += targ.shape[0]

        # if not is_listy(last_target): last_target = [last_target]
        # increment count of total examples
        # self.count += last_target[0].size(0)

        #
        # val = self.func(last_output, *last_target)
        # self.val += last_target[0].size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, torch.tensor(self.n_exact_matches / self.count))


class MultiLabelRecall(Callback):
    _order = -30

    def __init__(self, eps=1e-15, thresh=0.3, sigmoid=True, average="micro"):
        self.eps, self.thresh, self.sigmoid, self.average = eps, thresh, sigmoid, average

    def on_epoch_begin(self, **kwargs):
        self.tp, self.total_pred, self.total_targ = 0, 0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        pred, targ = ((last_output.sigmoid() if self.sigmoid else last_output) > self.thresh).byte(), last_target.byte()
        m = pred * targ
        self.tp += m.sum(0).float()
        self.total_pred += pred.sum(0).float()
        self.total_targ += targ.sum(0).float()

    def on_epoch_end(self, last_metrics, **kwargs):
        self.total_pred += self.eps
        self.total_targ += self.eps
        if self.average == "micro":
            recall = self.tp.sum() / self.total_targ.sum()
        elif self.average == "macro":
            recall = self.tp / self.total_targ.mean()
        else:
            raise Exception("Choose one of the average types: [micro, macro]")

        return add_metrics(last_metrics, recall)


class MultiLabelPrecision(Callback):
    _order = -25

    def __init__(self, eps=1e-15, thresh=0.3, sigmoid=True, average="micro"):
        self.eps, self.thresh, self.sigmoid, self.average= eps, thresh, sigmoid, average

    def on_epoch_begin(self, **kwargs):
        self.tp, self.total_pred, self.total_targ = 0, 0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        pred, targ = ((last_output.sigmoid() if self.sigmoid else last_output) > self.thresh).byte(), last_target.byte()
        m = pred * targ
        self.tp += m.sum(0).float()
        self.total_pred += pred.sum(0).float()
        self.total_targ += targ.sum(0).float()

    def on_epoch_end(self, last_metrics, **kwargs):
        self.total_pred += self.eps
        self.total_targ += self.eps
        if self.average == "micro":
            precision = self.tp.sum() / self.total_pred.sum()
        elif self.average == "macro":
            precision = self.tp / self.total_pred
        elif self.average == "weighted":
            precision = self.tp / self.total_pred

        else:
            raise Exception("Choose one of the average types: [micro, macro]")

        return add_metrics(last_metrics, precision)


class TrainAnalyser:
    def __init__(self, modelname: str):
        self.modelname = modelname
        self.df = pandas.read_csv('models/' + modelname + '.csv')

    def plot_metrics(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)

        fig.suptitle(self.modelname)

        # go through metrics
        for i in range(3, self.df.shape[1]-1):
            ax1.plot(self.df.iloc[:, i], label=self.df.columns[i])

        ax1.legend()
        ax1.set_ylim(0., 1.)

        for i in range(1, 3):
            ax2.plot(self.df.iloc[:, i], label=self.df.columns[i])

        ax2.legend()
        ax2.set_xlabel('epoch')

        return fig
