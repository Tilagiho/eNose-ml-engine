from fastai.torch_core import *
from fastai.callback import *
from fastai.callbacks import *
from fastai.basic_data import *
from fastai.basic_train import *

from sklearn.metrics import multilabel_confusion_matrix

from fastai.train import *


class MultiLabelClassificationInterpretation(Interpretation):
    "Interpretation methods fpr multi-label classification models."
    def __init__(self, learn:Learner, preds:Tensor, y_true:Tensor, losses:Tensor, ds_type:DatasetType=DatasetType.Valid):
        super().__init__(learn, preds, y_true, losses, ds_type)

        # # prepare predicted labels
        # apply threshold to predictions
        pred_classes = torch.zeros_like(preds, dtype=torch.bool)
        pred_classes[preds>0.3] = True

        # add column for no class being predicted
        no_class_predicted = ~pred_classes.any(dim=1)
        no_class_predicted.unsqueeze_(1)
        self.pred_classes = torch.cat([pred_classes, no_class_predicted], dim=1)

        # # prepare true classes
        y_true = self.y_true.bool()
        no_class_true = ~y_true.any(dim=1)
        no_class_true.unsqueeze_(1)
        self.true_classes = torch.cat([y_true, no_class_true], dim=1)

    def confusion_matrix(self, slice_size: int = None, target_type='single_class_labels'):
        "Confusion matrix as an `np.ndarray`. Only implemented for targets with exactly one label."
        if target_type != 'single_class_labels':
            # TODO: use sci-kit multilabel conf matrix
            raise NotImplementedError

        cm = torch.zeros([self.true_classes.shape[1], self.true_classes.shape[1]])

        if slice_size is None:
            for i in range(self.true_classes.shape[0]):
                c = self.true_classes[i].long().argmax(dim=0)

                for j in range(self.true_classes.shape[1]):
                    if i == c and self.pred_classes[c, c]:
                        cm[c, c] += 1
                    elif self.pred_classes[i, j]:
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
        tick_marks = np.arange(self.data.c + 1)     # c + 1, because multilabeled datasets don't include 'No Smell'
        plt.xticks(tick_marks, self.data.y.classes, rotation=90)
        plt.yticks(tick_marks, self.data.y.classes, rotation=0)

        if plot_txt:
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
                plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        ax = fig.gca()
        ax.set_ylim(len(self.data.y.classes) + 1 - .5, -.5) # len + 1, because multilabeled datasets don't include 'No Smell'

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
    _order = -20

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
    _order = -20

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