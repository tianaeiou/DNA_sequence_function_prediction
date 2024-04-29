import numpy as np
import torch
from sklearn import metrics


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, labels, id=None, phase=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError

    def confusion_matrix(self):
        raise NotImplementedError


class accuracy_score(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.y_pred = []
        self.y_true = []
        self.task_specific_metric = {}

    def __call__(self, outputs, labels, id=None, phase=None):
        # _, preds = torch.max(outputs, 1)
        preds = (outputs >= 0.5).float()
        self.phase = phase
        self.y_pred.append(preds.detach().cpu().numpy())
        self.y_true.append(labels.detach().cpu().numpy())
        return self.value()

    def reset(self):
        self.y_pred = []
        self.y_true = []
        self.task_specific_metric = {}

    def value(self):
        return float(sum(sum(np.concatenate(self.y_true) == np.concatenate(self.y_pred))) / (
                    np.concatenate(self.y_pred).shape[0] * np.concatenate(self.y_pred).shape[1]))

    def name(self):
        return 'Accuracy'

    def report(self):
        print(self.value)

