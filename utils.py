import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp

def convert_one_hot(label):
    x = np.zeros((label.size, label.max() + 1))
    x[np.arange(label.size), label] = 1
    return x

def get_scores_one_cluster(ftrain, ftest, shrunkcov=False):
    if shrunkcov:
        print("Using ledoit-wolf covariance estimator.")
        cov = lambda x: ledoit_wolf(x)[0]
    else:
        cov = lambda x: np.cov(x.T, bias=True)

    dtest = np.sum(
        (ftest - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    return dtest
def compute_multiclass_auc(y_score,y_label,n_classes=3):

    y_label = convert_one_hot(y_label)

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc


if __name__ == "__main__":

    y_label = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    y_score = np.array([
        [0.8, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.1, 0.3],
        [0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.2, 0.75, 0.05],
        [0.05, 0.05, 0.9], [0.1, 0.3, 0.6], [0.12, 0.8, 0.08],
    ])
    roc_auc = compute_multiclass_auc(y_score,y_label)

    print()
