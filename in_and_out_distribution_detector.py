import os

import numpy as np
import faiss
from utils import get_scores_one_cluster
from tqdm import tqdm
import torch

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def get_scores(ftrain, ftest, num_cluster):
    if num_cluster == 1:
        return get_scores_one_cluster(ftrain, ftest)
    else:
        ypred = get_clusters(ftrain, num_cluster)
        return get_scores_multi_cluster(ftrain, ftest, ypred)


def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    ftrain = ftrain.astype(np.float32)
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred


def get_scores_multi_cluster(ftrain, ftest, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    dtest = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    dtest = np.min(dtest, axis=0)

    return dtest


def get_eval_results(ftrain, ftest, num_cluster):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)

    dtest = get_scores(ftrain, ftest, num_cluster)

    dtest = normalization(dtest)

    return dtest


def ssd_select_samples_out_distribution(neg_feats_training, samples, min_num=576, num_cluster=9):
    samples_ = samples

    pos_score_thisbag = get_eval_results(
        np.copy(neg_feats_training),
        np.copy(samples_),
        num_cluster)

    return pos_score_thisbag

def ssd_select_samples_in_distribution(neg_feats_training, samples, min_num=576, num_cluster=1):
    samples_ = samples

    pos_score_thisbag = get_eval_results(
        np.copy(neg_feats_training),
        np.copy(samples_),
        num_cluster)

    return pos_score_thisbag





