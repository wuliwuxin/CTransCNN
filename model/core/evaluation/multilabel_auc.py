import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings


def add_multi_label_auc(pred, target, thr=None, k=None):
    if thr is None and k is None:
        thr = 0.5
        warnings.warn('Neither thr nor k is given, set thr as 0.5 by '
                      'default.')
    elif thr is not None and k is not None:
        warnings.warn('Both thr and k are given, use threshold in favor of '
                      'top-k.')

    if thr is not None:
        # a label is predicted positive if the confidence is no lower than thr
        pos_inds = pred >= thr

    else:
        # top-k labels will be predicted positive for any example
        sort_inds = np.argsort(-pred, axis=1)
        sort_inds_ = sort_inds[:, :k]
        inds = np.indices(sort_inds_.shape)
        pos_inds = np.zeros_like(pred)
        pos_inds[inds[0], sort_inds_] = 1

    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    total_auc = 0.

    for i in range(target.shape[1]):
        try:
            auc = roc_auc_score(target[:, i], pred[:, i])
        except ValueError:
            auc = 0.5
        total_auc += auc
    multi_auc = total_auc / target.shape[1]
    return multi_auc
