# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch


def average_performance(pred, target, thr=None, k=None):
    """Calculate CP, CR, CF1, OP, OR, OF1, where C stands for per-class
    average, O stands for overall average, P stands for precision, R stands for
    recall and F1 stands for F1-score.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
        thr (float): The confidence threshold. Defaults to None.
        k (int): Top-k performance. Note that if thr and k are both given, k
            will be ignored. Defaults to None.

    Returns:
        tuple: (CP, CR, CF1, OP, OR, OF1)
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')
    if thr is None and k is None:
        thr = 0.5
        warnings.warn('Neither thr nor k is given, set thr as 0.5 by '
                      'default.')
    elif thr is not None and k is not None:
        warnings.warn('Both thr and k are given, use threshold in favor of '
                      'top-k.')

    assert pred.shape == \
           target.shape, 'pred and target should be in the same shape.'

    # 计算每个类的AUC值，并画出ROC曲线，最后平均AUC
    # from sklearn.metrics import multilabel_confusion_matrix, roc_curve, auc
    # from sklearn.metrics import roc_auc_score
    # import os
    # # 计算每个标签的混淆矩阵
    # confusion_matrices = multilabel_confusion_matrix(target, pred > 0.5)  # y_true, y_pred
    # 计算每个标签的 ROC 曲线和 AUC 值
    # 打印每个标签的 AUC 值

    # chest11
    # labels = ['ETT-Abnormal', 'ETT-Borderline', 'ETT-Normal', 'NGT-Abnormal', 'NGT-Borderline',
    #           'NGT-Incompletely Imaged', 'NGT-Normal', 'CVC-Abnormal', 'CVC-Borderline', 'CVC-Normal',
    #           'Swan Ganz Catheter Present']

    # chest14
    # labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    #           'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    #           'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax']

    # tcmc tongue
    # labels = ['Qixu', 'Qiyu', 'Shire', 'Tanshi', 'Tebing', 'Xueyu', 'Yinxu']

    # import os
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()

    # # fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), pred.ravel())
    # # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # # # 直接调用函数计算
    # # micro_auc = roc_auc_score(target, pred, average='micro')
    #
    # fprs, tprs, aucs = [], [], []
    # for i in range(target.shape[1]):
    #     fpr, tpr, thresholds = roc_curve(target[:, i], pred[:, i])
    #     roc_auc = auc(fpr, tpr)
    # #     # auc = roc_auc_score(target[:, i], pred[:, i])
    # #     # print(f'{labels[i]}: AUC = {auc:.4f}')
    #     fprs.append(fpr)
    #     tprs.append(tpr)
    #     aucs.append(roc_auc)

    # 保存fprs[i]为npy文件
    # np.save(f'fprs_{i}.npy', fprs[i])
    # 保存fprs和tprs为npy文件
    #
    # save_root = '/home/dell/桌面/wuxin/project/mmclassification-master/tools/visualizations/huiyi_PRCV/ablation_chest11'
    # net_name = 'Model 5'
    # net = os.path.join(save_root, net_name)
    # np.save(net+'/fprs.npy', np.array(fprs))
    # np.save(net+'/tprs.npy', np.array(tprs))
    # np.save(net+'/roc_auc.npy', np.array(aucs))

    # Compute micro-average ROC curve and ROC area
    # 微平均方式计算TPR/FPR，最后得到AUC

    # for i in range(target.shape[1]):
    #     auc = roc_auc_score(target[:, i], pred[:, i])
    #     print(f'{labels[i]}: AUC = {auc:.4f}')

    # print()
    # total_auc = 0.
    # for i, auc in enumerate(aucs):
    #     total_auc += auc
    #     print(f'{labels[i]}: AUC = {auc:.4f}')
    # mean_auc = total_auc / len(labels)
    # print(f'mean_auc = {mean_auc:.4f}')

    # 可以使用 Matplotlib 画出 ROC 曲线
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    #
    # for i in range(target.shape[1]):
    #     plt.plot(fprs[i], tprs[i], label=f'{labels[i]}, AUC = {aucs[i]:.4f}')
    # plt.plot([0, 1], [0, 1], linestyle='--', label='Random guessing')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('decoder_C2T_T2C_rep_method2-att-sigmoid_softmax Chest11 ROC curve')
    # plt.legend()
    # plt.show()

    eps = np.finfo(np.float32).eps
    target[target == -1] = 0
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

    tp = (pos_inds * target) == 1  # True Positive, 被判定为正样本，事实上也是证样本
    fp = (pos_inds * (1 - target)) == 1  # False Positive, 被判定为正样本，但事实上是负样本。
    fn = ((1 - pos_inds) * target) == 1  # False Negative,被判定为负样本，但事实上是正样本

    precision_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fp.sum(axis=0), eps)  # np.sum(a, axis=0)  列求和;  np.sum(a, axis=1)  行求和
    recall_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fn.sum(axis=0), eps)
    CP = precision_class.mean() * 100.0  # 平均准确率
    CR = recall_class.mean() * 100.0  # 平均召回率
    CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)  # maximum 查找数组元素的逐元素最大值
    OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
    OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
    OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)

    """
    Multilabel case:

    >>> import numpy as np
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> X, y = make_multilabel_classification(random_state=0)
    >>> clf = MultiOutputClassifier(clf).fit(X, y)
    >>> # get a list of n_output containing probability arrays of shape
    >>> # (n_samples, n_classes)
    >>> y_pred = clf.predict_proba(X)
    >>> # extract the positive columns for each output
    >>> y_pred = np.transpose([pred[:, 1] for pred in y_pred])
    >>> roc_auc_score(y, y_pred, average=None)
    array([0.82..., 0.86..., 0.94..., 0.85... , 0.94...])
    >>> from sklearn.linear_model import RidgeClassifierCV
    >>> clf = RidgeClassifierCV().fit(X, y)
    >>> roc_auc_score(y, clf.decision_function(X), average=None)
    array([0.81..., 0.84... , 0.93..., 0.87..., 0.94...])
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import hamming_loss
    from sklearn.metrics import zero_one_loss
    # multi_auc = roc_auc_score(target, pred, average=None)
    # 初始化多标签AUC值
    # 初始化多标签AUC值
    total_auc = 0.

    # 计算每个标签的AUC值，并对所有标签的AUC值求平均
    for i in range(target.shape[1]):
        try:
            auc = roc_auc_score(target[:, i], pred[:, i], average=None) * 100
        except ValueError:
            auc = 0.5 * 100
        total_auc += auc
    multi_auc = total_auc / target.shape[1]
    # total_auc = 0.
    # total_h_loss = 0.
    # total_z_o_loss = 0.
    # # 初始化多标签AUC值
    # # 计算每个标签的AUC值，并对所有标签的AUC值求平均
    # for i in range(target.shape[1]):
    #     try:
    #         auc = roc_auc_score(target[:, i], pred[:, i])
    #         h_loss = hamming_loss(target[:, i], pred[:, i])  # hamming loss越小越好
    #         z_o_loss = zero_one_loss(target[:, i], pred[:, i])  # 在多标签分类中，zero_one_loss函数对应于子集零一损失：对于每个样本，必须正确预测整个标签集合，否则该样本的损失等于一
    #     except ValueError:
    #         auc = 0.5
    #         h_loss = 1.
    #         z_o_loss = 1.
    #     total_auc += auc
    #     total_h_loss += h_loss
    #     total_z_o_loss += z_o_loss
    # multi_auc = total_auc / target.shape[1]
    # avg_hamming_loss = total_h_loss / target.shape[1]
    # avg_zero_one_loss = total_z_o_loss / target.shape[1]
    # from sklearn.metrics import hamming_loss
    # from sklearn.metrics import zero_one_loss
    # avg_hamming_loss = hamming_loss(target, pred)  # hamming loss越小越好
    # avg_zero_one_loss = zero_one_loss(target, pred)  # 在多标签分类中，zero_one_loss函数对应于子集零一损失：对于每个样本，必须正确预测整个标签集合，否则该样本的损失等于一
    # return CP, CR, CF1, OP, OR, OF1, multi_auc, avg_hamming_loss, avg_zero_one_loss
    return CP, CR, CF1, OP, OR, OF1, multi_auc


# from sklearn.metrics import multilabel_confusion_matrix
# multilabel_confusion_matrix(y_true, y_pred)
#
#
from sklearn.metrics import roc_auc_score, hamming_loss, zero_one_loss


def add_multi_label_auc(pred, target, thr=None, k=None):
    # 将标签转换为numpy数组
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    # 初始化多标签AUC值
    total_auc = 0.

    # 计算每个标签的AUC值，并对所有标签的AUC值求平均
    for i in range(target.shape[1]):
        try:
            auc = roc_auc_score(target[:, i], pred[:, i])
        except ValueError:
            auc = 0.5
        total_auc += auc
    multi_auc = total_auc / target.shape[1]
    return multi_auc

#
# def add_evalution(pred, target, thr=None, k=None):
#     if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
#         pred = pred.detach().cpu().numpy()
#         target = target.detach().cpu().numpy()
#     elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
#         raise TypeError('pred and target should both be torch.Tensor or'
#                         'np.ndarray')
#     if thr is None and k is None:
#         thr = 0.5
#         warnings.warn('Neither thr nor k is given, set thr as 0.5 by '
#                       'default.')
#     elif thr is not None and k is not None:
#         warnings.warn('Both thr and k are given, use threshold in favor of '
#                       'top-k.')
#     h_loss = hamming_loss(target, pred)  # hamming loss越小越好
#     # z_o_loss = zero_one_loss(target, pred)  # 在多标签分类中，zero_one_loss函数对应于子集零一损失：对于每个样本，必须正确预测整个标签集合，否则该样本的损失等于一
#     # return h_loss, z_o_loss
#     return h_loss
