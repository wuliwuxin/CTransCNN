import mmcv
import numpy as np
import warnings
import torch
from .builder import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class My_MltilabelData(MultiLabelDataset):

    def __init__(self, **kwargs):
        super(My_MltilabelData, self).__init__(**kwargs)

    def load_annotations(self):
        """Load annotations.

        Returns:
            list[dict]
        """
        data_infos = []

        lines = mmcv.list_from_file(self.ann_file) 
        for line in lines:
            imgrelativefile, imglabel = line.strip().rsplit('\t', 1)
            gt_label = np.asarray(list(map(int, imglabel.split(","))), dtype=np.int8)
            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=imgrelativefile),
                gt_label=gt_label.astype(np.int8))
            data_infos.append(info)
        return data_infos

    def calculate_class_acc(pred, target, thr=None, k=None):
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

        tp = (pos_inds * target) == 1
        fp = (pos_inds * (1 - target)) == 1
        fn = ((1 - pos_inds) * target) == 1
        tn = ((1 - pos_inds) * (1 - target)) == 1

        precision_class = tp.sum(axis=0) / np.maximum(
            tp.sum(axis=0) + fp.sum(axis=0), eps) * 100.0
        recall_class = tp.sum(axis=0) / np.maximum(
            tp.sum(axis=0) + fn.sum(axis=0), eps) * 100.0

        allClassIsPredictTrue_pictureFlag = np.all(tp + tn, axis=1)
        picture_acc = np.sum(allClassIsPredictTrue_pictureFlag) / len(allClassIsPredictTrue_pictureFlag) * 100.0
        return precision_class.tolist(), recall_class.tolist(), picture_acc

    def evaluate(self, results, metric='mAP', metric_options=None, indices=None, logger=None):
        eval_results = super().evaluate(results, metric, metric_options, indices, logger)

        # results and gt_labels
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should ' \
                                           'be of the same length as gt_labels.'

        precision_class, recall_class, picture_acc = self.calculate_class_acc(results, gt_labels, thr=0.5)
        assert len(precision_class) == len(recall_class) == len(self.CLASSES), "必然长度一样"
        precision_class_topk, recall_class_topk, picture_acc_topk = self.calculate_class_acc(results, gt_labels, k=1)
        assert len(precision_class) == len(recall_class) == len(self.CLASSES), "必然长度一样"

        eval_results["picture_acc_thr"] = picture_acc
        eval_results["picture_acc_top1"] = picture_acc_topk
        for index, (classname, precision) in enumerate(zip(self.CLASSES, precision_class)):
            eval_results[classname + "_precision_thr"] = precision
        for index, (classname, recall) in enumerate(zip(self.CLASSES, recall_class)):
            eval_results[classname + "_recall_thr"] = recall

        for index, (classname, precision) in enumerate(zip(self.CLASSES, precision_class_topk)):
            eval_results[classname + "_precision_top1"] = precision
        for index, (classname, recall) in enumerate(zip(self.CLASSES, recall_class_topk)):
            eval_results[classname + "_recall_top1"] = recall

        for k, v in eval_results.items():
            print(str(k).ljust(len("10000temporaryPictures_healthCode_recall_thr") + 10), ": ", str(v).ljust(30))
        return eval_results
