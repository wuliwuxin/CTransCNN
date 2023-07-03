import mmcv
import numpy as np

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
        # img_ids 是一个列表，每个元素是一个字符串，表示图片的名称
        lines = mmcv.list_from_file(self.ann_file)  # self.ann_file 是字符串，此文件中的每一行都是我们自己存的
        for line in lines:
            # imgrelativefile, imglabel = line.strip().rsplit('\t', 1)
            imgrelativefile, imglabel = line.strip().rsplit('\t', 1)
            gt_label = np.asarray(list(map(int, imglabel.split(","))), dtype=np.int8)
            # imgrelativefile = str(imgrelativefile)
            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=imgrelativefile),
                gt_label=gt_label.astype(np.int8))
            data_infos.append(info)
        return data_infos
