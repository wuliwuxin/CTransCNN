from .cls_head import ClsHead
from .linear_head import LinearClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead

from .my_hybird_head import My_Hybird_Head

__all__ = [
    'ClsHead', 'LinearClsHead', 'MultiLabelClsHead',
    'MultiLabelLinearClsHead',  'My_Hybird_Head',
]
