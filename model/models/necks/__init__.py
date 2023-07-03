# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .my_gap import My_GlobalAveragePooling

__all__ = ['GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales', 'My_GlobalAveragePooling']
