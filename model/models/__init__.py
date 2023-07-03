# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, CLASSIFIERS, HEADS, LOSSES, NECKS,
                      build_backbone, build_classifier, build_head, build_loss,
                      build_neck)
# noqa = NO-QA (无质量保证) 忽略特定的警告(PEP8 F401和F403已导入但未使用)
from .classifiers import *  # noqa: F401,F403
from .heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403

__all__ = [
    'BACKBONES', 'HEADS', 'NECKS', 'LOSSES', 'CLASSIFIERS', 'build_backbone',
    'build_head', 'build_neck', 'build_loss', 'build_classifier'
]
