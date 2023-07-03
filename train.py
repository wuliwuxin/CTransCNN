# Copyright (c) OpenMMLab. All rights reserved.
"""
在mmcls目录下，mmclassification为我们收录了常用的模型、预处理方法、数据集读取方法、训练参数。为了灵活调试各种参数，我们需要深入到mmcls目录中。
mmcls结构:
1. mmcls第一级目录下主要有apis, core, datasets, models, utils五个模块。
2. apis：封装了训练、测试、推理的过程
3. core：提供了一些工具，fp16，评判标准
4. datasets：提供了数据加载接口、数据增强方法
    1. 在这一层的py文件，用于读取数据
    2. pipline目录，该目录下收集了常用的数据增强方法
    3. samplers目录，采样方式
5. models：提供了常用的backbone、head、loss、necks……
    1. ackbones目录，集成了常见的主干网络
    2. heads目录，提供了多种分类的输出方式
    3. necks目录，数据backbone与head之间，更好的利用backbone提供的特征，收录了GAP
    4. losses目录，提供了常见的损失函数
    5. classifiers目录，将backbone、necks、head、losses集成起来
    6. utils目录，提供了部分模型封装好的模块
6. utils：辅助功能  日志模块在这个部分
"""

"""
多标签分类tips
相信大家在做项目时都会遇到数据量不足，数据样本分布不均等问题。下面我结合自己的经验给大家提供一些tips
1.数据量有限时（train数据少于1w张时），小模型的效果会由于大模型（过拟合）。
2.使用ClassBalancedDataset，对样本少的类别重复采样，可以有效提高召回。
3.loss，我试验过focal loss，asymmetric loss，效果很差（召回很高，但是准确率低得离谱）。focal与 asymmetric loss都是为了降低负样本在loss的占比，从而平衡正负样本。
asymmetric将gamma分为gamma+与gamma-，相较于focal loss中gamma=2来说，asymmetric更加夸张的降低了负样本的loss（因为我的数据类别分布极不均匀，在实验中，
正样本loss大概是负样本loss的几十倍甚至几百倍），这就导致模型极度聚焦于正样本，模型只知道什么是正确的，对错误的并不care，因此recall提高了，precision却低的离谱。
focal loss也存在这样的问题，但是alpha可以稍微缓解这样的情况。
4.魔改模型结构，注意力机制，dropout，框架等等各种。
"""
import argparse
import copy
import os
import os.path as osp
import time
import warnings
warnings.filterwarnings('ignore')
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from model import __version__
from model.apis import init_random_seed, set_random_seed, train_model
from model.datasets import build_dataset
from model.models import build_classifier
from model.utils import (auto_select_device, collect_env, get_root_logger,
                         setup_multi_processes)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--device', help='device used for training. (Deprecated)')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,

        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--ipu-replicas',
        type=int,
        default=None,
        help='num of ipu replicas to use')

    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.ipu_replicas is not None:
        cfg.ipu_replicas = args.ipu_replicas
        args.device = 'ipu'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    cfg.device = args.device or auto_select_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_classifier(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # save mmcls version, config file content and class names in
    # runner as meta data
    meta.update(
        dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES))

    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        device=cfg.device,
        meta=meta)


if __name__ == '__main__':
    main()
