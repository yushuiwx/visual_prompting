# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision import datasets
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.tools import *
import models_mae
from engine_pretrain import train_one_epoch, validate
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter 
from omegaconf import DictConfig, OmegaConf
import logging
import yaml
from datasets.merge_datasets import MergedImageFolder
from datasets.grid_datasets import *
import random

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--break_after_epoch', type=int, metavar='N', help='break training after X epochs, to tune hyperparams and avoid messing with training schedule')

    # Dataset parameters
    # parser.add_argument('--imagenet_percent', default=0.5, type=float)
    parser.add_argument('--subsample', action='store_true')
    parser.set_defaults(subsample=False)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # new add
    parser.add_argument('--val_freq', default=100, type=int)
    parser.add_argument('--eval_data_dir', default='./',
                        help='path where to save data')
    parser.add_argument('--yaml_path', default='./',type=str)
    parser.add_argument('--min_random_scale', default=0.2, type=float)
    parser.add_argument('--in_context_pairs_number', default=3, type=int, 
                        help='random sample 2 ~ in_context_pairs_number for in-context learning')
    # parser.add_argument('--avg_percent', action='store_true', help='avg sample all datasets if true')
    parser.add_argument('--save_validate_image_results', action='store_true')
    # parser.add_argument('--vqgan_cpkt',type=str)
    # parser.add_argument('--vqgan_yaml',type=str)
    return parser

def main(args, cfg):
    args.second_input_size = 224
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if args.distributed and dist.get_rank() == 0:
        print("=" * 50, "Datasets configs", "=" * 50)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 116)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation for training data
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(args.min_random_scale, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transforms_validate = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.9999, 1.0), interpolation=3),
        transforms.ToTensor()
    ])


    train_datasets_list = []
    validate_dataset_list = {}
    # supervised datasets
    if 'append_supervised' in cfg.datasets:
        supervised_train_dataset = PairDataset_train(args, cfg)
        if supervised_train_dataset.__len__() != 0:
            train_datasets_list.append(supervised_train_dataset)

    # IamgeNet datasets
    if 'imageNet' in cfg.datasets:
        ImageNet_train_dataset = datasets.ImageFolder(os.path.join(cfg.datasets.imageNet.image_path, 'train'), transform=transforms_train)
        train_datasets_list.append(ImageNet_train_dataset)

    # CVF datasets
    if 'cvf' in cfg.datasets:
        CVF_train_dataset = datasets.ImageFolder(os.path.join(cfg.datasets.cvf.image_path, 'train'), transform=transforms_train)
        train_datasets_list.append(CVF_train_dataset)
    
    # validate datasets
    if args.distributed and dist.get_rank() == 0:
        if 'cvf' in cfg.datasets:
            CVF_validate_dataset = datasets.ImageFolder(os.path.join(cfg.datasets.cvf.image_path, 'val'), transform=transforms_validate)
            validate_dataset_list['CVF'] = CVF_validate_dataset
        
        if 'append_supervised' in cfg.datasets:
            json_path = cfg.datasets.append_supervised.json_path
            for task in json_path.keys():
                for task_data in json_path[task]:
                    if "val_json" in task_data.keys():
                        print("====> Loading from ", task_data.val_json, "for validating.")
                        sub_validate_datasets = PairDataset_validate_for_single_task(args, cfg, task_data.val_json)
                        validate_dataset_list[task] = sub_validate_datasets

        if 'imageNet' in cfg.datasets:
            ImageNet_validate_dataset = datasets.ImageFolder(os.path.join(cfg.datasets.imageNet.image_path, 'val'), transform=transforms_validate)
            validate_dataset_list['ImageNet'] = ImageNet_validate_dataset


    # sample some training data for visualization and check
    # if args.distributed and dist.get_rank() == 0 and supervised_train_dataset.__len__() != 0::
    #     supervised_samples_train_dir = create_directory_if_not_exists(os.path.join(args.output_dir, 'training_samples_visualization', 'supervised', 'train'))
    #     supervised_samples_validate_dir = create_directory_if_not_exists(os.path.join(args.output_dir, 'training_samples_visualization', 'supervised', 'validate'))
    #     CVF_samples_train_dir = create_directory_if_not_exists(os.path.join(args.output_dir, 'training_samples_visualization', 'CVF', 'train'))
    #     CVF_samples_validate_dir = create_directory_if_not_exists(os.path.join(args.output_dir, 'training_samples_visualization', 'CVF', 'validate'))
        
    #     for idx in range(20):
    #         if 'append_supervised' in cfg.datasets: 
    #             save_normalized_tensor_as_rgb_image(supervised_train_dataset.__getitem__(random.randint(0, supervised_train_dataset.__len__()) - 1)[0], os.path.join(supervised_samples_train_dir, str(idx) + '.png'))
    #             #save_normalized_tensor_as_rgb_image(supervised_validate_dataset.__getitem__(random.randint(0, supervised_validate_dataset.__len__()) -1)[0]['grid'], os.path.join(supervised_samples_validate_dir, str(idx) + '.png'))
    #             for task_type in validate_dataset_list.keys():
    #                 if task_type not in ['ImageNet', 'CVF']:
    #                     save_normalized_tensor_as_rgb_image(validate_dataset_list[task_type].__getitem__(random.randint(0, validate_dataset_list[task_type].__len__()) -1)[0]['grid'], \
    #                                                      os.path.join(supervised_samples_validate_dir, task_type + '_' + str(idx) + '.png'))
    #         if 'cvf' in cfg.datasets:
    #             save_normalized_tensor_as_rgb_image(CVF_train_dataset[random.randint(0, CVF_train_dataset.__len__()) - 1][0], os.path.join(CVF_samples_train_dir, str(idx) + '.png'))
    #             save_normalized_tensor_as_rgb_image(CVF_validate_dataset[random.randint(0, CVF_validate_dataset.__len__()) - 1][0], os.path.join(CVF_samples_validate_dir, str(idx) + '.png'))

    # merge all datasets into one
    if args.distributed and dist.get_rank() == 0:
        print("we will sample training datas from", train_datasets_list)
    final_train_dataset = MergedImageFolder(datasets_list = train_datasets_list)


    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            final_train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(final_train_dataset)
    
    # if args.distributed and dist.get_rank() == 0:
    #     sampler_val = torch.utils.data.SequentialSampler(CVF_validate_dataset)

    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        final_train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.distributed and dist.get_rank() == 0:
        data_loader_val_dict = {}
        for key in validate_dataset_list.keys():
            if key in ['ImageNet', 'CVF']:
                sub_validate_dataset = validate_dataset_list[key]
                sub_validate_dataset_sampler = torch.utils.data.SequentialSampler(sub_validate_dataset)
                
                data_loader_val_dict[key] = torch.utils.data.DataLoader(
                    sub_validate_dataset, sampler=sub_validate_dataset_sampler,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                )
            else:
                data_loader_val_dict[key] = validate_dataset_list[key]

    # define the model
    model = models_mae.__dict__[args.model]()

    model.to(device)
    epoch_size = len(final_train_dataset)
    print(f'epoch_size is {epoch_size}')
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    base_lr = (args.lr * 256 / eff_batch_size)
    print("base lr: %.2e" % base_lr)
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    for k, v in model_without_ddp.named_parameters():
        if 'vae' in k:
            v.requires_grad = False

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # making tensorboard 
    args.tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
    if args.distributed and dist.get_rank() == 0:
        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir, exist_ok=True)
  
        writer = SummaryWriter(args.tensorboard_dir)
    else:
        writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if (epoch % args.val_freq == 0 or epoch + 1 == args.epochs) and (args.distributed and dist.get_rank() == 0):
            # validating
            validate(model, data_loader_val_dict, device, epoch, args, cfg=cfg, writer=writer)

        train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            epoch_size=epoch_size // eff_batch_size,
            writer=writer
        )
        if args.output_dir and (epoch % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if misc.is_main_process():
        run.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.yaml_path is None:
        print("Please provide data_dir paths in `configs/prjpaths/default.yaml`")
        assert 0
    
    # loading the yamls to get dataset paths and output dir
    with open(args.yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        cfg = DictConfig(config_dict)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, cfg)
