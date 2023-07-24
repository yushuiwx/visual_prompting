# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import torch.distributed as dist

# from evaluate.evaluate_reasoning import *
# from evaluate.evaluate_colorization import *
# from evaluate.evaluate_segmentation import *
from evaluate import evaluate_reasoning
from evaluate import evaluate_colorization
from evaluate import evaluate_segmentation
from evaluate import evaluate_deraining
from evaluate import evaluate_depth
from evaluate import evaluate_light_enhance
from evaluate import evaluate_semantic_segmentation
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    writer=None,
                    args=None,
                    epoch_size=1):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    data_loader_i = iter(data_loader)
    for data_iter_step in metric_logger.log_every(range(epoch_size), print_freq, header):
        (batch) = next(data_loader_i)
        # we use a per iteration (instead of per epoch) lr scheduler
        if isinstance(batch, tuple):
            samples, visual_tokens = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = visual_tokens.to(device, non_blocking=True)
        else: # hack for consistency
            samples = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = samples

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            loss_dict, _, _ = model(samples, visual_tokens, mask_ratio=args.mask_ratio)

        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        # tnsorboard
        epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        if args.distributed and dist.get_rank() == 0:
            writer.add_scalar('Training CE-Loss', loss_value_reduce, global_step=epoch_1000x, walltime=None)
            writer.add_scalar('lr', lr, global_step=epoch_1000x, walltime=None)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model, data_loader_val_dict, device, epoch, args, cfg, writer):
    model.eval()
    print("=" * 50, 'Validation', '=' * 50)
    with torch.no_grad():
        # evaluate in CVF
        # CVF_validate_dataset = data_loader_val_dict['CVF']
        # validate_in_IamgeFolder(model, CVF_validate_dataset, device, epoch, args, writer, tag='CVF')

        # # evaluate in ImageNet
        # if 'ImageNet' in data_loader_val_dict.keys():
        #     ImageNet_validate_dataset = data_loader_val_dict['ImageNet']
        #     validate_in_IamgeFolder(model, ImageNet_validate_dataset, device, epoch, args, writer, tag='ImageNet')

        # # for reasoning
        # evaluate_reasoning.evaluate_reasoning_in_training(args, model.module, writer, epoch)

        # # for colorization
        # if 'colorization' in data_loader_val_dict.keys():
        #     evaluate_colorization.evaluate_colorization_in_training(args, model.module, writer, epoch)

        # # for foreground segmentation
        # evaluate_segmentation.evaluate_segmentation_in_training(args, model.module, writer, epoch)

        # # fro deraining
        # if 'deraining' in data_loader_val_dict.keys():
        #     evaluate_deraining.evaluate_deraining_in_training(args, model.module, writer, epoch, data_loader_val_dict['deraining'])

        # # for depth estimation
        # if 'depth_estimation' in data_loader_val_dict.keys():
        #     evaluate_depth.evaluate_depth_in_training(args, model.module, writer, epoch, data_loader_val_dict['depth_estimation'])

        # # for light enhance
        # if 'light_enhance' in data_loader_val_dict.keys():
        #     evaluate_light_enhance.evaluate_light_enhance_in_training(args, model.module, writer, epoch, data_loader_val_dict['light_enhance'])

        # # for semantic segmentation
        evaluate_semantic_segmentation.evaluate_semantic_segmentation_in_training(args, cfg, model.module, writer, epoch, data_loader_val_dict['semantic_segmentation'])


    print("=" * 112)
    model.train()


def validate_in_IamgeFolder(model, validate_dataset, device, epoch, args, writer, tag):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1e5
    for data_iter_step, (batch, _) in tqdm(enumerate(metric_logger.log_every(validate_dataset, print_freq, header))):
        samples = batch
        samples = samples.to(device, non_blocking=True)
        visual_tokens = samples

        with torch.cuda.amp.autocast():
            loss_dict, _, _ = model(samples, visual_tokens, mask_ratio=args.mask_ratio)

        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})
    print(tag + " val loss:", metric_logger)
    writer.add_scalar(tag + ' validate CE-Loss', metric_logger.meters['mae'].value, global_step=epoch, walltime=None)
