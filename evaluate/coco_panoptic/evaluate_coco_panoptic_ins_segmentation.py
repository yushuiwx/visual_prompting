import sys
import os
sys.path.append('../../')
sys.path.append('../')
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))
import os.path

import torchvision
from tqdm import trange
from evaluate.reasoning_dataloader import *
from evaluate.mae_utils import *
import argparse
from pathlib import Path
from tqdm import tqdm
from tqdm import tqdm
import torch.distributed as dist
from evaluate.segmentation_utils import *

def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--data_path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tta_option', default=0, type=int)
    parser.add_argument('--ckpt', help='resume from checkpoint')

    parser.set_defaults(autoregressive=False)
    return parser


def _generate_result_for_canvas(args, model, canvas):
    """canvas is already in the right range."""
    ids_shuffle, len_keep = generate_mask_for_evaluation()
    _, im_paste, _ = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device)
    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas.cpu().detach()) * 255, 0, 255).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return np.uint8(canvas), np.uint8(im_paste)


def evaluate_coco_panoptic_segmentation_in_training(args, cfg, model, writer, epoch, ds):
    padding = ds.__get_padding__()
    if args.save_validate_image_results:
        sample_freq = max(int(len(ds) // 20), 1)
        save_dir = os.path.join(args.output_dir, 'validate_results_visualization', 'epoch' + str(epoch), 'semantic_segmentation_grid')
        # compute_save_dir = os.path.join(args.output_dir, 'validate_results_visualization', 'epoch' + str(epoch), 'semantic_segmentation_single')
        os.makedirs(save_dir, exist_ok=True)
        # os.makedirs(compute_save_dir, exist_ok=True)

    PALETTE = define_colors_per_location_mean_sep()
    dataset_name = 'ade20k_sem_seg_val'
    gt_path = os.path.join(cfg.datasets.append_supervised.root_path, 'ADEChallengeData2016/annotations_detectron2/validation')
    evaluator = ADE20k_SemSegEvaluatorCustom(
        input_size=args.input_size,
        dataset_name=dataset_name,
        distributed=False,
        gt_path=gt_path,
        palette=PALETTE,
        dist_type='abs',
    )
    process_list = []
    for idx in trange(len(ds)):
        canvas = ds[idx][0]['grid']
        original_image, generated_result = _generate_result_for_canvas(args, model, canvas)
        if args.save_validate_image_results and (idx + 1) // sample_freq == 0:
            Image.fromarray(np.uint8(original_image)).save(
                os.path.join(save_dir, 'original_' + ds[idx][0]['filename']))
            Image.fromarray(np.uint8(generated_result)).save(
                os.path.join(save_dir, ds[idx][0]['filename']))

        crop_size = args.input_size // 2
        assert generated_result.shape[-1] == 3
        pred = generated_result[- crop_size:, - crop_size:, :] # h, w, c
        process_list.append({"pred": pred, 'filename': ds[idx][0]['filename']})
    
    evaluator.reset()
    evaluator.process(process_list)
    results = evaluator.evaluate()

    copy_paste_results = {}
    for key in ['mIoU', 'fwIoU', 'mACC', 'pACC']:
        copy_paste_results[key] = results['sem_seg'][key]
        copy_paste_results[key] = results['sem_seg'][key]
    print("Semantic-Segmentation evaluating:", copy_paste_results)
    writer.add_scalar('Semantic-Segmentation-mIoU', copy_paste_results['mIoU'], global_step=epoch, walltime=None)
    writer.add_scalar('Semantic-Segmentation-fwIoU', copy_paste_results['fwIoU'], global_step=epoch, walltime=None)
    writer.add_scalar('Semantic-Segmentation-pACC', copy_paste_results['pACC'], global_step=epoch, walltime=None)
    writer.add_scalar('Semantic-Segmentation-mACC', copy_paste_results['mACC'], global_step=epoch, walltime=None)


if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
