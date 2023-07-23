import sys
import os
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
# from evaluate.ADE20kSemSegEvaluatorCustom import * 

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
    canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return np.uint8(canvas), np.uint8(im_paste)


def evaluate_semantic_segmentation_in_training(args, model, writer, epoch, ds):

    eval_dict = {'mIoU': 0, 'fwIoU': 0, 'mACC': 0, 'pACC': 0}

    if args.save_validate_image_results:
        sample_freq = max(int(len(ds) // 20), 1)
        save_dir = os.path.join(args.output_dir, 'validate_results_visualization', 'epoch' + str(epoch), 'semantic_segmentation')
        os.makedirs(save_dir, exist_ok=True)

    PALETTE = define_colors_per_location_mean_sep()
    dataset_name = 'ade20k_sem_seg_val'
    evaluator = SemSegEvaluatorCustom(
        dataset_name,
        distributed=False,
        output_dir=output_folder,
        palette=PALETTE,
        pred_dir=pred_dir,
        dist_type=args.dist_type,
    )
        
    for idx in trange(len(ds)):
        canvas = ds[idx][0]['grid']
        original_image, generated_result = _generate_result_for_canvas(args, model, canvas)

        if args.save_validate_image_results and (idx + 1) // sample_freq == 0:
            Image.fromarray(np.uint8(original_image)).save(
                os.path.join(save_dir, f'original_{idx}.png'))
            Image.fromarray(np.uint8(generated_result)).save(
                os.path.join(save_dir, f'generated_{idx}.png'))

        current_metric = calculate_metric(args, original_image, generated_result)
        for i, j in current_metric.items():
            eval_dict[i] += (j / len(ds))


    
    writer.add_scalar('Deraining-MSE', eval_dict['mse'], global_step=epoch, walltime=None)


if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
