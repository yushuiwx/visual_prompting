import sys
import os
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))
import os.path
import torch.nn as nn
import torchvision
from tqdm import trange
from evaluate.derain_dataloader import DatasetDeraining
from evaluate.reasoning_dataloader import *
from evaluate.mae_utils import *
import argparse
from pathlib import Path
from tqdm import tqdm
from tqdm import tqdm
import torch.distributed as dist
# from main_pretrain import transforms_validate 
import yaml
from datasets.merge_datasets import MergedImageFolder
from datasets.grid_datasets import *
from vqgan import get_vq_model
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter  

def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--yaml_path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--ckpt', help='resume from checkpoint')

    parser.set_defaults(autoregressive=False)
    return parser


def _generate_result_for_canvas(args, model, canvas, vqvae):
    """canvas is already in the right range."""
    ids_shuffle, len_keep = generate_mask_for_evaluation()
    _, im_paste, L_mask, logits = generate_image_with_output_logits(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                    len_keep, device=args.device)
        # compute ce loss
    with torch.no_grad():
        target = vqvae.get_codebook_indices(canvas.unsqueeze(0).to(args.device)).flatten(1)
    # print(target.shape, ids_shuffle.shape, len_keep, L_mask.shape, logits.shape)
    loss = nn.CrossEntropyLoss(reduction='none')(input=logits.permute(0, 2, 1), target=target)
    loss = (loss * L_mask).sum() / L_mask.sum()
        
    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas.cpu().detach()) * 255, 0, 255).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)

    return np.uint8(canvas), np.uint8(im_paste), logits, loss


def calculate_metric(args, target, ours):
    ours = (np.transpose(ours/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
    target = (np.transpose(target/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

    target = target[:, 113:, 113:]
    ours = ours[:, 113:, 113:]
    mse = np.mean((target - ours)**2)
    return {'mse': mse}


def evaluate(yaml_path, model, vqvae):
    
    # preparing config
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        cfg = DictConfig(config_dict)

    args.in_context_pairs_number = 2
    args.input_size = 224
    args.min_random_scale = 0.2

    validate_dataset_list = {}
    json_path = cfg.datasets.append_supervised.json_path
    for task in json_path.keys():
        for task_data in json_path[task]:
            if "val_json" in task_data.keys():
                print("====> Loading from ", task_data.val_json, "for validating.")
                sub_validate_datasets = PairDataset_validate_for_single_task(args, cfg, task_data.val_json)
                validate_dataset_list[task] = sub_validate_datasets

    eval_ce_loss_dict = {}
    for task in validate_dataset_list.keys():
        eval_ce_loss = 0.
        ds = validate_dataset_list[task]
        for idx in trange(len(ds)):
            canvas = ds[idx][0]['grid']
            original_image, generated_result, logits, loss = _generate_result_for_canvas(args, model, canvas, vqvae)

            if args.output_dir and (idx + 1) // 50 == 0:
                Image.fromarray(np.uint8(original_image)).save(
                    os.path.join(args.output_dir, f'original_{idx}.png'))
                Image.fromarray(np.uint8(generated_result)).save(
                    os.path.join(args.output_dir, f'generated_{idx}.png'))

            eval_ce_loss += (loss / len(ds))
    eval_ce_loss_dict[task] = eval_ce_loss
    return eval_ce_loss_dict


def evaluate_deraining_in_training(args, model, writer, epoch, ds):

    eval_dict = {'mse': 0.}
    if args.save_validate_image_results:
        sample_freq = max(int(len(ds) // 20), 1)
        save_dir = os.path.join(args.output_dir, 'validate_results_visualization', 'epoch' + str(epoch), 'deraining')
        os.makedirs(save_dir, exist_ok=True)
        
    for idx in range(len(ds)):
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

    if args.distributed and dist.get_rank() == 0:
        str_out = ''
        for key in eval_dict.keys():
            str_out += key + ': ' + '{:.4f}'.format(eval_dict[key]) + ' '
        print('deraining evaluating: ' + str_out)
        writer.add_scalar('Deraining-MSE', eval_dict['mse'], global_step=epoch, walltime=None)


if __name__ == '__main__':
    args = get_args()
    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    root_output_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)

    # loading vqvae
    vqvae = get_vq_model().eval()
    vqvae.to(args.device)
    
    validate_cpkt_list = {}
    if os.path.isdir(args.ckpt):
        cpkt_list = os.listdir(args.ckpt)
        for item in cpkt_list:
            if item.endswith('.pth'):
                index = int(item.split('.')[0].split('-')[-1])
                validate_cpkt_list[index] = os.path.join(args.ckpt, item)
    else:
        validate_cpkt_list[0] = args.ckpt

    validate_cpkt_list = dict(sorted(validate_cpkt_list.items(), key=lambda item: item[0]))

    for index in validate_cpkt_list.keys():
        # loading mae model
        args.ckpt = validate_cpkt_list[index]
        args.output_dir = os.path.join(root_output_dir, str(index))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        model = prepare_model(args.ckpt, arch=args.model)
        _ = model.to(args.device)

        eval_ce_loss_dict = evaluate(args.yaml_path, model, vqvae)
        # print(index, eval_ce_loss_dict)
        for task in eval_ce_loss_dict.keys():
            writer.add_scalar('Val CE-loss for ' + task, eval_ce_loss_dict[task], global_step=index, walltime=None)
