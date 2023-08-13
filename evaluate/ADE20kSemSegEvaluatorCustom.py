import glob
import json
import os
import argparse
import logging
import sys
sys.path.insert(0, "./")
import numpy as np
import torch
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.evaluation import SemSegEvaluator
import torchvision.transforms as transforms
try:
    np.int
except:
    np.int = np.int32
    np.float = np.float32


def get_args_parser():
    parser = argparse.ArgumentParser('ADE20k semantic segmentation', add_help=False)
    parser.add_argument('--pred_dir', type=str, help='dir to ckpt', required=True)
    parser.add_argument('--dist_type', type=str, help='color type',
                        default='abs', choices=['abs', 'square', 'mean'])
    parser.add_argument('--suffix', type=str, help='model epochs',
                        default="default")
    return parser.parse_args()

def define_colors_per_location_mean_sep():
    num_locations = 150
    num_sep_per_channel = int(num_locations ** (1 / 3)) + 1  # 19
    separation_per_channel = 256 // num_sep_per_channel

    color_list = []
    for location in range(num_locations):
        num_seq_r = location // num_sep_per_channel ** 2
        num_seq_g = (location % num_sep_per_channel ** 2) // num_sep_per_channel
        num_seq_b = location % num_sep_per_channel
        assert (num_seq_r <= num_sep_per_channel) and (num_seq_g <= num_sep_per_channel) \
               and (num_seq_b <= num_sep_per_channel)

        R = 255 - num_seq_r * separation_per_channel
        G = 255 - num_seq_g * separation_per_channel
        B = 255 - num_seq_b * separation_per_channel
        assert (R < 256) and (G < 256) and (B < 256)
        assert (R >= 0) and (G >= 0) and (B >= 0)
        assert (R, G, B) not in color_list

        color_list.append((R, G, B))
        # print(location, (num_seq_r, num_seq_g, num_seq_b), (R, G, B))

    return color_list

class ADE20k_SemSegEvaluatorCustom(SemSegEvaluator):
    def __init__(
        self,
        input_size,
        dataset_name,
        gt_path,
        distributed=True,
        output_dir=None,
        palette=None,
        pred_dir=None,
        dist_type=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
        """
        self._logger = logging.getLogger(dataset_name)
        color_to_idx = {}
        for cls_idx, color in enumerate(palette):
            color = tuple(color)
            # in ade20k, foreground index starts from 1
            color_to_idx[color] = cls_idx + 1
        self.color_to_idx = color_to_idx
        self.palette = torch.tensor(palette, dtype=torch.float, device="cuda")  # (num_cls, 3)
        self.pred_dir = pred_dir
        self.dist_type = dist_type
        self._ignore_label = 255
        self._num_classes = 150
        self.gt_path = gt_path
        self.gt_transforms = transforms.Compose([
            transforms.Resize((input_size // 2, input_size // 2), interpolation=3),
        ])
        self._contiguous_id_to_dataset_id = None
        self._distributed = False
        self._output_dir = None
        self._compute_boundary_iou = True
        self._class_names = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ade20k_class_names.json'), "rb"))


    def process(self, inputs):

        for input in tqdm.tqdm(inputs):
            # output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)  # chw --> hw
            # output = input['pred'].argmax(dim=0).to(self._cpu_device)
            pred = self.post_process_segm_output(input['pred'])
            gt_filename = os.path.join(self.gt_path, input['filename'].replace('jpg', 'png'))
            # gt = np.array(Image.open(gt_filename), dtype=np.int)
            gt = np.array(self.gt_transforms(Image.open(gt_filename)), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            b_gt = self._mask_to_boundary(gt.astype(np.uint8))
            b_pred = self._mask_to_boundary(pred.astype(np.uint8))

            self._b_conf_matrix += np.bincount(
                (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["filename"]))

    def post_process_segm_output(self, segm):
        """
        Post-processing to turn output segm image to class index map

        Args:
            segm: (H, W, 3)

        Returns:
            class_map: (H, W)
        """
        segm = torch.from_numpy(segm).float().to(self.palette.device)  # (h, w, 3)
        # pred = torch.einsum("hwc, kc -> hwk", segm, self.palette)  # (h, w, num_cls)
        h, w, k = segm.shape[0], segm.shape[1], self.palette.shape[0]
        if self.dist_type == 'abs':
            dist = torch.abs(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3))  # (h, w, k)
        elif self.dist_type == 'square':
            dist = torch.pow(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3), 2)  # (h, w, k)
        elif self.dist_type == 'mean':
            dist_abs = torch.abs(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3))  # (h, w, k)
            dist_square = torch.pow(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3), 2)  # (h, w, k)
            dist = (dist_abs + dist_square) / 2.
        else:
            raise NotImplementedError
        dist = torch.sum(dist, dim=-1)
        pred = dist.argmin(dim=-1).cpu()  # (h, w)
        pred = np.array(pred, dtype=np.int)

        return pred


if __name__ == '__main__':
    dataset_name = 'ade20k_sem_seg_val'

    PALETTE = define_colors_per_location_mean_sep()

    evaluator = ADE20k_SemSegEvaluatorCustom(
        input_size=224,
        dataset_name=dataset_name,
        distributed=False,
        gt_path='/mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/Painter/Painter/datasets/ADEChallengeData2016/annotations_detectron2/validation',
        palette=PALETTE,
        dist_type='abs'    
    )

    inputs = []
    root = '/mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/code_for_github/test/validate_results_visualization/epoch0/semantic_segmentation_grid'
    for file_anme in os.listdir(root):
        if "original_" not in file_anme:
            # print(type(file_anme))
            # print(type(file_anme.split('.')[0].split('_')[1]))
            pred = np.array(Image.open(os.path.join(root, file_anme)))[- 112:, - 112:, :]
            Image.fromarray(pred).save('pred.png')
            inputs.append({"pred": pred, 'filename': file_anme})
    # for file_name in prediction_list:
    #     # keys in input: "file_name", keys in output: "sem_seg"
    #     input_dict = {"file_name": file_name}
    #     output_dict = {"sem_seg": file_name}
    #     inputs.append(input_dict)
    #     outputs.append(output_dict)

    evaluator.reset()
    evaluator.process(inputs)
    results = evaluator.evaluate()
    # print(results)

    copy_paste_results = {}
    for key in ['mIoU', 'fwIoU', 'mACC', 'pACC']:
        copy_paste_results[key] = results['sem_seg'][key]
    print(copy_paste_results)

    # result_file = os.path.join(output_folder, "results.txt")
    # print("writing to {}".format(result_file))
    # with open(result_file, 'w') as f:
    #     print(results, file=f)
    #     print(copy_paste_results, file=f)
