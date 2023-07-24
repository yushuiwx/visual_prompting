# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os.path, sys
sys.path.append('./')
import json
from typing import Any, Callable, List, Optional, Tuple
import random
import datasets.pair_transforms as pair_transforms
from PIL import Image
import numpy as np
import random
import torch
import torchvision
from torchvision.datasets.vision import VisionDataset, StandardTransform
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms

class PairStandardTransform(StandardTransform):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(transform=transform, target_transform=target_transform)

    def __call__(self, input: Any, target: Any, interpolation1: Any, interpolation2: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input, target = self.transform(input, target, interpolation1, interpolation2)
        return input, target


# simple augmentation
def get_transform_for_datasets(args):
    transform_train1 = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.input_size, scale=(args.min_random_scale, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.RandomHorizontalFlip(),
            pair_transforms.ToTensor()])
    transform_train2 = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.input_size, scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor()])
    transform_train3 = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.input_size, scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor()])
    transform_val = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.input_size, scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor()])

    return PairStandardTransform(transform_train1, None), PairStandardTransform(transform_train2, None), \
                                                        PairStandardTransform(transform_train3, None), PairStandardTransform(transform_val, None)

class PairDataset_train(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        args,
        cfg,
    ) -> None:
        super().__init__(root=None)

        root_path = cfg.datasets.append_supervised.root_path
        json_path = cfg.datasets.append_supervised.json_path

        self.root = root_path
        self.in_context_pairs_number = args.in_context_pairs_number
        self.transforms1, self.transforms2, self.transforms3, self.transform_val = get_transform_for_datasets(args)
        self.final_transfroms = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(args.min_random_scale, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # processing training and validating json files
        json_path_list = []
        for task in json_path.keys():
            for task_data in json_path[task]:
                json_path_list.append(task_data.train_json)
        # loading pairs from json files
        print("Loading from yamls, get", json_path_list, "for training.")
        self.pairs = []
        for idx, json_path in enumerate(json_path_list):
            cur_pairs = json.load(open(os.path.join(root_path, json_path)))
            self.pairs.extend(cur_pairs)

        # merge into same type
        self.pair_type_dict = {}
        for idx, pair in enumerate(self.pairs):
            if "type" in pair:
                if pair["type"] not in self.pair_type_dict:
                    self.pair_type_dict[pair["type"]] = [idx]
                else:
                    self.pair_type_dict[pair["type"]].append(idx)

        # print("=" * 50, action, "=" * 50)
        for t in self.pair_type_dict:
            print(t, len(self.pair_type_dict[t]))

    def _load_image(self, path: str) -> Image.Image:
        while True:
            try:
                img = Image.open(os.path.join(self.root, path))
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(1)
            else:
                break
        # process for nyuv2 depth: scale to 0~255
        if "sync_depth" in path:
            # nyuv2's depth range is 0~10m
            img = np.array(img) / 10000.
            img = img * 255
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return img

    def _combine_images(self, image, image2, padding=None, t=None):
        c, h, w = image.shape
        if t < 0.5:
            dst = torch.cat([image, torch.ones(c, h, padding), image2], dim=2)
        else:
            dst = torch.cat([image, torch.zeros(c, h, padding), image2], dim=2)
        return dst

    def _combine_pairs(self, pairs, pairs2, padding=None, t=None):
        c, h, w = pairs.shape
        if t < 0.5:
            dst = torch.cat([pairs, torch.ones(c, padding, w), pairs2], dim=1)
        else:
            dst = torch.cat([pairs, torch.zeros(c, padding, w), pairs2], dim=1)
        return dst

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pair = self.pairs[index]
        image = self._load_image(pair['image_path'])
        target = self._load_image(pair['target_path'])

        # decide mode for interpolation
        pair_type = pair['type']
        if "depth" in pair_type or "pose" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
        elif "image2" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'nearest'
        elif "2image" in pair_type:
            interpolation1 = 'nearest'
            interpolation2 = 'bicubic'
        else:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
        
        # no aug for instance segmentation
        if "inst" in pair['type'] and self.transforms2 is not None:
            cur_transforms = self.transforms2
        elif "pose" in pair['type'] and self.transforms3 is not None:
            cur_transforms = self.transforms3
        else:
            cur_transforms = self.transforms1
        
        padding = random.randint(0, 8)
        t = random.random()
        image, target = cur_transforms(image, target, interpolation1, interpolation2)
        generated_grids = self._combine_images(image, target, padding, t)
        
        # use_some_pairs:
        pair_type = pair['type']

        # sample the in_context pair belonging to the same type
        in_context_pairs_number = random.randint(2, self.in_context_pairs_number)
        in_context_pairs_indexs = []
        for _ in range(in_context_pairs_number):
            pair2_index = random.choice(self.pair_type_dict[pair_type])
            if pair2_index not in in_context_pairs_indexs:
                in_context_pairs_indexs.append(pair2_index)
                pair2 = self.pairs[pair2_index]
                image2 = self._load_image(pair2['image_path'])
                target2 = self._load_image(pair2['target_path'])
                assert pair2['type'] == pair_type
                image2, target2 = cur_transforms(image2, target2, interpolation1, interpolation2)
                in_context_grid = self._combine_images(image2, target2, padding, t)
                generated_grids = self._combine_pairs(in_context_grid, generated_grids, padding, t)
        # resize to 224
        generated_grids = self.final_transfroms(generated_grids)
        return generated_grids, None # label, to fit ImageFolder

    def __len__(self) -> int:
        return len(self.pairs)


# class PairDataset_validate(VisionDataset):
#     """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

#     It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

#     Args:
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.PILToTensor``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         transforms (callable, optional): A function/transform that takes input sample and its target as entry
#             and returns a transformed version.
#     """

#     def __init__(
#         self,
#         args,
#         cfg,
#     ) -> None:
#         super().__init__(root=None)

        
#         root_path = cfg.datasets.append_supervised.root_path
#         json_path = cfg.datasets.append_supervised.json_path
#         self.padding = 2
#         self.flipped_order = False
#         self.root = root_path
#         self.in_context_pairs_number = args.in_context_pairs_number
#         _, _, _, self.transform_val = get_transform_for_datasets(args)
#         self.final_transfroms = transforms.Compose([
#             transforms.RandomResizedCrop(args.input_size, scale=(0.9999, 1.0), interpolation=3),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         # processing training and validating json files
#         json_path_list = []
#         for task in json_path.keys():
#             for task_data in json_path[task]:
#                 json_path_list.append(task_data.val_json)
#         # loading pairs from json files
#         print("Loading from yamls, get", json_path_list, "for validating.")
#         self.pairs = []
#         for idx, json_path in enumerate(json_path_list):
#             cur_pairs = json.load(open(os.path.join(root_path, json_path)))
#             self.pairs.extend(cur_pairs)

#         # merge into same type
#         self.pair_type_dict = {}
#         for idx, pair in enumerate(self.pairs):
#             if "type" in pair:
#                 if pair["type"] not in self.pair_type_dict:
#                     self.pair_type_dict[pair["type"]] = [idx]
#                 else:
#                     self.pair_type_dict[pair["type"]].append(idx)

#         # print("=" * 50, action, "=" * 50)
#         for t in self.pair_type_dict:
#             print(t, len(self.pair_type_dict[t]))

#     def _load_image(self, path: str) -> Image.Image:
#         while True:
#             try:
#                 img = Image.open(os.path.join(self.root, path))
#             except OSError as e:
#                 print(f"Catched exception: {str(e)}. Re-trying...")
#                 import time
#                 time.sleep(1)
#             else:
#                 break
#         # process for nyuv2 depth: scale to 0~255
#         if "sync_depth" in path:
#             # nyuv2's depth range is 0~10m
#             img = np.array(img) / 10000.
#             img = img * 255
#             img = Image.fromarray(img)
#         img = img.convert("RGB")
#         return img

#     def create_grid_from_images(self, support_img, support_mask, query_img, query_mask):
#         canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
#                              2 * support_img.shape[2] + 2 * self.padding))
#         canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
#         if self.flipped_order:
#             canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
#             canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
#             canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
#         else:
#             canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
#             canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
#             canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

#         return canvas

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         pair = self.pairs[index]
#         image = self._load_image(pair['image_path'])
#         target = self._load_image(pair['target_path'])

#         # decide mode for interpolation
#         pair_type = pair['type']
#         if "depth" in pair_type or "pose" in pair_type:
#             interpolation1 = 'bicubic'
#             interpolation2 = 'bicubic'
#         elif "image2" in pair_type:
#             interpolation1 = 'bicubic'
#             interpolation2 = 'nearest'
#         elif "2image" in pair_type:
#             interpolation1 = 'nearest'
#             interpolation2 = 'bicubic'
#         else:
#             interpolation1 = 'bicubic'
#             interpolation2 = 'bicubic'
        
#         # no aug for instance segmentation
#         image, target = self.transform_val(image, target, interpolation1, interpolation2)
        
#         # use_some_pairs:
#         pair_type = pair['type']

#         # sample the in_context pair belonging to the same type
#         pair2_index = random.choice(self.pair_type_dict[pair_type])
#         pair2 = self.pairs[pair2_index]
#         image2 = self._load_image(pair2['image_path'])
#         target2 = self._load_image(pair2['target_path'])
#         assert pair2['type'] == pair_type
#         image2, target2 = self.transform_val(image2, target2, interpolation1, interpolation2)
#         grid = self.create_grid_from_images(image2, target2, image, target)
#         grid = self.final_transfroms(grid)

#         batch = {'image2': image2, 'target2': target2, 'image': image,
#                  'target': target, 'grid': grid, 'type': pair_type}
#         return batch, None

#     def __len__(self) -> int:
#         return len(self.pairs)



class PairDataset_validate_for_single_task(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        args,
        cfg,
        validate_json_path,
    ) -> None:
        super().__init__(root=None)

        
        root_path = cfg.datasets.append_supervised.root_path
        self.padding = 0
        self.flipped_order = False
        self.root = root_path
        self.in_context_pairs_number = args.in_context_pairs_number
        _, _, _, self.transform_val = get_transform_for_datasets(args)
        self.final_transfroms = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.9999, 1.0), interpolation=3),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.pairs = json.load(open(os.path.join(root_path, validate_json_path)))
        self.type = self.pairs[0]["type"]
        # print("=" * 50, action, "=" * 50)
        print(self.type, len(self.pairs))

    def _load_image(self, path: str) -> Image.Image:
        while True:
            try:
                img = Image.open(os.path.join(self.root, path))
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(1)
            else:
                break
        # process for nyuv2 depth: scale to 0~255
        if "sync_depth" in path:
            # nyuv2's depth range is 0~10m
            img = np.array(img) / 10000.
            img = img * 255
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return img

    def __get_padding__(self):
        return self.padding

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask):
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if self.flipped_order:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pair = self.pairs[index]
        image = self._load_image(pair['image_path'])
        target = self._load_image(pair['target_path'])

        # decide mode for interpolation
        pair_type = pair['type']
        if "depth" in pair_type or "pose" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
        elif "image2" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'nearest'
        elif "2image" in pair_type:
            interpolation1 = 'nearest'
            interpolation2 = 'bicubic'
        else:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
        
        # no aug for instance segmentation
        image, target = self.transform_val(image, target, interpolation1, interpolation2)

        # sample the in_context pair belonging to the same type
        pair2 = random.choice(self.pairs)
        image2 = self._load_image(pair2['image_path'])
        target2 = self._load_image(pair2['target_path'])
        assert pair2['type'] == pair_type
        image2, target2 = self.transform_val(image2, target2, interpolation1, interpolation2)
        grid = self.create_grid_from_images(image2, target2, image, target)
        grid = self.final_transfroms(grid)

        batch = {'image2': image2, 'target2': target2, 'image': image,
                 'target': target, 'grid': grid, 'type': pair_type, 'filename':pair['image_path'].split('/')[-1]}

        return batch, None


    def __len__(self) -> int:
        return len(self.pairs)