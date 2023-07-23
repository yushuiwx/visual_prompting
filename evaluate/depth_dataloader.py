"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import json


class DatasetDepth(Dataset):
    def __init__(self, datapath, image_transform, mask_transform, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False):
        self.padding = padding
        self.random = random
        self.use_original_imgsize = use_original_imgsize
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform
        test_json = os.path.join(datapath, 'nyu_depth_v2', 'nyuv2_test_image_depth.json')
        with open(test_json, 'r') as f:
            derain_list = json.load(f)
        self.ds = derain_list
        self.flipped_order = flipped_order
        np.random.seed(5)
        self.datapath = datapath

    def __len__(self):
        return len(self.ds)

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask):
        if self.reverse_support_and_query:
            support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
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

    def __getitem__(self, idx):
        support_idx = np.random.choice(np.arange(0, len(self.ds)-1))
        query_paths, support_paths = self.ds[idx], self.ds[support_idx]
        query_img, query_mask = self.image_transform(self.read_img(query_paths['image_path'])), self.mask_transform(self.read_img(query_paths['target_path']))
        support_img, support_mask = self.image_transform(self.read_img(support_paths['image_path'])), self.mask_transform(self.read_img(support_paths['target_path']))
        grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask)
        batch = {'query_img': query_img, 'query_mask': query_mask, 'support_img': support_img,
                 'support_mask': support_mask, 'grid': grid}

        return batch

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.datapath, img_name))