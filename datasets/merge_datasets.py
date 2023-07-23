from torchvision import datasets
from typing import Optional, Callable, Tuple, Any
import torch
import random
import numpy as np

class MergedImageFolder(torch.utils.data.Dataset):
    def __init__(self, datasets_list, avg = True):
        # dataset1 is smaller than dataset2 / dataset1:0.75
        self.datasets_list = datasets_list
        self.sample_number_list = [item.__len__() for item in datasets_list]
        if avg:
            self.weights_list = [1.0 / len(datasets_list) for _ in datasets_list]
        else:
            self.sum_weights = sum(self.sample_number_list)
            self.weights_list = [item / self.sum_weights for item in self.sample_number_list]
        print("merge weights list:", self.weights_list)
        
    def __len__(self):
        return min(self.sample_number_list)

    
    def __getitem__(self, index: int):
        selected_dataset = random.choices(self.datasets_list, weights=self.weights_list, k=1)[0]
        index = random.randint(0, len(selected_dataset) - 1)
        return selected_dataset[index][0]