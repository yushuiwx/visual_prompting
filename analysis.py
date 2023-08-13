import torch
import os

t1 = torch.load('/mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/code_for_github/visual_prompting_8k/tokenizers/8k_xh_ckpt.pth')
t2 = torch.load('/mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/jax_to_torch/torch_cpkt/59136918.pth')
t3 = torch.load('/mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/code_for_github/test/checkpoint-60.pth')

print(t1.keys())
print("=" * 100)
print(t2['model'].keys())
print("=" * 100)
print(t3['model'].keys())
