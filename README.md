# Visual-Prompting
Visual-Prompting (MAE+VQVAE reconstruction)

## Datasets Preparation
There are three parts:
* CVF datasets
* ImageNet datasets
* Supervised vision datasets

support for ， validating in segmentation, object detectio, reasoning, colorization and CVF val-set. More downstream tasks validation is coming soon.... 
* VOC2012 (for segmentation and object detection): https://microsoft-my.sharepoint.com/:u:/p/v-wuxun/IQCN2LCN0bahTL5p8fVUJHC2AdK1Igb1UYlBBhVxx8TBD1k?email=ytongbai%40jhu.edu&e=EeIy0a
* InageNet Val Sub-set (for colorization): https://microsoft-my.sharepoint.com/:u:/p/v-wuxun/IQCPdewMPkTkSonImgTTcXf8AZo0drH5JUaOPkZB7BE1F_0?email=ytongbai%40jhu.edu&e=G3diM2
* the datasets should look like:
```
${eval_data_dir}/
    ImageNet/
        ImageNet_subset_val/
    VOC2012/
        VOCdevkit/
```
## Description

[jax_to_torch_for_mae.py](jax_to_torch_for_mae.py): transform jax mae model to torch.

[jax_to_torch_for_vqvae.py](jax_to_torch_for_vqvae.py): transform jax vqvae model to torch.

[torch_vqvae_model.py](torch_vqvae_model.py): torch-version vqvae inference model.

[mae_vqvae_recon_visualize.ipynb](mae_vqvae_recon_visualize.ipynb): torch-version vqvae image reconstruction visualization.

[mae_vqvae_recon_visualize.ipynb](mae_vqvae_recon_visualize.ipynb): torch-version mae+vqvae image reconstruction visualization.

[./mae/](./mae/): the modified folder of the official torch-version mae model.

[./figs/](./figs/): the folder for figures used for reconstruction.

[./torch_ckpts/](./torch_ckpts/): Please put transformed torch-version mae model [0613_ckpt_torch.pth (google drive link)](https://drive.google.com/file/d/18OKC86aKypcChjMSf6eglRyiAv4cxHZN/view?usp=sharing)) and vqvae model [xh_ckpt.pth (google drive link)](https://drive.google.com/file/d/1Z8Rua3E_WVBLIZ8bKI7-9ZQkqTHlBv7O/view?usp=sharing) in this folder. Specifically, torch-version 0613_ckpt_torch.pth and xh_ckpt.pth are transformed from jax-version [ckpt_xh.npy](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/ybai20_jh_edu/EXOe22D0z7VMkR75BwDcXxABXRs8eqPJ-mCDEEauIYxRDg?email=xxh11102019%40outlook.com&e=4%3aqhRDlo&at=9) and [checkpoint.zip](https://livejohnshopkins-my.sharepoint.com:443/:u:/g/personal/ybai20_jh_edu/EZZPDlPP639DsbwvofxXwloBY8YhFttRnJ4waPaUV9tbnA?email=xxh11102019%40outlook.com&e=4%3aoOaIvC&at=9) respectively.


