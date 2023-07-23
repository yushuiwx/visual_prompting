# Visual-Prompting
Visual-Prompting (MAE+VQVAE reconstruction)

## Datasets Preparation
There are three parts:
* CVF datasets
```
${CVF}/
	train/
	val/
```
* ImageNet datasets
```
${ImageNet}/
	train/
	val/
```
* Supervised vision datasets
```
$Painter_ROOT/datasets/
    nyu_depth_v2/
        sync/
        official_splits/
        nyu_depth_v2_labeled.mat
        datasets/nyu_depth_v2/
        nyuv2_sync_image_depth.json  # generated
        nyuv2_test_image_depth.json  # generated
    ade20k/
        images/
        annotations/
        annotations_detectron2/  # generated
        annotations_with_color/  # generated
        ade20k_training_image_semantic.json  # generated
        ade20k_validation_image_semantic.json  # generated
    ADEChallengeData2016/  # sim-link to $Painter_ROOT/datasets/ade20k
    coco/
        train2017/
        val2017/
        annotations/
            instances_train2017.json
            instances_val2017.json
            person_keypoints_val2017.json
            panoptic_train2017.json
            panoptic_val2017.json
            panoptic_train2017/
            panoptic_val2017/
        panoptic_semseg_val2017/  # generated
        panoptic_val2017/  # sim-link to $Painter_ROOT/datasets/coco/annotations/panoptic_val2017
        pano_sem_seg/  # generated
            panoptic_segm_train2017_with_color
            panoptic_segm_val2017_with_color
            coco_train2017_image_panoptic_sem_seg.json
            coco_val2017_image_panoptic_sem_seg.json
        pano_ca_inst/  # generated
            train_aug0/
            train_aug1/
            ...
            train_aug29/
            train_org/
            train_flip/
            val_org/
            coco_train_image_panoptic_inst.json
            coco_val_image_panoptic_inst.json
    coco_pose/
        person_detection_results/
            COCO_val2017_detections_AP_H_56_person.json
        data_pair/  # generated
            train_256x192_aug0/
            train_256x192_aug1/
            ...
            train_256x192_aug19/
            val_256x192/
            test_256x192/
            test_256x192_flip/
        coco_pose_256x192_train.json  # generated
        coco_pose_256x192_val.json  # generated
    derain/
        train/
            input/
            target/
        test/
            Rain100H/
            Rain100L/
            Test100/
            Test1200/
            Test2800/
        derain_train.json
        derain_test_rain100h.json
    denoise/
        SIDD_Medium_Srgb/
        train/
        val/
        denoise_ssid_train.json  # generated
        denoise_ssid_val.json  # generated
    light_enhance/
        our485/
            low/
            high/
        eval15/
            low/
            high/
        enhance_lol_train.json  # generated
        enhance_lol_val.json  # generated
```
please motified yamls/base.yaml to add or remove datasets:
```yaml
datasets:
  cvf: 
    image_path: '/mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/datatsets/CVF_debug'
  imageNet:
    image_path: '/mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/datatsets/ImageNet'
  append_supervised:
    root_path: '/mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/Painter/Painter/datasets'
    json_path: 
      deraining:
        - datasets: 'MRNet'
          train_json: derain/derain_train.json
          val_json: derain/derain_test_rain100h.json
      
      colorization:
        - datasets: 'ImageNet'
          train_json: colorization/colorization_ImageNet_train.json
          val_json: colorization/colorization_ImageNet_val.json

      light_enhance:
        - datasets: 'LOL'
          train_json: light_enhance/enhance_lol_train.json
          val_json: light_enhance/enhance_lol_val.json

      depth_estimation:
        - datasets: 'nyu_depth_v2'
          train_json: nyu_depth_v2/nyuv2_sync_image_depth.json
          val_json: nyu_depth_v2/nyuv2_test_image_depth.json
```

## Train & Validate
```
bash scripts/train.sh
bash scripts/evaluate_*.sh
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

