cd evaluate && python evaluate_reasoning.py \
--model mae_vit_large_patch16 \
--output_dir /mnt1/msranlpintern/amlt_exp/wuxun/results/analysis/test \
--ckpt /mnt1/msranlpintern/amlt_exp/wuxun/results/finetune/visual_prompting_finetune_17k_lr5e-5/checkpoint-1699.pth \
--dataset_type size_shape  \
--tta_option 1