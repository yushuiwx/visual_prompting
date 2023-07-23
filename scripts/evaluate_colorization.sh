python -m evaluate.evaluate_colorization \
--model mae_vit_large_patch16 \
--output_dir ./results/colorization \
--ckpt ./checkpoints/checkpoint-1000.pth \
--data_path /mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/datatsets/ImageNet >> ./results/colorization.log