cd evaluate && python evaluate_segmentation.py \
    --model mae_vit_large_patch16 \
    --base_dir /scratch/wuxun/yutong/datatsets/Datasets_VAT/VOC2012 \
    --output_dir ../results/segmentation \
    --ckpt ../checkpoints/checkpoint-1000.pth \
    --split 0 \
    --dataset_type pascal