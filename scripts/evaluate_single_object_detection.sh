cd evaluate && python evaluate_segmentation.py \
    --task detection \
    --model mae_vit_large_patch16 \
    --base_dir /scratch/wuxun/yutong/datatsets/Datasets_VAT/VOC2012 \
    --output_dir ../results/single_object_detection \
    --ckpt ../checkpoints/checkpoint-1000.pth \
    --dataset_type pascal_det