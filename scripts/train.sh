outdir=/mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/code_for_github/test
mkdir -p $outdir
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py \
--model mae_vit_large_patch16 --input_size 224 --batch_size 64 \
--mask_ratio 0.75 \
--warmup_epochs 15 \
--epochs 1700 --blr 1e-4 \
--save_ckpt_freq 20 \
--output_dir ${outdir} \
--start_epoch 0 \
--in_context_pairs_number 6 \
--yaml_path /mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/code_for_github/visual_prompting/yamls/base.yaml \
--eval_data_dir /mnt1/msranlpintern/wuxun/SemDeDup/cili/Datasets_VAT \
--save_validate_image_results \
--val_freq 1 >> $outdir/train.log
