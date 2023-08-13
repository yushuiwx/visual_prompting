out_dir=$1
cpkt_path=$2
python full_validate.py \
--model mae_vit_large_patch16 --input_size 224 --batch_size 64 \
--mask_ratio 0.75 \
--save_ckpt_freq 50 --output_dir ${out_dir} \
--pin_mem --resume ${cpkt_path} \
--data_path /mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/datatsets/CVF >> ${out_dir}/CVF.log


# python full_validate.py \
# --model mae_vit_large_patch16 --input_size 224 --batch_size 64 \
# --mask_ratio 0.5 \
# --save_ckpt_freq 50 --output_dir ${out_dir} \
# --pin_mem --resume ${cpkt_path} \
# --pic_name 'ratio_0.5' \
# --data_path /mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/datatsets/CVF >> ${out_dir}/CVF_0.25.log


# python full_validate.py \
# --model mae_vit_large_patch16 --input_size 224 --batch_size 64 \
# --mask_ratio 0.75 \
# --save_ckpt_freq 50 --output_dir ${out_dir} \
# --pin_mem --resume ${cpkt_path} \
# --pic_name 'ratio_0.75' \
# --data_path /mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/datatsets/CVF >> ${out_dir}/CVF_0.25.log