out_dir=$1
cpkt_path=$2
python -m evaluate.evaluate_deraining \
--model mae_vit_large_patch16 \
--output_dir ${out_dir}/deraining \
--ckpt ${cpkt_path} \
--data_path /mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/Painter/Painter/datasets >> ${out_dir}/deraining.log
