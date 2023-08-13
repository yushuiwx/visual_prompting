
output_dir=/mnt1/msranlpintern/amlt_exp/wuxun/results/test_results/colorization
mkdir -p $output_dir
python -m evaluate.calculate_ce_loss_in_local \
--model mae_vit_large_patch16 \
--output_dir $output_dir \
--yaml_path /mnt1/msranlpintern/amlt_exp/wuxun/code/finetune_in_downstream_tasks/visual_prompting_8k/yamls/eval_colorization.yaml \
--ckpt /mnt1/msranlpintern/amlt_exp/wuxun/results/finetune_in_downstream_tasks/colorization/59136808_lr_1e-5 \