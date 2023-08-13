#!/bin/bash
sudo apt-get update
sudo apt-get install netcat

# Function to check if a port is available
function is_port_available() {
  local port=$1
  if ! nc -z 127.0.0.1 "$port"; then
    return 0
  else
    return 1
  fi
}

task=$1
cpkt=$2
base_port=29101
max_port=29999
outdir="/mnt1/msranlpintern/amlt_exp/wuxun/results/finetune_in_downstream_tasks/${task}/${cpkt}"
mkdir -p "$outdir"

available_port=""
for ((port = base_port; port <= max_port; port++)); do
  if is_port_available "$port"; then
    echo "Port $port is available."
    available_port="$port"
    break
  else
    echo "Port $port is in use."
  fi
done

if [ -z "$available_port" ]; then
  echo "No available ports in the range $base_port-$max_port."
else
  echo "Launching experiment with port $available_port..."
  python -m torch.distributed.launch --nproc_per_node=8 --master_port="$available_port" main_pretrain.py \
    --model mae_vit_large_patch16 --input_size 224 --batch_size 64 \
    --mask_ratio 0.75 \
    --warmup_epochs 15 \
    --epochs 1700 --blr 1e-4 \
    --save_ckpt_freq 20 \
    --output_dir "$outdir" \
    --start_epoch 0 \
    --in_context_pairs_number 4 \
    --yaml_path "/mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/code_for_github/visual_prompting_8k/yamls/finetune_in_${task}.yaml" \
    --eval_data_dir /mnt1/msranlpintern/wuxun/SemDeDup/cili/Datasets_VAT \
    --resume "/mnt1/msranlpintern/wuxun/SemDeDup/cili/scratch/wuxun/yutong/jax_to_torch/torch_cpkt/${cpkt}.pth" \
    --start_epoch 0 \
    --save_validate_image_results \
    --val_freq 20 >> "$outdir/train.log"
fi
