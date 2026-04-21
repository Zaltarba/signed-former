#!/bin/bash
set -e

source "$(dirname "$0")/.venv/bin/activate" 2>/dev/null || true

COMMIT=$(git rev-parse --short HEAD)
DESCRIPTION=$(git log -1 --pretty=%s)

if python run.py \
  --is_training 1 \
  --model_id PEMS08_custom \
  --model CustomModel \
  --data PEMS \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --features M \
  --seq_len 192 \
  --label_len 0 \
  --pred_len 12 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --d_model 128 \
  --n_heads 8 \
  --e_layers 2 \
  --d_ff 32 \
  --dropout 0.05 \
  --embed timeF \
  --train_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --patience 3 \
  --keep_ratio 0.2 \
  --time_budget 600 \
  --patch_len 32 \
  --stride 16 \
  --n_stacks 1 \
  --attention_window 10 \
  --use_norm 1 \
  --lradj fixed \
  --itr 1 \
  --des "autorun"; then
  echo "Experiment completed successfully."
else
  GPU_MEM="0.0"
  WRITE_HEADER=false
  [ ! -f results.tsv ] && WRITE_HEADER=true
  if [ "$WRITE_HEADER" = true ]; then
    echo -e "commit_hash\tMSE\tgpu_mem_gb\tstatus\tdescription" >> results.tsv
  fi
  echo -e "${COMMIT}\tN/A\t${GPU_MEM}\tcrash\t${DESCRIPTION}" >> results.tsv
  echo "Experiment crashed. Logged to results.tsv."
  exit 1
fi
