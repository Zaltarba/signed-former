#!/bin/bash
set -e

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
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 3 \
  --d_ff 512 \
  --dropout 0.1 \
  --embed timeF \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --patience 3 \
  --keep_ratio 0.03 \
  --time_budget 300 \
  --patch_len 12 \
  --stride 6 \
  --n_stacks 1 \
  --attention_window 10 \
  --use_norm 1 \
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
