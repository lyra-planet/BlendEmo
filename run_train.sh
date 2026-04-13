#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
BLEMORE_ROOT="$(cd "$THIS_DIR/.." && pwd)"
TRAIN="$THIS_DIR/src/train.py"

DATA_DIR="${DATA_DIR:-$BLEMORE_ROOT/ble-datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-$THIS_DIR/outputs_cv}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "$OUTPUT_DIR"

python "$TRAIN" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --hidden_dim 192 \
  --dropout 0.35 \
  --lmf_rank 10 \
  --prompt_layers 1 \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr 3e-4 \
  --weight_decay 6e-3 \
  --num_epochs 40 \
  --warmup_epochs 3 \
  --patience 12 \
  --stats_max_frames_per_file 80 \
  --mixup_alpha 0.2 \
  --set_loss_weight 0.45 \
  --ratio_loss_weight 0.25 \
  --soft_loss_weight 0.9 \
  --mix_loss_weight 0.4 \
  --salience_loss_weight 0.35 \
  --rdrop_weight 0.8 \
  --label_smooth_eps 0.05 \
  --focal_gamma 2.0 \
  --sp_loss_weight 0 \
  --mi_loss_weight 0 \
  --graph_sparse_weight 0 \
  --router_balance_weight 0 \
  --audio_gate_reg_weight 0 \
  2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Done. Results: $OUTPUT_DIR/results.csv"
