#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the local model checkpoint}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/r2ai_magic/data/generated/wjb_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/r2ai_magic/checkpoints/qwen36-27b/attacker-sft}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODEL_LOADER="${MODEL_LOADER:-qwen3_5_multimodal_text}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-qwen_chatml_literal}"
CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-}"
if [[ -z "${CHAT_TEMPLATE_KWARGS_JSON}" ]]; then
  CHAT_TEMPLATE_KWARGS_JSON='{}'
fi
EXTRA_ARGS=()

if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max-train-steps "$MAX_TRAIN_STEPS")
fi
if [[ "${CPU_OFFLOAD:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--cpu-offload)
fi
if [[ "${SKIP_FINAL_SAVE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-final-save)
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m r2ai_magic.scripts.validate_sft_data \
  --model-path "$MODEL_PATH" \
  --train-file "$DATA_DIR/magic_sft_train.json" \
  --validation-file "$DATA_DIR/magic_sft_validation.json" \
  --cutoff-len "${CUTOFF_LEN:-8192}" \
  --model-loader "$MODEL_LOADER" \
  --chat-template-mode "$CHAT_TEMPLATE_MODE" \
  --chat-template-kwargs-json "$CHAT_TEMPLATE_KWARGS_JSON" \
  --max-unsharded-weight-gib "${MAX_UNSHARDED_WEIGHT_GIB:-100}"

torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
  -m r2ai_magic.fsdp_sft \
  --model-path "$MODEL_PATH" \
  --train-file "$DATA_DIR/magic_sft_train.json" \
  --validation-file "$DATA_DIR/magic_sft_validation.json" \
  --output-dir "$OUTPUT_DIR" \
  --cutoff-len "${CUTOFF_LEN:-8192}" \
  --epochs "${EPOCHS:-3}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --warmup-ratio "${WARMUP_RATIO:-0.1}" \
  --save-steps "${SAVE_STEPS:-1000}" \
  --eval-steps "${EVAL_STEPS:-1000}" \
  --log-steps "${LOG_STEPS:-5}" \
  --seed "${SEED:-42}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --model-loader "$MODEL_LOADER" \
  --chat-template-mode "$CHAT_TEMPLATE_MODE" \
  --chat-template-kwargs-json "$CHAT_TEMPLATE_KWARGS_JSON" \
  --max-unsharded-weight-gib "${MAX_UNSHARDED_WEIGHT_GIB:-100}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-1}" \
  "${EXTRA_ARGS[@]}"
