#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the local Ministral checkpoint}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-${PROJECT_DIR}/data/sft/prepared_official_seed42}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/sft/prepared_official_seed42_ministral_reasoning}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/r2ai_magic/checkpoints/official_sorrybench_sft_seed42_4gpu_20260819/ministral3-14b-reasoning}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -m r2ai_magic.scripts.prepare_ministral_reasoning_sft \
  --source-dir "${SOURCE_DATA_DIR}" \
  --output-dir "${DATA_DIR}" \
  --system-prompt-file "${MODEL_PATH}/SYSTEM_PROMPT.txt"

export PROJECT_DIR MODEL_PATH DATA_DIR OUTPUT_DIR
export MODEL_LOADER=multimodal_causal_lm
export CHAT_TEMPLATE_MODE=tokenizer
export CHAT_TEMPLATE_KWARGS_JSON='{}'
export NPROC_PER_NODE=4
export CUTOFF_LEN=32768
export EPOCHS=3
export MICRO_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=2
export LEARNING_RATE=1e-5
export WEIGHT_DECAY=0.0
export WARMUP_RATIO=0.1
export SAVE_STEPS=200
export EVAL_STEPS=800
export LOG_STEPS=10
export NUM_WORKERS=2
export SEED=42
export SAVE_TOTAL_LIMIT=1

exec bash "${PROJECT_DIR}/r2ai_magic/scripts/run_magic_generic_causal_sft.sh"
