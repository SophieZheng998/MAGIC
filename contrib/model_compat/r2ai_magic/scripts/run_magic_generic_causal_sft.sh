#!/usr/bin/env bash
set -euo pipefail

# Generic text-only path for DeepSeek-Distill, Mistral/Ministral, and other
# AutoModelForCausalLM checkpoints. Multimodal and large expert-parallel models
# intentionally require a separate audited loader.
: "${MODEL_PATH:?Set MODEL_PATH to a local text-only Hugging Face checkpoint.}"
: "${DATA_DIR:?Set DATA_DIR to the generated MAGIC SFT directory.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints.}"

export MODEL_LOADER="${MODEL_LOADER:-generic_causal_lm}"
export CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-tokenizer}"
export CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-}"
if [[ -z "${CHAT_TEMPLATE_KWARGS_JSON}" ]]; then
  export CHAT_TEMPLATE_KWARGS_JSON='{}'
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$PROJECT_DIR/r2ai_magic/scripts/run_qwen36_attacker_sft.sh"
