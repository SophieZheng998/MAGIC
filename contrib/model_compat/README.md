# Qwen, Ministral and DeepSeek compatibility

This contribution collects our original model adapters from the MAGIC experiment
checkout and saved runtime sources. It is separate from `magic_model_support/`
and does not replace the repository's existing trainer or dependency pins.
Maintainers can review and integrate the components independently.

| Model | Included work |
| --- | --- |
| Qwen3.6-27B | Language-only loading/export, literal ChatML loss masking, text/service checkpoint views, weight-prefix and FlashAttention position handling, VERL rendering and response-logit patches. |
| Ministral-3-14B-Reasoning | Native multimodal loader with independent LM head, `[THINK]` target conversion, text-weight synchronization/export, RoPE/YaRN patches. |
| DeepSeek-V4-Flash | Megatron Bridge full/LoRA SFT entries, single-stage pipeline adaptation, checkpoint metadata audit, and saved native/dense-rollout runtime sources. See [DeepSeek notes](deepseek_v4/README.md). |

## Use and review

Run the contribution's modules from this directory so `r2ai_magic` is importable:

```bash
cd contrib/model_compat
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

For Qwen SFT, supply prepared `magic_sft_train.json` and
`magic_sft_validation.json` arrays with `instruction`, `input`, `system`, `output`:

```bash
MODEL_PATH=/models/Qwen3.6-27B DATA_DIR=/data/magic-sft OUTPUT_DIR=/runs/qwen-sft \
bash r2ai_magic/scripts/run_qwen36_attacker_sft.sh
```

For Ministral, the launcher converts canonical `<think>...</think><answer>...</answer>`
targets using the checkpoint's `SYSTEM_PROMPT.txt` before training:

```bash
MODEL_PATH=/models/Ministral-3-14B-Reasoning-2512 \
SOURCE_DATA_DIR=/data/magic-sft DATA_DIR=/data/ministral-sft OUTPUT_DIR=/runs/ministral-sft \
bash r2ai_magic/scripts/run_ministral_reasoning_sft_4gpu.sh
```

The historical launcher hyperparameters are retained. The generic launcher
rejects large MoE models; use the separate DeepSeek backend for DeepSeek-V4.

`r2ai_magic/rl/` contains the original agent loop, contracts, weight helpers and
backend patches for review. It is not wired into the current main-branch trainer.
The runtime hooks are opt-in through their `MAGIC_*` environment switches; the
Ministral hook is under `runtime/ministral/` and the Qwen hook is under
`r2ai_magic/runtime/vllm-qwen35-sft-adapter/`.

See [dependency notes](DEPENDENCIES.md) before configuring a runtime. The pinned
backend patches do not target arbitrary newer versions. This branch contains no
model weights, experiment data, cluster launchers, credentials or run logs.

`source_manifest.json` records original source hashes and the final contribution
hashes. Packaging changes replace machine-specific defaults with explicit inputs
and document the scope of historical runtime references. CPU validation is
recorded in [VALIDATION.md](VALIDATION.md); GPU training was not rerun for this branch.
