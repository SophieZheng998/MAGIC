# DeepSeek-V4 model and runtime adaptations

These files come from our DeepSeek-V4-Flash MAGIC experiments. They are supplied
for integration review; they do not automatically change MAGIC's main trainer.

## Megatron Bridge SFT

- `../r2ai_magic/scripts/prepare_deepseek_v4_megatron_sft.py`: convert prepared
  MAGIC SFT records to the Bridge conversation format.
- `../r2ai_magic/scripts/train_deepseek_v4_megatron_sft.py`: full-SFT recipe entry,
  with text-only recipe imports and caller-supplied model/data/checkpoint paths.
- `train_deepseek_v4_magic_lora_sft.py`: the original Bridge LoRA recipe entry.
- `train_deepseek_v4_magic_lora_sft_single8.py`: disables virtual/interleaved
  pipeline layout for PP=1, enables whole-layer activation recomputation and
  separates native adapter checkpoint saving from HF export. PP=1 must be
  selected explicitly; the inherited argument default is PP=2.
- `../r2ai_magic/scripts/audit_deepseek_v4_sft.py`: read-only checkpoint metadata
  inspection for quantized expert weights and training-memory estimates.

The LoRA entry targets QKV/projection and both MLP projections, with shared expert
adapters and normalized MoE LoRA. It saves native distributed adapter checkpoints;
those files are not directly interchangeable with HF PEFT exports. An example
configuration inspection (requires the compatible Bridge environment) is:

```bash
python deepseek_v4/train_deepseek_v4_magic_lora_sft_single8.py \
  --hf-path /models/DeepSeek-V4-Flash \
  --dataset-root /data/magic-bridge \
  --pretrained-checkpoint /models/deepseek-megatron-base \
  --save /runs/deepseek-adapters \
  --pipeline-parallel-size 1 --expert-parallel-size 8 --dry-run
```

## Dense merged-checkpoint configuration

`normalize_merged_config.py` packages our merge-config repair. It removes only
`mlp_layer_types` when the original base lacks that training-only field and the
expected three-hash-layer layout matches. It saves the original configuration,
refuses to overwrite an existing backup and leaves all weight files unchanged.
Run it only for a matching dense export:

```bash
python deepseek_v4/normalize_merged_config.py \
  --model-dir /models/deepseek-dense-merged \
  --base-model /models/DeepSeek-V4-Flash
```

This metadata repair alone does not verify numerical merge correctness or inference.

## Two historical rollout approaches

`magic_grpo_native_entry_v15.py` is the earlier native Megatron rollout entry.
It adapts the tokenizer/response interface, isolates control collectives in a
Gloo group and skips base hyper-connection imports for adapter-only PEFT loads.
It is retained as a historical implementation, not the later dense-sync launcher.

`runtime_reference/swift/megatron/trainers/rollout_mixin.py` and
`runtime_reference/vllm/model_executor/models/deepseek_v4.py` preserve the later
September 12 runtime sources. The former contains LoRA merge/export/unmerge and
weight-transfer handling; the latter contains the corresponding DeepSeek model
weight-loader implementation. They include upstream code as well as local changes;
retain the upstream license notices when extracting changes.

The original released expert tensors use compressed storage, whereas merged
training tensors can be BF16. Do not assume dense BF16 updates can be loaded into
a runtime initialized with the compressed expert representation. The later
experiment used a dense merged runtime for full weight synchronization.

These snapshots are review material, not automatically installed patches. Their
presence does not establish successful training at every topology or context
length. No GPU job was launched for this contribution; integration still needs
model loading, rollout, nonzero-gradient, save/reload and synchronization checks.
