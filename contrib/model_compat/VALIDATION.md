# Validation — 2026-10-29

34 distinct CPU unit tests passed:

- 12 FSDP helper tests: assistant loss masks, model-family selection, checkpoint
  metadata, independent Ministral LM heads and distributed evaluation sampling.
- 18 original reasoning/contract tests: target conversion, visible-answer parsing,
  guard parsing, reward contracts and weight-name mapping.
- 4 DeepSeek merged-config tests: validated schema removal, original-config backup,
  unchanged tensor payloads and rejection of incompatible/preexisting state.

The 30 original tests ran on the CPU development machine with
`CUDA_VISIBLE_DEVICES=''`, Python 3.12, torch 2.11.0+cu130, transformers 5.5.3 and
safetensors 0.8.0. The four new DeepSeek config tests ran locally with Python 3.14.
The 18 standard-library tests also passed locally. The CPU test runtime is not a
substitute for the documented historical GPU/backend environment.

To run all tests from this contribution directory with the test dependencies:

```bash
python -m unittest \
  r2ai_magic.tests.test_fsdp_sft \
  r2ai_magic.tests.test_prepare_ministral_reasoning_sft \
  r2ai_magic.tests.test_magic_rl_contracts \
  deepseek_v4.test_normalize_merged_config -v
```

Also verified: syntax of all 30 Python files, `bash -n` for all three shell
launchers, nine standalone CLI help entry points, source-file hashes, staged-diff
whitespace and absence of machine-specific paths/credential patterns in the package.

No GPU training, real checkpoint loading, multi-host rollout, backend-patch
application or numerical LoRA synchronization was rerun. Historical runtime
reference files require review against the recipient's runtime before use.
