# Dependency notes

Use separate environments for the Qwen/Ministral VERL path and the DeepSeek
Megatron paths. The repository root `requirements.txt` is unchanged.

## Qwen / Ministral

The original environment recorded Python 3.12, CUDA 12.9, PyTorch 2.10.0+cu129,
vLLM 0.18.0, Transformers 5.5.3, FlashAttention 2.8.3, Ray 2.54.1 and VERL commit
`5ba5f2f51281522150a6c1c87e3f841d1d220042` (`0.10.0.dev0`). The version inventory
is in `requirements-qwen-ministral.txt`.

That inventory is **not a resolver-clean installation recipe**: vLLM 0.18.0
metadata declares `transformers<5`, while these model adapters use Transformers 5
APIs. Provision an isolated compatible CUDA environment and review/apply the
version-specific patches in `r2ai_magic/rl/patches/`. Do not install the old vendored
`src/verl` on top of this runtime. Backend patch application and real-model GPU
smoke tests are integration steps for the receiving project.

The standard-library tests can run without these training packages. The FSDP
unit tests additionally require torch, transformers and safetensors.

## DeepSeek

The original Megatron Bridge SFT experiment recorded Bridge
`c9ab9cc124a055392927eee3b2220bc0049caa18` and Megatron-LM
`ed6b1f65502aec7f2fe27e14a1245c29e435c2a6`, with Transformers 5.8.1,
NVIDIA ModelOpt 0.44.0rc5 and fast-hadamard-transform 1.1.0. These are source-era
references, not a claim that an unmodified installation runs every later LoRA entry.

The later native/full-sync experiments used a separate patched MS-Swift/MCore
Bridge runtime and a vLLM runtime. The saved files in `deepseek_v4/runtime_reference/`
are the September 12 source snapshots, **not** a complete distributable environment.
Their exact original package commits are not recorded in the local snapshot, so
this contribution does not invent a universal DeepSeek requirements lock or copy
those files automatically into site-packages. Review the modified model loader and
weight-transfer code against the target runtime before integration.
