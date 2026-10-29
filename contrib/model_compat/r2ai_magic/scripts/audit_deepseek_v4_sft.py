#!/usr/bin/env python3
"""Read-only DeepSeek-V4 checkpoint audit for full-parameter SFT.

This utility intentionally does not import Transformers or load model tensors.  It
can therefore distinguish checkpoint-format/resource blockers from Python package
errors before reserving GPUs.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import struct
from collections import Counter
from pathlib import Path


DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
}


def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_size))


def inspect_checkpoint(
    model_path: Path, gpus: int, per_gpu_memory_gb: float
) -> dict:
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())

    dtype_numel: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    tensor_count = 0
    fp4_packed_weight_numel = 0
    scale_numel = 0

    for shard in sorted(glob.glob(str(model_path / "model-*.safetensors"))):
        for name, metadata in read_safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            tensor_count += 1
            numel = math.prod(metadata["shape"])
            dtype = metadata["dtype"]
            dtype_numel[dtype] += numel
            dtype_bytes[dtype] += numel * DTYPE_BYTES.get(dtype, 0)
            if name.endswith(".scale"):
                scale_numel += numel
            if ".experts." in name and name.endswith(".weight") and dtype == "I8":
                # DeepSeek-V4 stores two E2M1 FP4 values per int8 storage element.
                fp4_packed_weight_numel += numel

    remote_files = {
        name: (model_path / name).is_file()
        for name in (
            "configuration_deepseek_v4.py",
            "modeling_deepseek_v4.py",
        )
    }
    quant = config.get("quantization_config") or {}
    blockers = []
    if config.get("model_type") == "deepseek_v4" and not config.get("auto_map"):
        blockers.append(
            "No config.auto_map: trust_remote_code cannot supply a DeepSeek-V4 "
            "Transformers implementation."
        )
    if not all(remote_files.values()):
        blockers.append("The checkpoint does not contain Transformers remote-code model files.")
    if config.get("expert_dtype") == "fp4":
        blockers.append(
            "Routed experts are packed FP4; this is not a BF16 full-training checkpoint."
        )
    if quant.get("quant_method") == "fp8":
        blockers.append(
            "The remaining linear weights use checkpoint-native FP8 quantization."
        )

    # V4-Flash's main network is approximately 285B parameters.  The checkpoint
    # also contains DSpark/speculative weights, which is why counting all logical
    # packed values produces a larger figure.
    main_parameter_billions = 285.0
    memory = {
        "main_parameter_billions": main_parameter_billions,
        "bf16_weights_plus_gradients_plus_adam_states_total_gb_lower_bound": (
            main_parameter_billions * 12
        ),
        "bf16_weights_plus_gradients_plus_adam_states_per_gpu_gb_lower_bound": (
            main_parameter_billions * 12 / gpus
        ),
        "eight_bit_optimizer_total_gb_lower_bound": main_parameter_billions * 6,
        "eight_bit_optimizer_per_gpu_gb_lower_bound": main_parameter_billions * 6 / gpus,
        "per_gpu_capacity_gb": per_gpu_memory_gb,
        "includes_activations": False,
    }
    if memory[
        "bf16_weights_plus_gradients_plus_adam_states_per_gpu_gb_lower_bound"
    ] > per_gpu_memory_gb:
        blockers.append(
            f"Even ideal {gpus}-way sharding exceeds {per_gpu_memory_gb:g} GB/GPU "
            "before activations for ordinary full-parameter Adam training."
        )

    return {
        "model_path": str(model_path),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "config_transformers_version": config.get("transformers_version"),
        "auto_map": config.get("auto_map"),
        "remote_code_files": remote_files,
        "expert_dtype": config.get("expert_dtype"),
        "quantization_config": quant,
        "stored_total_bytes_from_index": index.get("metadata", {}).get("total_size"),
        "tensor_count": tensor_count,
        "storage_numel_by_dtype": dict(dtype_numel),
        "storage_bytes_by_dtype": dict(dtype_bytes),
        "fp4_packed_expert_storage_numel": fp4_packed_weight_numel,
        "fp4_logical_expert_parameter_numel": fp4_packed_weight_numel * 2,
        "quantization_scale_numel": scale_numel,
        "memory_lower_bounds": memory,
        "full_parameter_sft_supported": False,
        "blockers": blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--per-gpu-memory-gb", type=float, default=141.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = inspect_checkpoint(
        Path(args.model_path), args.gpus, args.per_gpu_memory_gb
    )
    print(json.dumps(result, indent=2, sort_keys=True))
