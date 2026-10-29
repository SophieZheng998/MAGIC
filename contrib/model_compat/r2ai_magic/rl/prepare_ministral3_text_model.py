#!/usr/bin/env python3
"""Create a canonical text-only Ministral-3 checkpoint without loading tensors.

The source may be either the native Mistral multimodal checkpoint or the
Transformers checkpoint emitted by ``fsdp_sft.py``.  Tensor bytes are copied
unchanged while multimodal tensors are omitted and language-model names are
normalized to the ``Ministral3ForCausalLM`` namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any


TEXT_ARCHITECTURE = "MistralForCausalLM"
TEXT_MODEL_TYPE = "mistral"


def _read_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        raw_header = handle.read(header_size)
    if len(raw_header) != header_size:
        raise ValueError(f"truncated safetensors header: {path}")
    return 8 + header_size, json.loads(raw_header.decode("utf-8"))


def _source_files(source: Path) -> dict[str, Path]:
    single = source / "model.safetensors"
    if single.is_file():
        _, header = _read_header(single)
        return {name: single for name in header if name != "__metadata__"}
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"no safetensors checkpoint in {source}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid weight map: {index_path}")
    return {name: source / shard for name, shard in weight_map.items()}


def _canonical_name(name: str) -> str | None:
    mappings = (
        ("model.language_model.model.", "model."),
        ("model.language_model.", "model."),
        ("language_model.model.", "model."),
        ("language_model.lm_head.", "lm_head."),
    )
    if name.startswith("lm_head."):
        return name
    for prefix, replacement in mappings:
        if name.startswith(prefix):
            return replacement + name[len(prefix) :]
    return None


def _copy_range(source: Path, offset: int, size: int, output: Any) -> None:
    with source.open("rb") as handle:
        handle.seek(offset)
        remaining = size
        while remaining:
            block = handle.read(min(16 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"truncated tensor data in {source}")
            output.write(block)
            remaining -= len(block)


def _metadata(source_files: dict[str, Path]) -> dict[str, tuple[Path, int, dict[str, Any]]]:
    headers: dict[Path, tuple[int, dict[str, Any]]] = {}
    result = {}
    for name, path in source_files.items():
        if path not in headers:
            headers[path] = _read_header(path)
        data_start, header = headers[path]
        if name not in header:
            raise KeyError(f"{name} is absent from {path}")
        result[name] = (path, data_start, header[name])
    return result


def convert(source: Path, tokenizer_source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    tokenizer_source = tokenizer_source.resolve()
    if output.exists():
        raise FileExistsError(output)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError(f"expected a multimodal Ministral-3 config in {source}")
    text_config = dict(text_config)
    text_config["architectures"] = [TEXT_ARCHITECTURE]
    text_config["model_type"] = TEXT_MODEL_TYPE
    text_config["tie_word_embeddings"] = False
    text_config["dtype"] = config.get("dtype", "bfloat16")

    source_files = _source_files(source)
    source_meta = _metadata(source_files)
    selected: list[tuple[str, str, Path, int, dict[str, Any]]] = []
    seen: dict[str, str] = {}
    for original in sorted(source_files):
        canonical = _canonical_name(original)
        if canonical is None:
            continue
        previous = seen.get(canonical)
        if previous is not None:
            raise ValueError(
                f"canonical key collision for {canonical}: {previous} and {original}"
            )
        seen[canonical] = original
        path, data_start, metadata = source_meta[original]
        selected.append((canonical, original, path, data_start, metadata))

    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    missing = required - set(seen)
    if missing:
        raise ValueError(f"missing canonical text tensors: {sorted(missing)}")
    if len(selected) != 363:
        raise ValueError(f"expected 363 text tensors, found {len(selected)}")

    output.mkdir(parents=True)
    final_weights = output / "model.safetensors"
    temporary = output / ".model.safetensors.tmp"
    new_header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    cursor = 0
    for canonical, _, _, _, metadata in selected:
        old_start, old_end = metadata["data_offsets"]
        size = int(old_end) - int(old_start)
        item = {key: value for key, value in metadata.items() if key != "data_offsets"}
        item["data_offsets"] = [cursor, cursor + size]
        new_header[canonical] = item
        cursor += size
    encoded = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for _, _, path, data_start, metadata in selected:
            old_start, old_end = metadata["data_offsets"]
            _copy_range(path, data_start + int(old_start), int(old_end) - int(old_start), handle)
        handle.flush()
        os.fsync(handle.fileno())
    expected_size = 8 + len(encoded) + cursor
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(
            f"converted size mismatch: {temporary.stat().st_size} != {expected_size}"
        )
    _, validation_header = _read_header(temporary)
    validation_keys = {name for name in validation_header if name != "__metadata__"}
    if validation_keys != set(seen):
        raise RuntimeError("converted safetensors key validation failed")
    os.replace(temporary, final_weights)

    (output / "config.json").write_text(
        json.dumps(text_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        source_path = tokenizer_source / name
        if source_path.is_file():
            shutil.copy2(source_path, output / name)
    summary = {
        "source": str(source),
        "tokenizer_source": str(tokenizer_source),
        "output": str(output.resolve()),
        "architecture": TEXT_ARCHITECTURE,
        "model_type": TEXT_MODEL_TYPE,
        "tensor_count": len(selected),
        "tensor_bytes": cursor,
        "tie_word_embeddings": False,
    }
    (output / "text_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.tokenizer_source, args.output)


if __name__ == "__main__":
    main()
