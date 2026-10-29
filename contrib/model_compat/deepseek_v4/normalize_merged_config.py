"""Remove the training-only MLP schema from a compatible dense HF export.

Packaged from the original DeepSeek MAGIC merge-config repair; tensor files are
never opened or rewritten. The pre-repair configuration is retained alongside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def normalize(model_dir: Path, base_model: Path) -> dict:
    path = model_dir / "config.json"
    raw = path.read_bytes()
    base = json.loads((base_model / "config.json").read_text())
    config = json.loads(raw)
    if "mlp_layer_types" in base:
        raise ValueError("Base checkpoint already defines mlp_layer_types; refusing schema removal")
    if config.get("num_hash_layers") != 3 or base.get("num_hash_layers") != 3:
        raise ValueError("Expected the original three-hash-layer DeepSeek configuration")
    layers = config.get("mlp_layer_types")
    if not isinstance(layers, list) or len(layers) != config.get("num_hidden_layers"):
        raise ValueError("Missing or inconsistent mlp_layer_types")
    if layers[:3] != ["hash_moe"] * 3 or any(layer != "moe" for layer in layers[3:]):
        raise ValueError("Unexpected MLP layer layout; refusing schema removal")
    del config["mlp_layer_types"]
    updated = (json.dumps(config, indent=2) + "\n").encode()
    backup = path.with_name("config.before-mlp-schema-repair.json")
    with backup.open("xb") as handle:
        handle.write(raw)
    fd, temporary = tempfile.mkstemp(prefix=".config-mlp-repair-", dir=model_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(updated)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "before_sha256": hashlib.sha256(raw).hexdigest(),
        "after_sha256": hashlib.sha256(updated).hexdigest(),
        "backup": str(backup),
        "weight_files_changed": False,
        "inference_gate_passed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize(args.model_dir, args.base_model), indent=2))


if __name__ == "__main__":
    main()
