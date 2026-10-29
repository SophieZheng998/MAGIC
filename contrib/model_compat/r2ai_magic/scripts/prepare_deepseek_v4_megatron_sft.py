from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the audited MAGIC split to Megatron Bridge local-jsonl SFT."
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}.")
    required = ("system", "instruction", "input", "output")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} in {path} is not an object.")
        for field in required:
            if not isinstance(record.get(field), str):
                raise ValueError(f"Record {index} in {path} has invalid {field!r}.")
        if not record["output"].strip():
            raise ValueError(f"Record {index} in {path} has an empty output.")
    return records


def _convert(record: dict[str, Any]) -> dict[str, str]:
    system = record["system"].strip()
    user = record["instruction"].strip()
    query = record["input"].strip()
    if query:
        user = f"{user}\n\n{query}"

    # DeepSeek-V4 does not ship a Jinja chat template. These are the exact role
    # markers from the checkpoint's encoding/encoding_dsv4.py. We deliberately
    # do not append its chat/thinking transition marker: MAGIC supervises the
    # literal <think>...</think><answer>...</answer> completion.
    prefix = BOS
    if system:
        prefix += system
    prefix += USER + user + ASSISTANT
    return {"input": prefix, "output": record["output"].strip()}


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            for row in rows:
                payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                handle.write(payload)
                digest.update(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    train = [_convert(row) for row in _load(args.train_file)]
    validation = [_convert(row) for row in _load(args.validation_file)]
    train_sha = _write_jsonl(args.output_dir / "training.jsonl", train)
    validation_sha = _write_jsonl(args.output_dir / "validation.jsonl", validation)
    manifest = {
        "schema": "megatron-bridge-local-jsonl-prompt-completion-v1",
        "chat_encoding": "deepseek-v4-official-role-markers-magic-literal-completion",
        "train_source": str(args.train_file.resolve()),
        "validation_source": str(args.validation_file.resolve()),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "training_sha256": train_sha,
        "validation_sha256": validation_sha,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
