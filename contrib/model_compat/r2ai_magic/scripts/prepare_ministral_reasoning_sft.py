from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


CANONICAL_MAGIC_OUTPUT = re.compile(
    r"^\s*<think>(?P<thinking>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-destructive Ministral Reasoning SFT view from prepared MAGIC data. "
            "Only canonical <think>...</think><answer>...</answer> records are accepted."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _convert_record(
    record: dict[str, Any], official_system_prompt: str
) -> tuple[dict[str, Any] | None, str | None]:
    output = record.get("output")
    if not isinstance(output, str):
        return None, "output_not_string"
    match = CANONICAL_MAGIC_OUTPUT.fullmatch(output)
    if match is None:
        return None, "noncanonical_magic_output"
    thinking = match.group("thinking").strip()
    answer = match.group("answer").strip()
    if not thinking:
        return None, "empty_thinking"
    if not answer:
        return None, "empty_answer"

    task_system_prompt = str(record.get("system") or "").strip()
    combined_system_prompt = official_system_prompt.strip()
    if task_system_prompt:
        combined_system_prompt += (
            "\n\n# TASK-SPECIFIC SYSTEM INSTRUCTIONS\n\n" + task_system_prompt
        )

    converted = dict(record)
    converted["system"] = combined_system_prompt
    converted["output"] = f"[THINK]{thinking}[/THINK]{answer}"
    converted["reasoning_protocol"] = "mistral_square_think_v1"
    return converted, None


def convert_file(
    source_path: Path, output_path: Path, official_system_prompt: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {source_path}.")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            rejected.append({"source_index": index, "reason": "record_not_object"})
            continue
        converted, reason = _convert_record(record, official_system_prompt)
        if reason is not None:
            rejected.append(
                {
                    "source_index": index,
                    "sample_id": record.get("sample_id"),
                    "reason": reason,
                }
            )
            continue
        assert converted is not None
        accepted.append(converted)

    _write_json_atomic(output_path, accepted)
    return rejected, {"source": len(payload), "accepted": len(accepted), "rejected": len(rejected)}


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    system_prompt_file = args.system_prompt_file.resolve()
    official_system_prompt = system_prompt_file.read_text(encoding="utf-8").strip()
    if "[THINK]" not in official_system_prompt or "[/THINK]" not in official_system_prompt:
        raise ValueError("The official system prompt does not contain the Mistral reasoning markers.")

    split_files = {
        "train": "magic_sft_train.json",
        "validation": "magic_sft_validation.json",
    }
    rejected_by_split: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, int]] = {}
    source_hashes: dict[str, str] = {}
    for split, filename in split_files.items():
        source_path = source_dir / filename
        rejected, split_counts = convert_file(
            source_path,
            output_dir / filename,
            official_system_prompt,
        )
        rejected_by_split[split] = rejected
        counts[split] = split_counts
        source_hashes[split] = _sha256(source_path)

    manifest = {
        "schema_version": "magic-ministral-reasoning-sft-v1",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "system_prompt_file": str(system_prompt_file),
        "system_prompt_sha256": _sha256(system_prompt_file),
        "source_sha256": source_hashes,
        "counts": counts,
        "transforms": [
            "prepend_official_reasoning_system_prompt",
            "append_task_specific_system_prompt",
            "convert_<think>_to_[THINK]",
            "remove_<answer>_wrapper",
            "reject_noncanonical_outputs",
        ],
        "rejected_records": rejected_by_split,
        "source_files_are_unchanged": True,
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
