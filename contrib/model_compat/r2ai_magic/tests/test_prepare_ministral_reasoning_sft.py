from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from r2ai_magic.scripts.prepare_ministral_reasoning_sft import (
    _convert_record,
    convert_file,
)


SYSTEM_PROMPT = "Think first: [THINK]scratch[/THINK] then answer."


class PrepareMinistralReasoningSFTTests(unittest.TestCase):
    def test_converts_magic_tags_and_combines_system_prompts(self) -> None:
        record = {
            "instruction": "Rewrite",
            "input": "prompt",
            "output": "<think>reason</think>\n<answer>rewrite</answer>",
            "system": "Red-team instructions",
            "sample_id": "sample-1",
        }
        converted, reason = _convert_record(record, SYSTEM_PROMPT)
        self.assertIsNone(reason)
        assert converted is not None
        self.assertEqual(converted["output"], "[THINK]reason[/THINK]rewrite")
        self.assertTrue(converted["system"].startswith(SYSTEM_PROMPT))
        self.assertIn("Red-team instructions", converted["system"])
        self.assertEqual(converted["reasoning_protocol"], "mistral_square_think_v1")

    def test_rejects_unclosed_thinking_trace(self) -> None:
        record = {"output": "<think>unfinished<answer>rewrite</answer>", "system": ""}
        converted, reason = _convert_record(record, SYSTEM_PROMPT)
        self.assertIsNone(converted)
        self.assertEqual(reason, "noncanonical_magic_output")

    def test_convert_file_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            output = root / "output.json"
            payload = [
                {
                    "output": "<think>reason</think><answer>answer</answer>",
                    "system": "task",
                },
                {"output": "<think>broken", "system": "task"},
            ]
            original_text = json.dumps(payload)
            source.write_text(original_text, encoding="utf-8")
            rejected, counts = convert_file(source, output, SYSTEM_PROMPT)
            self.assertEqual(source.read_text(encoding="utf-8"), original_text)
            self.assertEqual(counts, {"source": 2, "accepted": 1, "rejected": 1})
            self.assertEqual(rejected[0]["source_index"], 1)
            converted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(converted[0]["output"], "[THINK]reason[/THINK]answer")


if __name__ == "__main__":
    unittest.main()
