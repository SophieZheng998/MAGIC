from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from torch.utils.data import DistributedSampler

from r2ai_magic.fsdp_sft import (
    MagicSFTDataset,
    _checkpoint_weight_bytes,
    _is_expert_model_config,
    _is_mistral3_multimodal_config,
    _mistral3_checkpoint_has_separate_lm_head,
    _read_records,
    _resolve_model_loader,
    _tokenize_record,
)


class TinyChatTokenizer:
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


class NativeTemplateTokenizer(TinyChatTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize
        assert kwargs == {"model_flag": "test"}
        rendered = "<bos>" + "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def record() -> dict[str, str]:
    return {
        "instruction": "Rewrite this prompt.",
        "input": "Vanilla prompt",
        "output": "<think>reason</think>\n<answer>rewrite</answer>",
        "system": "Red-team system",
    }


class SFTTokenizationTests(unittest.TestCase):
    def test_distributed_eval_sampler_has_equal_steps_on_every_rank(self) -> None:
        dataset = list(range(2477))
        samplers = [
            DistributedSampler(dataset, num_replicas=4, rank=rank, shuffle=False, drop_last=False)
            for rank in range(4)
        ]
        self.assertEqual({len(sampler) for sampler in samplers}, {math.ceil(len(dataset) / 4)})
        self.assertEqual(sum(len(sampler) for sampler in samplers) - len(dataset), 3)

    def test_loss_mask_starts_at_literal_magic_output(self) -> None:
        tokenizer = TinyChatTokenizer()
        prefix, full = _tokenize_record(record(), tokenizer, "qwen_chatml_literal")
        supervised = "".join(chr(token) for token in full[len(prefix) :])
        self.assertEqual(supervised, "<think>reason</think>\n<answer>rewrite</answer><|im_end|>\n")

    def test_dataset_masks_prompt_and_keeps_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text(json.dumps([record()]), encoding="utf-8")
            dataset = MagicSFTDataset(
                path,
                TinyChatTokenizer(),
                cutoff_len=4096,
                template_mode="qwen_chatml_literal",
            )
            item = dataset[0]
            first_supervised = int((item["labels"] != -100).nonzero()[0])
            text = "".join(chr(token) for token in item["input_ids"][first_supervised:])
            self.assertTrue(text.startswith("<think>reason</think>"))

    def test_native_model_template_preserves_assistant_mask(self) -> None:
        prefix, full = _tokenize_record(
            record(),
            NativeTemplateTokenizer(),
            "tokenizer",
            {"model_flag": "test"},
        )
        supervised = "".join(chr(token) for token in full[len(prefix) :])
        self.assertEqual(
            supervised,
            "<think>reason</think>\n<answer>rewrite</answer></assistant>",
        )

    def test_tokenizer_mode_never_silently_falls_back(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Jinja2"):
            _tokenize_record(record(), TinyChatTokenizer(), "tokenizer")

    def test_reader_rejects_non_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text(json.dumps(record()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON array"):
                _read_records(path)

    def test_nonpersistent_buffers_are_not_checkpoint_requirements(self) -> None:
        import torch

        module = torch.nn.Module()
        module.register_buffer("cached_value", torch.ones(1), persistent=False)
        self.assertIn("cached_value", dict(module.named_buffers()))
        self.assertNotIn("cached_value", module.state_dict())

    def test_expert_models_are_detected_before_gpu_loading(self) -> None:
        dense = type("DenseConfig", (), {"num_local_experts": 0})()
        moe = type("MoeConfig", (), {"n_routed_experts": 256})()
        self.assertFalse(_is_expert_model_config(dense))
        self.assertTrue(_is_expert_model_config(moe))

    def test_checkpoint_size_uses_index_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 123456}, "weight_map": {}}),
                encoding="utf-8",
            )
            self.assertEqual(_checkpoint_weight_bytes(path), 123456)

    def test_mistral3_auto_selects_native_multimodal_loader(self) -> None:
        text_config = type("MinistralConfig", (), {"model_type": "ministral3"})()
        config = type("Mistral3Config", (), {"model_type": "mistral3", "text_config": text_config})()
        self.assertTrue(_is_mistral3_multimodal_config(config))
        self.assertEqual(_resolve_model_loader(config, "auto"), "multimodal_causal_lm")

    def test_explicit_loader_choice_is_not_overridden(self) -> None:
        config = type("Mistral3Config", (), {"model_type": "mistral3", "text_config": object()})()
        self.assertEqual(_resolve_model_loader(config, "generic_causal_lm"), "generic_causal_lm")

    def test_native_mistral3_index_detects_independent_lm_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "language_model.lm_head.weight": "model-1.safetensors",
                            "language_model.model.embed_tokens.weight": "model-1.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_mistral3_checkpoint_has_separate_lm_head(path))


if __name__ == "__main__":
    unittest.main()
