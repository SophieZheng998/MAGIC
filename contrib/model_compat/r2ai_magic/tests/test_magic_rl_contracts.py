import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from r2ai_magic.rl.contracts import (
    attacker_messages,
    attacker_template_from_raw_prompt,
    defender_messages,
    extract_answer,
    format_reward,
    mistral_reasoning_api_messages,
    parse_native_reasoning_response,
    parse_guard_output,
    safety_reward,
    split_native_reasoning,
)
from r2ai_magic.rl.vllm_weight_patch import (
    adapt_pixtral_text_only_weight_name,
    iter_pixtral_text_only_weights,
)


class MagicRLContractTest(unittest.TestCase):
    def test_mistral_reasoning_api_messages_uses_closed_thinking_block(self):
        source = [
            {
                "role": "system",
                "content": "before[THINK]worked example[/THINK]after",
            },
            {"role": "user", "content": "question"},
        ]
        encoded = mistral_reasoning_api_messages(source)
        self.assertEqual(
            encoded[0]["content"],
            [
                {"type": "text", "text": "before"},
                {
                    "type": "thinking",
                    "thinking": "worked example",
                    "closed": True,
                },
                {"type": "text", "text": "after"},
            ],
        )
        self.assertEqual(encoded[1], source[1])
        self.assertIsInstance(source[0]["content"], str)

    def test_pixtral_text_only_weight_adaptation(self):
        self.assertIsNone(
            adapt_pixtral_text_only_weight_name("model.vision_tower.patch_conv.weight")
        )
        self.assertIsNone(
            adapt_pixtral_text_only_weight_name(
                "model.multi_modal_projector.linear_1.weight"
            )
        )
        self.assertEqual(
            adapt_pixtral_text_only_weight_name(
                "model.language_model.model.layers.0.self_attn.q_proj.weight"
            ),
            "language_model.model.layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(
            adapt_pixtral_text_only_weight_name(
                "model.language_model.embed_tokens.weight"
            ),
            "language_model.model.embed_tokens.weight",
        )
        self.assertEqual(
            adapt_pixtral_text_only_weight_name("lm_head.weight"),
            "language_model.lm_head.weight",
        )
        self.assertEqual(
            adapt_pixtral_text_only_weight_name("model.language_model.lm_head.weight"),
            "language_model.lm_head.weight",
        )

    def test_pixtral_weight_adaptation_rejects_head_collision(self):
        with self.assertRaisesRegex(RuntimeError, "ambiguous weight update"):
            list(
                iter_pixtral_text_only_weights(
                    [
                        ("lm_head.weight", object()),
                        ("language_model.lm_head.weight", object()),
                    ]
                )
            )

    def test_official_prompts(self):
        with patch.dict("os.environ", {"MAGIC_REASONING_SYSTEM_PROMPT_FILE": ""}):
            self.assertEqual(attacker_messages("x")[1]["content"], "Task:\nx")
            self.assertEqual(
                defender_messages("y"),
                [{"role": "user", "content": "Question:\ny"}],
            )
        self.assertEqual(attacker_template_from_raw_prompt([{"role": "user", "content": "z"}]), "z")

    def test_answer_and_exact_format(self):
        good = "<think>reason</think>\n<answer>visible</answer>"
        self.assertEqual(extract_answer(good), "visible")
        self.assertEqual(format_reward(good), 0.5)
        self.assertEqual(format_reward("<answer>x</answer><think>y</think>"), -0.5)
        self.assertEqual(format_reward("<think>x</think><answer>y</answer><answer>z</answer>"), -0.5)

    def test_ministral_native_reasoning_format(self):
        good = "[THINK]reason[/THINK]\nvisible"
        self.assertEqual(extract_answer(good), "visible")
        self.assertEqual(format_reward(good), 0.5)
        self.assertEqual(format_reward("[THINK]reason[/THINK]"), -0.5)

    def test_ministral_native_reasoning_uses_last_close_for_visibility(self):
        malformed = "[THINK]reason[/THINK]continued reasoning[/THINK]final"
        self.assertEqual(
            split_native_reasoning(malformed),
            ("reason[/THINK]continued reasoning", "final"),
        )
        self.assertEqual(extract_answer(malformed), "final")
        self.assertEqual(format_reward(malformed), -0.5)

    def test_native_response_parser_accepts_direct_response_with_marker(self):
        self.assertEqual(
            parse_native_reasoning_response("direct response"),
            ("", "direct response", "missing_native_markers"),
        )

    def test_native_response_parser_rejects_unfinished_reasoning_as_invisible(self):
        self.assertEqual(
            parse_native_reasoning_response("[THINK]unfinished"),
            ("", "", "missing_close_marker"),
        )

    def test_native_response_parser_accepts_complete_reasoning(self):
        self.assertEqual(
            parse_native_reasoning_response("[THINK]reason[/THINK]answer"),
            ("reason", "answer", None),
        )

    def test_reasoning_system_prompt_is_added_to_both_roles(self):
        with TemporaryDirectory() as directory:
            prompt_file = Path(directory) / "SYSTEM_PROMPT.txt"
            prompt_file.write_text("official reasoning prompt", encoding="utf-8")
            with patch.dict("os.environ", {"MAGIC_REASONING_SYSTEM_PROMPT_FILE": str(prompt_file)}):
                self.assertTrue(attacker_messages("x")[0]["content"].startswith("official reasoning prompt"))
                self.assertEqual(defender_messages("y")[0], {"role": "system", "content": "official reasoning prompt"})

    def test_guard_parser_is_case_insensitive(self):
        self.assertEqual(parse_guard_output("safety: unsafe\nRefusal: no"), ("Unsafe", "No"))

    def test_safety_reward_harmful_refusal(self):
        reward = safety_reward(
            "Safety: Unsafe",
            "Safety: Safe\nRefusal: Yes",
            data_type="vanilla_harmful",
        )
        self.assertEqual(reward.reward_harm, 1.0)
        self.assertEqual(reward.reward_refusal, 0.5)
        self.assertEqual(reward.total, 1.5)

    def test_safety_reward_benign_compliance(self):
        reward = safety_reward(
            "Safety: Safe",
            "Safety: Safe\nRefusal: No",
            data_type="vanilla_benign",
        )
        self.assertEqual(reward.total, 1.5)

    def test_strict_guard_contract(self):
        with self.assertRaises(ValueError):
            safety_reward("Safety: Safe", "Safety: Safe", data_type="vanilla_benign")


if __name__ == "__main__":
    unittest.main()
