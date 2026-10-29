import json
import tempfile
import unittest
from pathlib import Path

from deepseek_v4.normalize_merged_config import normalize


class NormalizeMergedConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.model, self.base = root / "merged", root / "base"
        self.model.mkdir()
        self.base.mkdir()
        self.config = {"num_hash_layers": 3, "num_hidden_layers": 5,
                       "mlp_layer_types": ["hash_moe"] * 3 + ["moe"] * 2,
                       "architectures": ["DeepseekV4ForCausalLM"]}
        self.write()
        (self.base / "config.json").write_text(json.dumps({"num_hash_layers": 3}))

    def write(self):
        (self.model / "config.json").write_text(json.dumps(self.config))

    def test_only_training_schema_is_removed_and_original_is_saved(self):
        original = (self.model / "config.json").read_bytes()
        weights = self.model / "model.safetensors"
        weights.write_bytes(b"unchanged weight payload")
        result = normalize(self.model, self.base)
        actual = json.loads((self.model / "config.json").read_text())
        self.assertEqual(actual, {k: v for k, v in self.config.items() if k != "mlp_layer_types"})
        self.assertEqual((self.model / "config.before-mlp-schema-repair.json").read_bytes(), original)
        self.assertEqual(weights.read_bytes(), b"unchanged weight payload")
        self.assertFalse(result["inference_gate_passed"])

    def test_unexpected_layout_is_rejected_without_changes(self):
        self.config["mlp_layer_types"][0] = "moe"
        self.write()
        original = (self.model / "config.json").read_bytes()
        with self.assertRaises(ValueError):
            normalize(self.model, self.base)
        self.assertEqual((self.model / "config.json").read_bytes(), original)
        self.assertFalse((self.model / "config.before-mlp-schema-repair.json").exists())

    def test_base_schema_must_not_already_define_the_field(self):
        (self.base / "config.json").write_text(json.dumps(self.config))
        with self.assertRaises(ValueError):
            normalize(self.model, self.base)
        self.assertEqual(json.loads((self.model / "config.json").read_text()), self.config)

    def test_existing_backup_is_preserved(self):
        backup = self.model / "config.before-mlp-schema-repair.json"
        backup.write_bytes(b"existing backup")
        with self.assertRaises(FileExistsError):
            normalize(self.model, self.base)
        self.assertEqual(backup.read_bytes(), b"existing backup")
        self.assertEqual(json.loads((self.model / "config.json").read_text()), self.config)
