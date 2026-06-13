import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_AUGMENTATION_MODULE_PATH = PROJECT_ROOT / "verl" / "verl" / "utils" / "dataset" / "augmentation.py"
_AUGMENTATION_SPEC = importlib.util.spec_from_file_location("rl_dataset_augmentation", _AUGMENTATION_MODULE_PATH)
_AUGMENTATION_MODULE = importlib.util.module_from_spec(_AUGMENTATION_SPEC)
assert _AUGMENTATION_SPEC is not None and _AUGMENTATION_SPEC.loader is not None
sys.modules[_AUGMENTATION_SPEC.name] = _AUGMENTATION_MODULE
_AUGMENTATION_SPEC.loader.exec_module(_AUGMENTATION_MODULE)

RLDataAugmentationConfig = _AUGMENTATION_MODULE.RLDataAugmentationConfig
augment_rlhf_records = _AUGMENTATION_MODULE.augment_rlhf_records
build_augmentation_config = _AUGMENTATION_MODULE.build_augmentation_config
validate_augmentation_provenance = _AUGMENTATION_MODULE.validate_augmentation_provenance


class VerlDatasetAugmentationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "source_id": "q1",
                "prompt": [{"role": "user", "content": "Alice will solve 3 small tasks quickly."}],
                "question_id": "Alice",
            },
            {
                "source_id": "q2",
                "prompt": [{"role": "user", "content": "Find the large answer and show your work."}],
                "question_id": "q2",
            },
        ]

    def test_disabled_augmentation_preserves_records(self) -> None:
        config = RLDataAugmentationConfig(enabled=False, num_variants=2)
        out = augment_rlhf_records(records=self.records, prompt_key="prompt", config=config)
        self.assertEqual(out, self.records)
        self.assertIsNot(out, self.records)

    def test_enabled_augmentation_expands_records_with_provenance(self) -> None:
        config = RLDataAugmentationConfig(
            enabled=True,
            operators=("synonym_substitution",),
            num_variants=2,
            seed=11,
            protected_fields=("question_id",),
        )
        out = augment_rlhf_records(records=self.records, prompt_key="prompt", config=config)

        self.assertEqual(len(out), 4)
        for item in out:
            self.assertIn("source_id", item)
            self.assertIn("extra_info", item)
            self.assertIn("augmentation", item["extra_info"])
            augmentation = item["extra_info"]["augmentation"]
            validate_augmentation_provenance(augmentation)
            self.assertEqual(augmentation["dataset"], "polaris")
            self.assertEqual(augmentation["rl_framework"], "verl")
            self.assertEqual(augmentation["base_model"], "Qwen3-1.7B")
            self.assertEqual(item["source_id"], augmentation["source_id"])

    def test_enabled_augmentation_is_deterministic(self) -> None:
        config = RLDataAugmentationConfig(
            enabled=True,
            operators=("synonym_substitution",),
            num_variants=3,
            seed=5,
            protected_fields=("question_id",),
        )
        first = augment_rlhf_records(records=self.records, prompt_key="prompt", config=config)
        second = augment_rlhf_records(records=self.records, prompt_key="prompt", config=config)
        self.assertEqual(first, second)

    def test_enabled_augmentation_supports_three_modes(self) -> None:
        records = [
            {
                "source_id": "q1",
                "prompt": [
                    {
                        "role": "user",
                        "content": "Alice will solve 3 small tasks quickly and show the large result.",
                    }
                ],
                "question_id": "Alice",
            }
        ]
        config = RLDataAugmentationConfig(
            enabled=True,
            operators=("synonym_substitution", "word_shuffle", "typo_noise"),
            num_variants=3,
            seed=4,
            protected_fields=("question_id",),
        )

        out = augment_rlhf_records(records=records, prompt_key="prompt", config=config)

        self.assertEqual(len(out), 3)
        self.assertEqual(
            [item["extra_info"]["augmentation"]["aug_op"] for item in out],
            ["synonym_substitution", "word_shuffle", "typo_noise"],
        )
        self.assertTrue(all("Alice" in item["prompt"][0]["content"] for item in out))
        self.assertTrue(all("3" in item["prompt"][0]["content"] for item in out))

    def test_rejects_unsupported_operator(self) -> None:
        config = RLDataAugmentationConfig(
            enabled=True,
            operators=("unsupported",),
            num_variants=1,
        )
        with self.assertRaisesRegex(ValueError, "unsupported augmentation operators"):
            augment_rlhf_records(records=self.records, prompt_key="prompt", config=config)

    def test_rejects_similarity_below_threshold(self) -> None:
        records = [
            {
                "source_id": "q1",
                "prompt": [{"role": "user", "content": "solve quickly"}],
            }
        ]
        config = RLDataAugmentationConfig(
            enabled=True,
            operators=("synonym_substitution",),
            num_variants=1,
            semantic_similarity_threshold=0.99,
        )
        with self.assertRaisesRegex(ValueError, "semantic similarity below threshold"):
            augment_rlhf_records(records=records, prompt_key="prompt", config=config)

    def test_rejects_missing_source_id_field(self) -> None:
        records = [{"prompt": [{"role": "user", "content": "solve quickly"}]}]
        config = RLDataAugmentationConfig(
            enabled=True,
            operators=("synonym_substitution",),
            num_variants=1,
            source_id_field="missing_id",
        )
        with self.assertRaisesRegex(ValueError, "missing source identifier field"):
            augment_rlhf_records(records=records, prompt_key="prompt", config=config)

    def test_resolves_nested_source_id_field(self) -> None:
        records = [
            {
                "prompt": [{"role": "user", "content": "Find the large answer and show your work."}],
                "extra_info": {"index": 123, "split": "train"},
            }
        ]
        config = RLDataAugmentationConfig(
            enabled=True,
            operators=("synonym_substitution",),
            num_variants=1,
            source_id_field="extra_info.index",
        )
        out = augment_rlhf_records(records=records, prompt_key="prompt", config=config)
        self.assertEqual(out[0]["source_id"], "123")
        self.assertEqual(out[0]["extra_info"]["augmentation"]["source_id"], "123")

    def test_augments_last_user_message_not_system(self) -> None:
        records = [
            {
                "source_id": "q1",
                "prompt": [
                    {"role": "system", "content": "Please solve safely."},
                    {"role": "user", "content": "Find the large answer quickly."},
                ],
            }
        ]
        config = RLDataAugmentationConfig(
            enabled=True,
            operators=("synonym_substitution",),
            num_variants=1,
        )
        out = augment_rlhf_records(records=records, prompt_key="prompt", config=config)
        self.assertEqual(out[0]["prompt"][0]["content"], "Please solve safely.")
        self.assertNotEqual(out[0]["prompt"][1]["content"], "Find the large answer quickly.")

    def test_rejects_prompt_without_user_text(self) -> None:
        records = [
            {
                "source_id": "q1",
                "prompt": [{"role": "system", "content": "Find the large answer quickly."}],
            }
        ]
        config = RLDataAugmentationConfig(
            enabled=True,
            operators=("synonym_substitution",),
            num_variants=1,
        )
        with self.assertRaisesRegex(ValueError, "no user text content found"):
            augment_rlhf_records(records=records, prompt_key="prompt", config=config)

    def test_rejects_no_eligible_token_for_augmentation(self) -> None:
        records = [
            {
                "source_id": "q1",
                "prompt": [{"role": "user", "content": "123 456"}],
            }
        ]
        config = RLDataAugmentationConfig(
            enabled=True,
            operators=("synonym_substitution",),
            num_variants=1,
        )
        with self.assertRaisesRegex(ValueError, "no eligible token available"):
            augment_rlhf_records(records=records, prompt_key="prompt", config=config)

    def test_rejects_invalid_stack_identifiers(self) -> None:
        base_kwargs = {
            "enabled": True,
            "operators": ("synonym_substitution",),
            "num_variants": 1,
        }
        invalid_framework = RLDataAugmentationConfig(**base_kwargs, rl_framework="other")
        invalid_dataset = RLDataAugmentationConfig(**base_kwargs, dataset="gsm8k")
        invalid_model = RLDataAugmentationConfig(**base_kwargs, base_model="Llama")

        with self.assertRaisesRegex(ValueError, "rl_framework must be 'verl'"):
            augment_rlhf_records(records=self.records, prompt_key="prompt", config=invalid_framework)
        with self.assertRaisesRegex(ValueError, "dataset must be 'polaris'"):
            augment_rlhf_records(records=self.records, prompt_key="prompt", config=invalid_dataset)
        with self.assertRaisesRegex(ValueError, "base_model must be 'Qwen3-1.7B'"):
            augment_rlhf_records(records=self.records, prompt_key="prompt", config=invalid_model)

    def test_build_augmentation_config_parses_dict(self) -> None:
        config = build_augmentation_config(
            {
                "augmentation": {
                    "enabled": True,
                    "operators": ["synonym_substitution"],
                    "num_variants": 2,
                    "seed": 17,
                    "source_id_field": "my_id",
                    "protected_fields": ["entity"],
                    "semantic_similarity_threshold": 0.7,
                    "rl_framework": "verl",
                    "dataset": "polaris",
                    "base_model": "Qwen3-1.7B",
                }
            }
        )
        self.assertEqual(config.enabled, True)
        self.assertEqual(config.operators, ("synonym_substitution",))
        self.assertEqual(config.num_variants, 2)
        self.assertEqual(config.seed, 17)
        self.assertEqual(config.source_id_field, "my_id")
        self.assertEqual(config.protected_fields, ("entity",))
        self.assertEqual(config.semantic_similarity_threshold, 0.7)

    def test_build_config_defaults_to_polaris_nested_source_id(self) -> None:
        config = build_augmentation_config({"augmentation": {}})
        self.assertEqual(config.source_id_field, "extra_info.index")

    def test_build_config_accepts_three_augmentation_modes(self) -> None:
        config = build_augmentation_config(
            {
                "augmentation": {
                    "enabled": True,
                    "operators": ["synonym_substitution", "word_shuffle", "typo_noise"],
                }
            }
        )
        self.assertEqual(config.operators, ("synonym_substitution", "word_shuffle", "typo_noise"))

    def test_build_config_rejects_invalid_stack_even_when_disabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "rl_framework must be 'verl'"):
            build_augmentation_config(
                {"augmentation": {"enabled": False, "rl_framework": "other"}}
            )

    def test_rlhf_dataset_integration_disabled_and_enabled(self) -> None:
        rl_dataset_cls = _load_rlhf_dataset_class()

        class DummyTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **_kwargs):
                text = " ".join(
                    str(message.get("content", ""))
                    for message in messages
                    if isinstance(message, dict)
                )
                tokens = [token for token in text.split() if token]
                return list(range(max(1, len(tokens))))

        records = [
            {
                "data_source": "polaris",
                "prompt": [
                    {"role": "system", "content": "Follow instructions."},
                    {"role": "user", "content": "Find the large answer and show your work."},
                ],
                "extra_info": {"index": 0, "split": "train"},
            },
            {
                "data_source": "polaris",
                "prompt": [
                    {"role": "user", "content": "Alice will solve 3 small tasks quickly."},
                ],
                "extra_info": {"index": 1, "split": "train"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "fixture.jsonl"
            with data_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            # disabled parity
            disabled_cfg = _dict_config(
                {
                    "prompt_key": "prompt",
                    "filter_overlong_prompts": False,
                    "max_prompt_length": 1024,
                    "augmentation": {"enabled": False},
                }
            )
            disabled_dataset = rl_dataset_cls(
                data_files=str(data_path),
                tokenizer=DummyTokenizer(),
                config=disabled_cfg,
            )
            self.assertEqual(len(disabled_dataset), 2)
            disabled_item = disabled_dataset[0]
            self.assertNotIn("augmentation", disabled_item.get("extra_info", {}))

            # enabled expansion
            enabled_cfg = _dict_config(
                {
                    "prompt_key": "prompt",
                    "filter_overlong_prompts": False,
                    "max_prompt_length": 1024,
                    "augmentation": {
                        "enabled": True,
                        "operators": ["synonym_substitution"],
                        "num_variants": 2,
                        "seed": 3,
                        "source_id_field": "extra_info.index",
                    },
                }
            )
            enabled_dataset = rl_dataset_cls(
                data_files=str(data_path),
                tokenizer=DummyTokenizer(),
                config=enabled_cfg,
            )
            self.assertEqual(len(enabled_dataset), 4)
            enabled_item = enabled_dataset[0]
            self.assertIn("source_id", enabled_item)
            self.assertIn("augmentation", enabled_item["extra_info"])
            self.assertIn(enabled_item["source_id"], {"0", "1"})


if __name__ == "__main__":
    unittest.main()


def _dict_config(values: dict):
    class _SimpleConfig(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    return _SimpleConfig(values)


def _load_rlhf_dataset_class():
    rl_dataset_path = PROJECT_ROOT / "verl" / "verl" / "utils" / "dataset" / "rl_dataset.py"

    # Stub the minimal package namespace needed by rl_dataset imports without
    # importing the full verl package (which may require optional dependencies).
    verl_mod = types.ModuleType("verl")
    utils_mod = types.ModuleType("verl.utils")
    dataset_mod = types.ModuleType("verl.utils.dataset")
    import_utils_mod = types.ModuleType("verl.utils.import_utils")
    tokenizer_mod = types.ModuleType("verl.utils.tokenizer")
    fs_mod = types.ModuleType("verl.utils.fs")

    import_utils_mod.load_extern_object = lambda _path, _name: None
    tokenizer_mod.normalize_token_ids = lambda tokenized_prompt: tokenized_prompt
    fs_mod.copy_to_local = lambda src, cache_dir=None, use_shm=False: src

    sys.modules.setdefault("verl", verl_mod)
    sys.modules.setdefault("verl.utils", utils_mod)
    sys.modules.setdefault("verl.utils.dataset", dataset_mod)
    sys.modules.setdefault("verl.utils.import_utils", import_utils_mod)
    sys.modules.setdefault("verl.utils.tokenizer", tokenizer_mod)
    sys.modules.setdefault("verl.utils.fs", fs_mod)
    sys.modules["verl.utils.dataset.augmentation"] = _AUGMENTATION_MODULE

    spec = importlib.util.spec_from_file_location("rl_dataset_under_test", rl_dataset_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RLHFDataset
