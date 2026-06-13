import unittest

from grpo_trajectory_generation import build_grpo_trajectory_inputs, validate_trajectory_record
from training_config import TrainingStackConfig


class GrpoTrajectoryGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            {"source_id": "q1", "prompt": "Alice will solve 3 small tasks quickly.", "meta": {"split": "train"}},
            {"source_id": "q2", "prompt": "Find the large answer and show your work.", "meta": {"split": "train"}},
        ]

    def test_disabled_augmentation_preserves_baseline_behavior(self) -> None:
        config = TrainingStackConfig(augmentation_enabled=False)

        out = build_grpo_trajectory_inputs(self.samples, config=config, seed=41)

        self.assertEqual(out, self.samples)
        self.assertIsNot(out, self.samples)
        self.assertIsNot(out[0], self.samples[0])

    def test_enabled_augmentation_expands_and_preserves_source_link(self) -> None:
        config = TrainingStackConfig(
            augmentation_enabled=True,
            augmentation_operators=("synonym_substitution",),
            num_variants=2,
        )

        out = build_grpo_trajectory_inputs(
            self.samples,
            config=config,
            seed=3,
            protected_tokens_by_source={"q1": ("Alice",), "q2": tuple()},
        )

        self.assertEqual(len(out), 4)
        for item in out:
            self.assertIn(item["source_id"], {"q1", "q2"})
            self.assertIn("augmentation", item)
            self.assertEqual(item["augmentation"]["enabled"], True)
            self.assertEqual(item["augmentation"]["source_id"], item["source_id"])
            self.assertEqual(item["augmentation"]["aug_op"], "synonym_substitution")
            self.assertIn("seed", item["augmentation"])
            self.assertIn("variant_index", item["augmentation"])

        q1_prompts = [item["prompt"] for item in out if item["source_id"] == "q1"]
        self.assertTrue(all("Alice" in prompt for prompt in q1_prompts))
        self.assertTrue(all("3" in prompt for prompt in q1_prompts))

    def test_enabled_augmentation_is_deterministic_for_same_seed(self) -> None:
        config = TrainingStackConfig(
            augmentation_enabled=True,
            augmentation_operators=("synonym_substitution",),
            num_variants=3,
        )
        first = build_grpo_trajectory_inputs(self.samples, config=config, seed=17)
        second = build_grpo_trajectory_inputs(self.samples, config=config, seed=17)
        self.assertEqual(first, second)

    def test_enabled_augmentation_cycles_three_operators(self) -> None:
        config = TrainingStackConfig(
            augmentation_enabled=True,
            augmentation_operators=("synonym_substitution", "word_shuffle", "typo_noise"),
            num_variants=3,
        )

        out = build_grpo_trajectory_inputs(
            self.samples[:1],
            config=config,
            seed=3,
            protected_tokens_by_source={"q1": ("Alice",)},
        )

        self.assertEqual(len(out), 3)
        self.assertEqual(
            [item["augmentation"]["aug_op"] for item in out],
            ["synonym_substitution", "word_shuffle", "typo_noise"],
        )
        self.assertTrue(all(item["source_id"] == "q1" for item in out))

    def test_rejects_missing_source_id(self) -> None:
        config = TrainingStackConfig(
            augmentation_enabled=True,
            augmentation_operators=("synonym_substitution",),
            num_variants=1,
        )
        invalid_samples = [{"prompt": "Solve this."}]
        with self.assertRaisesRegex(ValueError, "source_id"):
            build_grpo_trajectory_inputs(invalid_samples, config=config, seed=1)

    def test_rejects_missing_prompt(self) -> None:
        config = TrainingStackConfig(
            augmentation_enabled=True,
            augmentation_operators=("synonym_substitution",),
            num_variants=1,
        )
        invalid_samples = [{"source_id": "q1"}]
        with self.assertRaisesRegex(ValueError, "prompt"):
            build_grpo_trajectory_inputs(invalid_samples, config=config, seed=1)

    def test_rejects_unsupported_operator(self) -> None:
        config = TrainingStackConfig(
            augmentation_enabled=True,
            augmentation_operators=("unknown_op",),
            num_variants=1,
        )
        with self.assertRaisesRegex(ValueError, "unsupported augmentation operators"):
            build_grpo_trajectory_inputs(self.samples, config=config, seed=1)

    def test_validate_record_rejects_missing_aug_op(self) -> None:
        record = {
            "source_id": "q1",
            "prompt": "Solve this.",
            "augmentation": {"source_id": "q1", "seed": 10},
        }
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_trajectory_record(record)

    def test_validate_record_rejects_non_int_seed(self) -> None:
        record = {
            "source_id": "q1",
            "prompt": "Solve this.",
            "augmentation": {"source_id": "q1", "aug_op": "synonym_substitution", "seed": "10"},
        }
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            validate_trajectory_record(record)

    def test_validate_record_rejects_mismatched_source_id(self) -> None:
        record = {
            "source_id": "q1",
            "prompt": "Solve this.",
            "augmentation": {"source_id": "q2", "aug_op": "synonym_substitution", "seed": 10},
        }
        with self.assertRaisesRegex(ValueError, "source_id must match"):
            validate_trajectory_record(record)

    def test_enabled_augmentation_rejects_no_op_prompt(self) -> None:
        config = TrainingStackConfig(
            augmentation_enabled=True,
            augmentation_operators=("synonym_substitution",),
            num_variants=1,
        )
        samples = [{"source_id": "q1", "prompt": "123 456"}]
        with self.assertRaisesRegex(ValueError, "no eligible token available"):
            build_grpo_trajectory_inputs(samples, config=config, seed=5)


if __name__ == "__main__":
    unittest.main()
