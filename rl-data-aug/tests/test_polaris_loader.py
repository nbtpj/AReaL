import unittest

from polaris_loader import build_grpo_trajectories_from_polaris, load_polaris_records
from training_config import TrainingStackConfig


class PolarisLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {"id": "p-1", "question": "Alice will solve 3 small tasks quickly.", "difficulty": "easy"},
            {"id": "p-2", "question": "Find the large answer and show your work.", "difficulty": "medium"},
        ]

    def test_load_polaris_records_normalizes_source_and_prompt(self) -> None:
        out = load_polaris_records(self.records)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["source_id"], "p-1")
        self.assertEqual(out[0]["prompt"], self.records[0]["question"])
        self.assertEqual(out[1]["source_id"], "p-2")
        self.assertEqual(out[1]["prompt"], self.records[1]["question"])

    def test_load_polaris_records_rejects_missing_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing question field"):
            load_polaris_records([{"id": "p-1"}])

    def test_load_polaris_records_rejects_missing_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing source field"):
            load_polaris_records([{"question": "Solve this."}])

    def test_build_trajectories_preserves_traceability_when_disabled(self) -> None:
        cfg = TrainingStackConfig(augmentation_enabled=False)
        out = build_grpo_trajectories_from_polaris(self.records, config=cfg, seed=13)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["source_id"], "p-1")
        self.assertEqual(out[1]["source_id"], "p-2")
        self.assertNotIn("augmentation", out[0])

    def test_build_trajectories_preserves_traceability_when_enabled(self) -> None:
        cfg = TrainingStackConfig(
            augmentation_enabled=True,
            augmentation_operators=("synonym_substitution",),
            num_variants=2,
        )
        out = build_grpo_trajectories_from_polaris(
            self.records,
            config=cfg,
            seed=9,
            protected_tokens_by_source={"p-1": ("Alice",), "p-2": tuple()},
        )
        self.assertEqual(len(out), 4)
        for item in out:
            self.assertIn(item["source_id"], {"p-1", "p-2"})
            self.assertIn("augmentation", item)
            self.assertEqual(item["augmentation"]["source_id"], item["source_id"])


if __name__ == "__main__":
    unittest.main()
