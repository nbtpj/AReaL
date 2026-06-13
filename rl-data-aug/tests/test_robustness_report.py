import json
from pathlib import Path
import tempfile
import unittest

from robustness_report import (
    generate_baseline_anchored_report,
    run_baseline_and_augmented_report,
    run_polaris_experiment,
    write_artifact,
)
from training_config import TrainingStackConfig


class RobustnessReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {"id": "p-1", "question": "Alice will solve 3 small tasks quickly.", "difficulty": "easy"},
            {"id": "p-2", "question": "Find the large answer and show your work.", "difficulty": "medium"},
            {"id": "p-3", "question": "Solve 5 compact riddles quickly.", "difficulty": "easy"},
        ]
        self.baseline_cfg = TrainingStackConfig(augmentation_enabled=False)
        self.augmented_cfg = TrainingStackConfig(
            augmentation_enabled=True,
            augmentation_operators=("synonym_substitution",),
            num_variants=2,
        )
        self.runtime_prefix = Path.cwd() / ".venv"

    def test_generate_report_for_baseline_and_augmented_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            baseline = run_polaris_experiment(
                raw_records=self.records,
                config=self.baseline_cfg,
                seed=11,
                robustness_split_value="easy",
                protected_tokens_by_source={"p-1": ("Alice",)},
                runtime_prefix=self.runtime_prefix,
            )
            augmented = run_polaris_experiment(
                raw_records=self.records,
                config=self.augmented_cfg,
                seed=11,
                robustness_split_value="easy",
                protected_tokens_by_source={"p-1": ("Alice",)},
                runtime_prefix=self.runtime_prefix,
            )

            baseline_path = write_artifact(baseline, output_dir / "baseline.json")
            augmented_path = write_artifact(augmented, output_dir / "augmented.json")
            report_path = generate_baseline_anchored_report(
                baseline_artifact_path=baseline_path,
                augmented_artifact_path=augmented_path,
                output_path=output_dir / "report.json",
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(report["stack"]["base_model"], "Qwen3-1.7B")
            self.assertEqual(report["stack"]["runtime_env"], ".venv")
            self.assertEqual(
                report["stack"]["validation_dataset_path"],
                "/storage/openpsi/users/zzy/sync/AIME24_converted_copy.parquet",
            )
            self.assertEqual(report["stack"]["n_rollout"], 8)
            self.assertEqual(report["stack"]["max_output_tokens"], 40960)
            self.assertEqual(report["stack"]["batch_size"], 128)
            self.assertEqual(report["stack"]["lr"], 5e-6)
            self.assertEqual(report["run_context"]["robustness_split"]["value"], "easy")
            self.assertEqual(report["run_context"]["baseline_seed"], 11)
            self.assertEqual(report["run_context"]["augmented_seed"], 11)
            self.assertIn("overall_proxy_quality", report["deltas"])
            self.assertIn("reproducibility", report)
            self.assertIn("baseline_sha256", report["reproducibility"])

    def test_run_baseline_and_augmented_report_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = run_baseline_and_augmented_report(
                raw_records=self.records,
                output_dir=tmpdir,
                baseline_seed=3,
                augmented_seed=5,
                robustness_split_value="easy",
                protected_tokens_by_source={"p-1": ("Alice",)},
                runtime_prefix=self.runtime_prefix,
            )
            self.assertTrue(paths["baseline"].exists())
            self.assertTrue(paths["augmented"].exists())
            self.assertTrue(paths["report"].exists())
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["run_context"]["baseline_seed"], 3)
            self.assertEqual(report["run_context"]["augmented_seed"], 5)

    def test_report_rejects_missing_baseline_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            baseline = run_polaris_experiment(
                raw_records=self.records,
                config=self.baseline_cfg,
                seed=7,
                runtime_prefix=self.runtime_prefix,
            )
            augmented = run_polaris_experiment(
                raw_records=self.records,
                config=self.augmented_cfg,
                seed=7,
                runtime_prefix=self.runtime_prefix,
            )
            del baseline["metrics"]["overall_proxy_quality"]

            baseline_path = write_artifact(baseline, output_dir / "baseline.json")
            augmented_path = write_artifact(augmented, output_dir / "augmented.json")
            with self.assertRaisesRegex(ValueError, "artifact metrics missing required keys"):
                generate_baseline_anchored_report(
                    baseline_artifact_path=baseline_path,
                    augmented_artifact_path=augmented_path,
                    output_path=output_dir / "report.json",
                )

    def test_report_rejects_stack_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            baseline = run_polaris_experiment(
                raw_records=self.records,
                config=self.baseline_cfg,
                seed=13,
                runtime_prefix=self.runtime_prefix,
            )
            augmented = run_polaris_experiment(
                raw_records=self.records,
                config=self.augmented_cfg,
                seed=13,
                runtime_prefix=self.runtime_prefix,
            )
            augmented["stack"]["dataset"] = "other"

            baseline_path = write_artifact(baseline, output_dir / "baseline.json")
            augmented_path = write_artifact(augmented, output_dir / "augmented.json")
            with self.assertRaisesRegex(ValueError, "pinned stack values"):
                generate_baseline_anchored_report(
                    baseline_artifact_path=baseline_path,
                    augmented_artifact_path=augmented_path,
                    output_path=output_dir / "report.json",
                )

    def test_report_rejects_missing_baseline_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            augmented = run_polaris_experiment(
                raw_records=self.records,
                config=self.augmented_cfg,
                seed=5,
                runtime_prefix=self.runtime_prefix,
            )
            augmented_path = write_artifact(augmented, output_dir / "augmented.json")
            missing_baseline = output_dir / "missing-baseline.json"

            with self.assertRaisesRegex(ValueError, "artifact not found"):
                generate_baseline_anchored_report(
                    baseline_artifact_path=missing_baseline,
                    augmented_artifact_path=augmented_path,
                    output_path=output_dir / "report.json",
                )

    def test_report_rejects_equally_wrong_pinned_stack_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            baseline = run_polaris_experiment(
                raw_records=self.records,
                config=self.baseline_cfg,
                seed=19,
                runtime_prefix=self.runtime_prefix,
            )
            augmented = run_polaris_experiment(
                raw_records=self.records,
                config=self.augmented_cfg,
                seed=19,
                runtime_prefix=self.runtime_prefix,
            )
            baseline["stack"]["max_output_tokens"] = 8
            augmented["stack"]["max_output_tokens"] = 8

            baseline_path = write_artifact(baseline, output_dir / "baseline.json")
            augmented_path = write_artifact(augmented, output_dir / "augmented.json")
            with self.assertRaisesRegex(ValueError, "must match pinned stack values"):
                generate_baseline_anchored_report(
                    baseline_artifact_path=baseline_path,
                    augmented_artifact_path=augmented_path,
                    output_path=output_dir / "report.json",
                )


if __name__ == "__main__":
    unittest.main()
