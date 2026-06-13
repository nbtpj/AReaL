import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from openpsi_stack_dry_run import (
    generate_openpsi_baseline_vs_augmented_report,
    run_openpsi_baseline_augmented_report,
    run_openpsi_stack_dry_run,
)
from training_config import (
    PINNED_BATCH_SIZE,
    PINNED_LR,
    PINNED_MAX_OUTPUT_TOKENS,
    PINNED_N_ROLLOUT,
    PINNED_RUNTIME_ENV,
    PINNED_VALIDATION_DATASET_PATH,
)


_PINNED_MODEL_PATH = Path("/storage/openpsi/models/Qwen__Qwen3-1.7B")


class _DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(token) for token in text.split()] or [0]


class _DummyModel:
    pass


def _dummy_generation_fn(
    prompt_text: str,
    *,
    tokenizer,
    model,
    ground_truth: str | None,
    max_new_tokens: int,
) -> str:
    del prompt_text, tokenizer, model, max_new_tokens
    return ground_truth or "unknown"


def _fixture_dataframe() -> pd.DataFrame:
    records = [
        {
            "data_source": "polaris",
            "prompt": [
                {"role": "system", "content": "Follow instructions."},
                {"role": "user", "content": "Find the large answer and show your work."},
            ],
            "label": 1,
            "difficulty": "0/8",
            "ability": "math",
            "reward_model": {"ground_truth": "A1", "style": "rule"},
            "extra_info": {"index": 1, "split": "train"},
        },
        {
            "data_source": "polaris",
            "prompt": [
                {"role": "user", "content": "Alice will solve 3 small tasks quickly."},
            ],
            "label": 2,
            "difficulty": "0/8",
            "ability": "math",
            "reward_model": {"ground_truth": "A2", "style": "rule"},
            "extra_info": {"index": 2, "split": "train"},
        },
        {
            "data_source": "polaris",
            "prompt": [
                {"role": "user", "content": "Compute 17 plus 25 exactly."},
            ],
            "label": 42,
            "difficulty": "0/8",
            "ability": "math",
            "reward_model": {"ground_truth": "42", "style": "rule"},
            "extra_info": {"index": 3, "split": "train"},
        },
    ]
    return pd.DataFrame(records)


def _fixture_dataframe_without_extra_info_or_difficulty() -> pd.DataFrame:
    records = [
        {
            "prompt": np.array(
                [
                    {
                        "role": "user",
                        "content": "Find the large answer and explain each step clearly.",
                    }
                ],
                dtype=object,
            ),
            "label": 101,
            "data_source": "polaris",
            "reward_model": {"ground_truth": "101", "style": "rule"},
        },
        {
            "prompt": np.array(
                [
                    {
                        "role": "user",
                        "content": "Alice solves a small puzzle quickly and checks the answer.",
                    }
                ],
                dtype=object,
            ),
            "label": 202,
            "data_source": "polaris",
            "reward_model": {"ground_truth": "202", "style": "rule"},
        },
        {
            "prompt": np.array(
                [
                    {
                        "role": "user",
                        "content": "Choose the large result that matches all constraints.",
                    }
                ],
                dtype=object,
            ),
            "label": 303,
            "data_source": "polaris",
            "reward_model": {"ground_truth": "303", "style": "rule"},
        },
    ]
    return pd.DataFrame(records, index=[11, 14, 19])


def _runtime_prefix() -> Path:
    return (Path.cwd() / PINNED_RUNTIME_ENV).resolve()


class OpenpsiStackDryRunTest(unittest.TestCase):
    def test_rejects_non_pinned_dataset_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset_path must be pinned"):
            run_openpsi_stack_dry_run(
                dataset_path=Path("/tmp/not-pinned.parquet"),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                runtime_prefix=_runtime_prefix(),
            )

    def test_rejects_non_pinned_max_new_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_new_tokens must be pinned"):
            run_openpsi_baseline_augmented_report(
                output_dir=Path("/tmp/openpsi-artifacts"),
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                model_loader=lambda _path: _DummyModel(),
                generation_fn=_dummy_generation_fn,
                max_new_tokens=8,
                runtime_prefix=_runtime_prefix(),
            )

    def test_rejects_runtime_outside_repo_venv(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime environment must resolve under"):
            run_openpsi_stack_dry_run(
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                runtime_prefix=Path("/tmp/external-venv"),
            )

    def test_rejects_wrong_model_identifier(self) -> None:
        with mock.patch("openpsi_stack_dry_run._require_existing_path", return_value=None):
            with self.assertRaisesRegex(ValueError, "must include pinned identifier"):
                run_openpsi_stack_dry_run(
                    dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                    model_path=Path("/tmp/models/Llama-8B"),
                    tokenizer_loader=lambda _path: _DummyTokenizer(),
                    runtime_prefix=_runtime_prefix(),
                )

    def test_runs_dry_run_with_stub_tokenizer(self) -> None:
        with mock.patch("openpsi_stack_dry_run._require_existing_path", return_value=None), mock.patch(
            "openpsi_stack_dry_run.pd.read_parquet",
            return_value=_fixture_dataframe(),
        ):
            artifact = run_openpsi_stack_dry_run(
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                max_samples=2,
                robustness_difficulty="0/8",
                runtime_prefix=_runtime_prefix(),
            )

        self.assertEqual(artifact["stack"]["rl_framework"], "verl")
        self.assertEqual(artifact["stack"]["dataset"], "polaris")
        self.assertEqual(artifact["stack"]["base_model"], "Qwen3-1.7B")
        self.assertEqual(artifact["training_config"]["runtime_env"], PINNED_RUNTIME_ENV)
        self.assertEqual(
            artifact["training_config"]["validation_dataset_path"],
            PINNED_VALIDATION_DATASET_PATH,
        )
        self.assertEqual(artifact["training_config"]["n_rollout"], PINNED_N_ROLLOUT)
        self.assertEqual(
            artifact["training_config"]["max_output_tokens"],
            PINNED_MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(artifact["training_config"]["batch_size"], PINNED_BATCH_SIZE)
        self.assertEqual(artifact["training_config"]["lr"], PINNED_LR)
        self.assertEqual(artifact["source_rows"], 2)
        self.assertEqual(artifact["augmented_rows"], 2)

    def test_run_baseline_augmented_report_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "openpsi_stack_dry_run._require_existing_path", return_value=None
        ), mock.patch(
            "openpsi_stack_dry_run.pd.read_parquet",
            return_value=_fixture_dataframe(),
        ):
            paths = run_openpsi_baseline_augmented_report(
                output_dir=Path(tmpdir) / "artifacts",
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                model_loader=lambda _path: _DummyModel(),
                generation_fn=_dummy_generation_fn,
                max_sources=2,
                num_variants=1,
                baseline_seed=3,
                augmented_seed=5,
                robustness_difficulty="0/8",
                runtime_prefix=_runtime_prefix(),
            )

            self.assertTrue(paths["baseline"].exists())
            self.assertTrue(paths["augmented"].exists())
            self.assertTrue(paths["report"].exists())

            baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
            augmented = json.loads(paths["augmented"].read_text(encoding="utf-8"))
            report = json.loads(paths["report"].read_text(encoding="utf-8"))

            self.assertEqual(baseline["metrics"]["source_count"], 2)
            self.assertEqual(baseline["metrics"]["trajectory_count"], 2)
            self.assertEqual(augmented["metrics"]["trajectory_count"], 2)
            self.assertEqual(baseline["dataset_path"], PINNED_VALIDATION_DATASET_PATH)
            self.assertEqual(baseline["generation_config"]["max_new_tokens"], PINNED_MAX_OUTPUT_TOKENS)
            self.assertEqual(report["training_config"]["max_output_tokens"], PINNED_MAX_OUTPUT_TOKENS)
            self.assertTrue(report["consistency_checks"]["training_config_match"])
            self.assertTrue(report["consistency_checks"]["generation_config_match"])

    def test_multi_variant_report_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "openpsi_stack_dry_run._require_existing_path", return_value=None
        ), mock.patch(
            "openpsi_stack_dry_run.pd.read_parquet",
            return_value=_fixture_dataframe(),
        ):
            paths = run_openpsi_baseline_augmented_report(
                output_dir=Path(tmpdir) / "artifacts",
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                model_loader=lambda _path: _DummyModel(),
                generation_fn=_dummy_generation_fn,
                max_sources=2,
                num_variants=2,
                baseline_seed=1,
                augmented_seed=2,
                runtime_prefix=_runtime_prefix(),
            )

            baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
            augmented = json.loads(paths["augmented"].read_text(encoding="utf-8"))
            report = json.loads(paths["report"].read_text(encoding="utf-8"))

            self.assertEqual(baseline["metrics"]["trajectory_count"], 2)
            self.assertEqual(augmented["metrics"]["trajectory_count"], 4)
            self.assertEqual(report["run_context"]["num_variants"], 2)
            self.assertTrue(report["consistency_checks"]["trajectory_count_scaled_match"])

    def test_rejects_insufficient_eligible_sources(self) -> None:
        with mock.patch("openpsi_stack_dry_run._require_existing_path", return_value=None), mock.patch(
            "openpsi_stack_dry_run.pd.read_parquet",
            return_value=_fixture_dataframe(),
        ):
            with self.assertRaisesRegex(ValueError, "insufficient eligible rows"):
                run_openpsi_baseline_augmented_report(
                    output_dir=Path("/tmp/openpsi-artifacts"),
                    dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                    model_path=_PINNED_MODEL_PATH,
                    tokenizer_loader=lambda _path: _DummyTokenizer(),
                    model_loader=lambda _path: _DummyModel(),
                    generation_fn=_dummy_generation_fn,
                    max_sources=3,
                    num_variants=1,
                    runtime_prefix=_runtime_prefix(),
                )

    def test_report_rejects_source_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "openpsi_stack_dry_run._require_existing_path", return_value=None
        ), mock.patch(
            "openpsi_stack_dry_run.pd.read_parquet",
            return_value=_fixture_dataframe(),
        ):
            artifacts_dir = Path(tmpdir) / "artifacts"
            paths = run_openpsi_baseline_augmented_report(
                output_dir=artifacts_dir,
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                model_loader=lambda _path: _DummyModel(),
                generation_fn=_dummy_generation_fn,
                max_sources=2,
                num_variants=1,
                runtime_prefix=_runtime_prefix(),
            )

            augmented = json.loads(paths["augmented"].read_text(encoding="utf-8"))
            augmented["source_ids"] = ["999"]
            paths["augmented"].write_text(json.dumps(augmented, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source_id sets mismatch"):
                generate_openpsi_baseline_vs_augmented_report(
                    baseline_artifact_path=paths["baseline"],
                    augmented_artifact_path=paths["augmented"],
                    output_path=artifacts_dir / "report-regen.json",
                )

    def test_report_rejects_evaluator_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "openpsi_stack_dry_run._require_existing_path", return_value=None
        ), mock.patch(
            "openpsi_stack_dry_run.pd.read_parquet",
            return_value=_fixture_dataframe(),
        ):
            artifacts_dir = Path(tmpdir) / "artifacts"
            paths = run_openpsi_baseline_augmented_report(
                output_dir=artifacts_dir,
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                model_loader=lambda _path: _DummyModel(),
                generation_fn=_dummy_generation_fn,
                max_sources=2,
                num_variants=1,
                runtime_prefix=_runtime_prefix(),
            )

            augmented = json.loads(paths["augmented"].read_text(encoding="utf-8"))
            augmented["run_context"]["evaluator"] = "different-evaluator"
            paths["augmented"].write_text(json.dumps(augmented, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "run_context.evaluator"):
                generate_openpsi_baseline_vs_augmented_report(
                    baseline_artifact_path=paths["baseline"],
                    augmented_artifact_path=paths["augmented"],
                    output_path=artifacts_dir / "report-regen.json",
                )

    def test_report_rejects_training_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "openpsi_stack_dry_run._require_existing_path", return_value=None
        ), mock.patch(
            "openpsi_stack_dry_run.pd.read_parquet",
            return_value=_fixture_dataframe(),
        ):
            artifacts_dir = Path(tmpdir) / "artifacts"
            paths = run_openpsi_baseline_augmented_report(
                output_dir=artifacts_dir,
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                model_loader=lambda _path: _DummyModel(),
                generation_fn=_dummy_generation_fn,
                max_sources=2,
                num_variants=1,
                runtime_prefix=_runtime_prefix(),
            )

            augmented = json.loads(paths["augmented"].read_text(encoding="utf-8"))
            augmented["training_config"]["batch_size"] = 64
            paths["augmented"].write_text(json.dumps(augmented, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "training_config"):
                generate_openpsi_baseline_vs_augmented_report(
                    baseline_artifact_path=paths["baseline"],
                    augmented_artifact_path=paths["augmented"],
                    output_path=artifacts_dir / "report-regen.json",
                )

    def test_report_rejects_pinned_max_output_tokens_mismatch_even_when_equal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "openpsi_stack_dry_run._require_existing_path", return_value=None
        ), mock.patch(
            "openpsi_stack_dry_run.pd.read_parquet",
            return_value=_fixture_dataframe(),
        ):
            artifacts_dir = Path(tmpdir) / "artifacts"
            paths = run_openpsi_baseline_augmented_report(
                output_dir=artifacts_dir,
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                model_loader=lambda _path: _DummyModel(),
                generation_fn=_dummy_generation_fn,
                max_sources=2,
                num_variants=1,
                runtime_prefix=_runtime_prefix(),
            )

            for key in ("baseline", "augmented"):
                artifact = json.loads(paths[key].read_text(encoding="utf-8"))
                artifact["generation_config"]["max_new_tokens"] = 8
                paths[key].write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "generation_config.max_new_tokens"):
                generate_openpsi_baseline_vs_augmented_report(
                    baseline_artifact_path=paths["baseline"],
                    augmented_artifact_path=paths["augmented"],
                    output_path=artifacts_dir / "report-regen.json",
                )

    def test_report_rejects_scaled_ground_truth_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "openpsi_stack_dry_run._require_existing_path", return_value=None
        ), mock.patch(
            "openpsi_stack_dry_run.pd.read_parquet",
            return_value=_fixture_dataframe(),
        ):
            artifacts_dir = Path(tmpdir) / "artifacts"
            paths = run_openpsi_baseline_augmented_report(
                output_dir=artifacts_dir,
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                model_loader=lambda _path: _DummyModel(),
                generation_fn=_dummy_generation_fn,
                max_sources=2,
                num_variants=2,
                runtime_prefix=_runtime_prefix(),
            )

            augmented = json.loads(paths["augmented"].read_text(encoding="utf-8"))
            augmented["run_context"]["ground_truth_count"] = 3
            paths["augmented"].write_text(json.dumps(augmented, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "run_context.ground_truth_count"):
                generate_openpsi_baseline_vs_augmented_report(
                    baseline_artifact_path=paths["baseline"],
                    augmented_artifact_path=paths["augmented"],
                    output_path=artifacts_dir / "report-regen.json",
                )

    def test_runs_dry_run_with_real_schema_fixture_without_extra_info_or_difficulty(self) -> None:
        with mock.patch("openpsi_stack_dry_run._require_existing_path", return_value=None), mock.patch(
            "openpsi_stack_dry_run.pd.read_parquet",
            return_value=_fixture_dataframe_without_extra_info_or_difficulty(),
        ):
            artifact = run_openpsi_stack_dry_run(
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                max_samples=2,
                runtime_prefix=_runtime_prefix(),
            )

        self.assertEqual(artifact["source_rows"], 2)
        self.assertEqual(artifact["augmented_rows"], 2)
        self.assertEqual(artifact["robustness_split"]["field"], "dataset_partition")
        self.assertEqual(artifact["robustness_split"]["value"], "all-pinned-validation")

    def test_report_synthesizes_source_ids_for_real_schema_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "openpsi_stack_dry_run._require_existing_path", return_value=None
        ), mock.patch(
            "openpsi_stack_dry_run.pd.read_parquet",
            return_value=_fixture_dataframe_without_extra_info_or_difficulty(),
        ):
            paths = run_openpsi_baseline_augmented_report(
                output_dir=Path(tmpdir) / "artifacts",
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                model_loader=lambda _path: _DummyModel(),
                generation_fn=_dummy_generation_fn,
                max_sources=2,
                num_variants=1,
                runtime_prefix=_runtime_prefix(),
            )

            baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
            augmented = json.loads(paths["augmented"].read_text(encoding="utf-8"))
            report = json.loads(paths["report"].read_text(encoding="utf-8"))

            self.assertEqual(baseline["source_ids"], ["11", "14"])
            self.assertEqual(augmented["source_ids"], ["11", "14"])
            self.assertEqual(report["source_ids"], ["11", "14"])
            self.assertEqual(baseline["robustness_split"]["field"], "dataset_partition")
            self.assertEqual(
                baseline["robustness_split"]["selection"],
                "full-dataset-no-difficulty-column",
            )
            self.assertTrue(report["consistency_checks"]["robustness_split_match"])

    @unittest.skipUnless(
        Path(PINNED_VALIDATION_DATASET_PATH).exists() and _PINNED_MODEL_PATH.exists(),
        "requires local pinned parquet and model paths",
    )
    def test_real_pinned_dataset_smoke_with_stub_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = run_openpsi_baseline_augmented_report(
                output_dir=Path(tmpdir) / "artifacts",
                dataset_path=Path(PINNED_VALIDATION_DATASET_PATH),
                model_path=_PINNED_MODEL_PATH,
                tokenizer_loader=lambda _path: _DummyTokenizer(),
                model_loader=lambda _path: _DummyModel(),
                generation_fn=_dummy_generation_fn,
                max_sources=1,
                num_variants=1,
                runtime_prefix=_runtime_prefix(),
            )

            baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(baseline["dataset_path"], PINNED_VALIDATION_DATASET_PATH)
            self.assertGreaterEqual(baseline["metrics"]["source_count"], 1)
            self.assertIn("field", report["robustness_split"])
            self.assertIn("value", report["robustness_split"])


if __name__ == "__main__":
    unittest.main()
