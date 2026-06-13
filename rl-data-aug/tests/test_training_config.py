import tempfile
from pathlib import Path
import unittest

from training_config import (
    PINNED_BATCH_SIZE,
    PINNED_LR,
    PINNED_MAX_OUTPUT_TOKENS,
    PINNED_N_ROLLOUT,
    PINNED_RUNTIME_ENV,
    PINNED_VALIDATION_DATASET_PATH,
    TrainingStackConfig,
    apply_augmentation_settings,
    validate_current_runtime,
)


class TrainingStackConfigTest(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        cfg = TrainingStackConfig()
        cfg.validate()

    def test_rejects_non_verl_framework(self) -> None:
        cfg = TrainingStackConfig(rl_framework="other")
        with self.assertRaisesRegex(ValueError, "rl_framework must be 'verl'"):
            cfg.validate()

    def test_rejects_non_polaris_dataset(self) -> None:
        cfg = TrainingStackConfig(dataset="gsm8k")
        with self.assertRaisesRegex(ValueError, "dataset must be 'polaris'"):
            cfg.validate()

    def test_rejects_non_qwen3_1_7b_model(self) -> None:
        cfg = TrainingStackConfig(base_model="Qwen/Qwen3-8B")
        with self.assertRaisesRegex(ValueError, "base_model must be 'Qwen3-1.7B'"):
            cfg.validate()

    def test_rejects_non_repo_local_runtime_env(self) -> None:
        cfg = TrainingStackConfig(runtime_env="venv")
        with self.assertRaisesRegex(ValueError, "runtime_env must be '.venv'"):
            cfg.validate()

    def test_rejects_non_pinned_validation_dataset_path(self) -> None:
        cfg = TrainingStackConfig(validation_dataset_path="/tmp/other.parquet")
        with self.assertRaisesRegex(ValueError, "validation_dataset_path must be"):
            cfg.validate()

    def test_rejects_non_pinned_n_rollout(self) -> None:
        cfg = TrainingStackConfig(n_rollout=4)
        with self.assertRaisesRegex(ValueError, "n_rollout must be '8'"):
            cfg.validate()

    def test_rejects_non_pinned_max_output_tokens(self) -> None:
        cfg = TrainingStackConfig(max_output_tokens=1024)
        with self.assertRaisesRegex(ValueError, "max_output_tokens must be '40960'"):
            cfg.validate()

    def test_rejects_non_pinned_batch_size(self) -> None:
        cfg = TrainingStackConfig(batch_size=64)
        with self.assertRaisesRegex(ValueError, "batch_size must be '128'"):
            cfg.validate()

    def test_rejects_non_pinned_learning_rate(self) -> None:
        cfg = TrainingStackConfig(lr=1e-5)
        with self.assertRaisesRegex(ValueError, "lr must be '5e-06'"):
            cfg.validate()

    def test_rejects_non_positive_variant_count(self) -> None:
        cfg = TrainingStackConfig(num_variants=0)
        with self.assertRaisesRegex(ValueError, "num_variants must be positive"):
            cfg.validate()

    def test_rejects_enabled_augmentation_without_operators(self) -> None:
        cfg = TrainingStackConfig(augmentation_enabled=True, augmentation_operators=tuple())
        with self.assertRaisesRegex(
            ValueError, "augmentation_enabled requires at least one augmentation operator"
        ):
            cfg.validate()

    def test_validate_current_runtime_accepts_repo_local_prefix(self) -> None:
        cfg = TrainingStackConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            runtime_prefix = project_root / PINNED_RUNTIME_ENV
            runtime_prefix.mkdir(parents=True, exist_ok=True)

            out = validate_current_runtime(cfg, prefix=runtime_prefix, project_root=project_root)

            self.assertEqual(out, runtime_prefix.resolve())

    def test_validate_current_runtime_accepts_python_executable_under_repo_venv(self) -> None:
        cfg = TrainingStackConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            runtime_bin = project_root / PINNED_RUNTIME_ENV / "bin"
            runtime_bin.mkdir(parents=True, exist_ok=True)
            python_executable = runtime_bin / "python"
            python_executable.write_text("", encoding="utf-8")

            out = validate_current_runtime(
                cfg,
                python_executable=python_executable,
                project_root=project_root,
            )

            self.assertEqual(out, (project_root / PINNED_RUNTIME_ENV).resolve())

    def test_validate_current_runtime_rejects_external_prefix(self) -> None:
        cfg = TrainingStackConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            expected_runtime = project_root / PINNED_RUNTIME_ENV
            expected_runtime.mkdir(parents=True, exist_ok=True)
            external_runtime = project_root / "external-venv"
            external_runtime.mkdir(parents=True, exist_ok=True)

            with self.assertRaisesRegex(ValueError, "runtime environment must resolve under"):
                validate_current_runtime(
                    cfg,
                    prefix=external_runtime,
                    project_root=project_root,
                )

    def test_disabled_augmentation_preserves_payload(self) -> None:
        base_payload = {"source_id": "q-1", "prompt": "question", "meta": {"difficulty": "easy"}}
        cfg = TrainingStackConfig(augmentation_enabled=False)

        out = apply_augmentation_settings(base_payload, cfg)

        self.assertEqual(out, base_payload)
        self.assertNotIn("augmentation", out)
        self.assertNotIn("stack", out)
        self.assertNotIn("training_config", out)
        self.assertEqual(
            base_payload,
            {"source_id": "q-1", "prompt": "question", "meta": {"difficulty": "easy"}},
        )

    def test_enabled_augmentation_adds_expected_fields(self) -> None:
        base_payload = {"source_id": "q-1", "prompt": "question"}
        cfg = TrainingStackConfig(
            augmentation_enabled=True,
            augmentation_operators=("synonym_substitution",),
            num_variants=3,
        )

        out = apply_augmentation_settings(base_payload, cfg)

        self.assertEqual(out["augmentation"]["enabled"], True)
        self.assertEqual(out["augmentation"]["operators"], ["synonym_substitution"])
        self.assertEqual(out["augmentation"]["num_variants"], 3)
        self.assertEqual(out["stack"]["rl_framework"], "verl")
        self.assertEqual(out["stack"]["dataset"], "polaris")
        self.assertEqual(out["stack"]["base_model"], "Qwen3-1.7B")
        self.assertEqual(out["training_config"]["runtime_env"], PINNED_RUNTIME_ENV)
        self.assertEqual(
            out["training_config"]["validation_dataset_path"],
            PINNED_VALIDATION_DATASET_PATH,
        )
        self.assertEqual(out["training_config"]["n_rollout"], PINNED_N_ROLLOUT)
        self.assertEqual(
            out["training_config"]["max_output_tokens"],
            PINNED_MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(out["training_config"]["batch_size"], PINNED_BATCH_SIZE)
        self.assertEqual(out["training_config"]["lr"], PINNED_LR)


if __name__ == "__main__":
    unittest.main()
