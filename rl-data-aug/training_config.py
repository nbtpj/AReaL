"""Project-level training stack configuration and validation utilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys


ALLOWED_RL_FRAMEWORK = "verl"
ALLOWED_DATASET = "polaris"
ALLOWED_BASE_MODEL = "Qwen3-1.7B"
PINNED_RUNTIME_ENV = ".venv"
PINNED_VALIDATION_DATASET_PATH = "/storage/openpsi/users/zzy/sync/AIME24_converted_copy.parquet"
PINNED_N_ROLLOUT = 8
PINNED_MAX_OUTPUT_TOKENS = 40960
PINNED_BATCH_SIZE = 128
PINNED_LR = 5e-6
_LR_ABS_TOL = 1e-12


def _resolve_expected_runtime_prefix(
    *,
    runtime_env: str,
    project_root: str | Path | None = None,
) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    return (root / runtime_env).expanduser().resolve()


def _resolve_runtime_prefix(
    *,
    python_executable: str | Path | None = None,
    prefix: str | Path | None = None,
) -> Path:
    if prefix is not None:
        return Path(prefix).expanduser().resolve()

    if python_executable is not None:
        executable = Path(python_executable).expanduser().resolve()
        if executable.parent.name == "bin" and executable.name.startswith("python"):
            return executable.parent.parent
        return executable.parent

    return Path(sys.prefix).expanduser().resolve()


def _is_same_or_child(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class TrainingStackConfig:
    """Pinned stack configuration for the augmentation project."""

    rl_framework: str = ALLOWED_RL_FRAMEWORK
    dataset: str = ALLOWED_DATASET
    base_model: str = ALLOWED_BASE_MODEL
    runtime_env: str = PINNED_RUNTIME_ENV
    validation_dataset_path: str = PINNED_VALIDATION_DATASET_PATH
    n_rollout: int = PINNED_N_ROLLOUT
    max_output_tokens: int = PINNED_MAX_OUTPUT_TOKENS
    batch_size: int = PINNED_BATCH_SIZE
    lr: float = PINNED_LR
    augmentation_enabled: bool = False
    augmentation_operators: tuple[str, ...] = field(default_factory=tuple)
    num_variants: int = 1

    def validate(self) -> None:
        if self.rl_framework != ALLOWED_RL_FRAMEWORK:
            raise ValueError(
                f"rl_framework must be '{ALLOWED_RL_FRAMEWORK}', got '{self.rl_framework}'"
            )
        if self.dataset != ALLOWED_DATASET:
            raise ValueError(f"dataset must be '{ALLOWED_DATASET}', got '{self.dataset}'")
        if self.base_model != ALLOWED_BASE_MODEL:
            raise ValueError(
                f"base_model must be '{ALLOWED_BASE_MODEL}', got '{self.base_model}'"
            )
        if self.runtime_env != PINNED_RUNTIME_ENV:
            raise ValueError(
                f"runtime_env must be '{PINNED_RUNTIME_ENV}', got '{self.runtime_env}'"
            )
        if self.validation_dataset_path != PINNED_VALIDATION_DATASET_PATH:
            raise ValueError(
                "validation_dataset_path must be "
                f"'{PINNED_VALIDATION_DATASET_PATH}', got '{self.validation_dataset_path}'"
            )
        if self.n_rollout != PINNED_N_ROLLOUT:
            raise ValueError(
                f"n_rollout must be '{PINNED_N_ROLLOUT}', got '{self.n_rollout}'"
            )
        if self.max_output_tokens != PINNED_MAX_OUTPUT_TOKENS:
            raise ValueError(
                "max_output_tokens must be "
                f"'{PINNED_MAX_OUTPUT_TOKENS}', got '{self.max_output_tokens}'"
            )
        if self.batch_size != PINNED_BATCH_SIZE:
            raise ValueError(
                f"batch_size must be '{PINNED_BATCH_SIZE}', got '{self.batch_size}'"
            )
        if not math.isclose(float(self.lr), PINNED_LR, rel_tol=0.0, abs_tol=_LR_ABS_TOL):
            raise ValueError(f"lr must be '{PINNED_LR}', got '{self.lr}'")
        if self.num_variants <= 0:
            raise ValueError(f"num_variants must be positive, got '{self.num_variants}'")
        if self.augmentation_enabled and not self.augmentation_operators:
            raise ValueError("augmentation_enabled requires at least one augmentation operator")


def validate_current_runtime(
    config: TrainingStackConfig,
    *,
    python_executable: str | Path | None = None,
    prefix: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Ensure runtime resolves to the pinned repository-local virtual environment."""

    config.validate()
    expected_prefix = _resolve_expected_runtime_prefix(
        runtime_env=config.runtime_env,
        project_root=project_root,
    )
    runtime_prefix = _resolve_runtime_prefix(
        python_executable=python_executable,
        prefix=prefix,
    )
    if not _is_same_or_child(runtime_prefix, expected_prefix):
        raise ValueError(
            "runtime environment must resolve under "
            f"'{expected_prefix}', got '{runtime_prefix}'"
        )
    return runtime_prefix


def pinned_training_metadata(config: TrainingStackConfig | None = None) -> dict[str, str | int | float]:
    """Return canonical pinned training metadata after validation."""

    cfg = config or TrainingStackConfig()
    cfg.validate()
    return {
        "runtime_env": cfg.runtime_env,
        "validation_dataset_path": cfg.validation_dataset_path,
        "n_rollout": cfg.n_rollout,
        "max_output_tokens": cfg.max_output_tokens,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
    }


def apply_augmentation_settings(base_payload: dict, config: TrainingStackConfig) -> dict:
    """Return a payload with optional augmentation settings while preserving baseline parity."""

    config.validate()
    payload = deepcopy(base_payload)
    if not config.augmentation_enabled:
        return payload

    payload["augmentation"] = {
        "enabled": True,
        "operators": list(config.augmentation_operators),
        "num_variants": config.num_variants,
    }
    payload["stack"] = {
        "rl_framework": config.rl_framework,
        "dataset": config.dataset,
        "base_model": config.base_model,
    }
    payload["training_config"] = pinned_training_metadata(config)
    return payload
