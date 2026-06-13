"""Baseline-vs-augmented robustness evaluation and reporting utilities."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable

from polaris_loader import build_grpo_trajectories_from_polaris
from training_config import TrainingStackConfig, pinned_training_metadata, validate_current_runtime


EVALUATOR_NAME = "prompt-heuristic-v1"
ROBUSTNESS_SPLIT_FIELD = "difficulty"
REQUIRED_METRICS = {
    "overall_proxy_quality",
    "overall_avg_prompt_words",
    "robustness_subset_proxy_quality",
    "robustness_subset_count",
}
EXPECTED_STACK = {
    "rl_framework": TrainingStackConfig().rl_framework,
    "dataset": TrainingStackConfig().dataset,
    "base_model": TrainingStackConfig().base_model,
    **pinned_training_metadata(TrainingStackConfig()),
}
STACK_KEYS = set(EXPECTED_STACK.keys())
RUN_CONTEXT_KEYS = {"evaluator", "robustness_split", "seed"}


def _stable_prompt_score(prompt: str) -> float:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _avg_prompt_words(trajectories: list[dict]) -> float:
    if not trajectories:
        return 0.0
    total_words = 0
    for item in trajectories:
        total_words += len(str(item.get("prompt", "")).split())
    return total_words / len(trajectories)


def evaluate_trajectories(trajectories: list[dict], robustness_split_value: str) -> dict[str, float]:
    """Compute deterministic heuristic metrics for one trajectory set."""

    if not trajectories:
        raise ValueError("cannot evaluate an empty trajectory set")

    overall_scores = [_stable_prompt_score(str(item["prompt"])) for item in trajectories]
    split_subset = [
        item
        for item in trajectories
        if str(item.get(ROBUSTNESS_SPLIT_FIELD, "")).lower() == robustness_split_value.lower()
    ]
    split_scores = [_stable_prompt_score(str(item["prompt"])) for item in split_subset]

    return {
        "overall_proxy_quality": sum(overall_scores) / len(overall_scores),
        "overall_avg_prompt_words": _avg_prompt_words(trajectories),
        "robustness_subset_proxy_quality": (
            sum(split_scores) / len(split_scores) if split_scores else 0.0
        ),
        "robustness_subset_count": float(len(split_subset)),
    }


def _stack_echo(config: TrainingStackConfig) -> dict[str, str | int | float]:
    return {
        "rl_framework": config.rl_framework,
        "dataset": config.dataset,
        "base_model": config.base_model,
        **pinned_training_metadata(config),
    }


def run_polaris_experiment(
    raw_records: Iterable[dict],
    config: TrainingStackConfig,
    seed: int,
    robustness_split_value: str = "easy",
    protected_tokens_by_source: dict[str, tuple[str, ...]] | None = None,
    python_executable: str | Path | None = None,
    runtime_prefix: str | Path | None = None,
) -> dict:
    """Run one deterministic experiment and return a structured artifact dictionary."""

    config.validate()
    validate_current_runtime(
        config,
        python_executable=python_executable,
        prefix=runtime_prefix,
        project_root=Path(__file__).resolve().parent,
    )
    records = list(raw_records)
    trajectories = build_grpo_trajectories_from_polaris(
        raw_records=records,
        config=config,
        seed=seed,
        protected_tokens_by_source=protected_tokens_by_source,
    )
    metrics = evaluate_trajectories(trajectories=trajectories, robustness_split_value=robustness_split_value)

    return {
        "mode": "augmented" if config.augmentation_enabled else "baseline",
        "stack": _stack_echo(config),
        "run_context": {
            "evaluator": EVALUATOR_NAME,
            "robustness_split": {
                "field": ROBUSTNESS_SPLIT_FIELD,
                "value": robustness_split_value,
            },
            "seed": seed,
        },
        "counts": {
            "source_records": len(records),
            "trajectories": len(trajectories),
        },
        "metrics": metrics,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def write_artifact(artifact: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_artifact(path: str | Path) -> dict:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ValueError(f"artifact not found: '{artifact_path}'")
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def _validate_artifact_shape(artifact: dict, expected_mode: str) -> None:
    if artifact.get("mode") != expected_mode:
        raise ValueError(f"expected mode '{expected_mode}', got '{artifact.get('mode')}'")

    if "metrics" not in artifact or not isinstance(artifact["metrics"], dict):
        raise ValueError("artifact is missing metrics dictionary")
    missing_metrics = REQUIRED_METRICS - set(artifact["metrics"].keys())
    if missing_metrics:
        raise ValueError(f"artifact metrics missing required keys: {sorted(missing_metrics)}")

    if "stack" not in artifact or not isinstance(artifact["stack"], dict):
        raise ValueError("artifact is missing stack dictionary")
    missing_stack = STACK_KEYS - set(artifact["stack"].keys())
    if missing_stack:
        raise ValueError(f"artifact stack missing required keys: {sorted(missing_stack)}")
    if artifact["stack"] != EXPECTED_STACK:
        raise ValueError("artifact stack metadata must match pinned stack values")

    if "run_context" not in artifact or not isinstance(artifact["run_context"], dict):
        raise ValueError("artifact is missing run_context dictionary")
    missing_context = RUN_CONTEXT_KEYS - set(artifact["run_context"].keys())
    if missing_context:
        raise ValueError(f"artifact run_context missing required keys: {sorted(missing_context)}")


def _validate_pair_consistency(baseline: dict, augmented: dict) -> None:
    if baseline["stack"] != augmented["stack"]:
        raise ValueError("baseline and augmented stack metadata mismatch")

    for context_key in ("evaluator", "robustness_split"):
        if baseline["run_context"][context_key] != augmented["run_context"][context_key]:
            raise ValueError(
                f"baseline and augmented run_context mismatch for '{context_key}'"
            )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _compute_deltas(baseline_metrics: dict, augmented_metrics: dict) -> dict[str, dict[str, float | None]]:
    deltas: dict[str, dict[str, float | None]] = {}
    for key in sorted(REQUIRED_METRICS):
        baseline_value = float(baseline_metrics[key])
        augmented_value = float(augmented_metrics[key])
        absolute_delta = augmented_value - baseline_value
        relative_delta = None if baseline_value == 0 else absolute_delta / baseline_value
        deltas[key] = {
            "baseline": baseline_value,
            "augmented": augmented_value,
            "absolute_delta": absolute_delta,
            "relative_delta": relative_delta,
        }
    return deltas


def generate_baseline_anchored_report(
    baseline_artifact_path: str | Path,
    augmented_artifact_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Generate a validated report comparing augmented run metrics against baseline."""

    baseline_path = Path(baseline_artifact_path)
    augmented_path = Path(augmented_artifact_path)
    baseline = load_artifact(baseline_path)
    augmented = load_artifact(augmented_path)

    _validate_artifact_shape(baseline, expected_mode="baseline")
    _validate_artifact_shape(augmented, expected_mode="augmented")
    _validate_pair_consistency(baseline, augmented)

    report = {
        "baseline_artifact": str(baseline_path),
        "augmented_artifact": str(augmented_path),
        "stack": deepcopy(baseline["stack"]),
        "run_context": {
            "evaluator": baseline["run_context"]["evaluator"],
            "robustness_split": deepcopy(baseline["run_context"]["robustness_split"]),
            "baseline_seed": baseline["run_context"]["seed"],
            "augmented_seed": augmented["run_context"]["seed"],
        },
        "deltas": _compute_deltas(baseline["metrics"], augmented["metrics"]),
        "consistency_checks": {
            "stack_match": True,
            "evaluator_match": True,
            "robustness_split_match": True,
        },
        "reproducibility": {
            "baseline_sha256": _hash_file(baseline_path),
            "augmented_sha256": _hash_file(augmented_path),
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }
    return write_artifact(report, output_path)


def run_baseline_and_augmented_report(
    raw_records: Iterable[dict],
    output_dir: str | Path,
    *,
    baseline_seed: int,
    augmented_seed: int,
    robustness_split_value: str = "easy",
    num_variants: int = 2,
    protected_tokens_by_source: dict[str, tuple[str, ...]] | None = None,
    python_executable: str | Path | None = None,
    runtime_prefix: str | Path | None = None,
) -> dict[str, Path]:
    """Run baseline+augmented experiments and emit a consolidated report artifact."""

    baseline_config = TrainingStackConfig(augmentation_enabled=False)
    augmented_config = TrainingStackConfig(
        augmentation_enabled=True,
        augmentation_operators=("synonym_substitution",),
        num_variants=num_variants,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_artifact = run_polaris_experiment(
        raw_records=raw_records,
        config=baseline_config,
        seed=baseline_seed,
        robustness_split_value=robustness_split_value,
        protected_tokens_by_source=protected_tokens_by_source,
        python_executable=python_executable,
        runtime_prefix=runtime_prefix,
    )
    augmented_artifact = run_polaris_experiment(
        raw_records=raw_records,
        config=augmented_config,
        seed=augmented_seed,
        robustness_split_value=robustness_split_value,
        protected_tokens_by_source=protected_tokens_by_source,
        python_executable=python_executable,
        runtime_prefix=runtime_prefix,
    )

    baseline_path = write_artifact(baseline_artifact, out_dir / "baseline-run.json")
    augmented_path = write_artifact(augmented_artifact, out_dir / "augmented-run.json")
    report_path = generate_baseline_anchored_report(
        baseline_artifact_path=baseline_path,
        augmented_artifact_path=augmented_path,
        output_path=out_dir / "baseline-vs-augmented-report.json",
    )
    return {
        "baseline": baseline_path,
        "augmented": augmented_path,
        "report": report_path,
    }
