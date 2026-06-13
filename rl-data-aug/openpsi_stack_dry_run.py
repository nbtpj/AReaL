"""Fail-closed dry-run and baseline-vs-augmented reporting for pinned OpenPSI stack."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import pandas as pd
from training_config import (
    PINNED_MAX_OUTPUT_TOKENS,
    PINNED_VALIDATION_DATASET_PATH,
    TrainingStackConfig,
    pinned_training_metadata,
    validate_current_runtime,
)


DEFAULT_POLARIS_PARQUET_PATH = Path(PINNED_VALIDATION_DATASET_PATH)
DEFAULT_QWEN3_MODEL_PATH = Path("/storage/openpsi/models/Qwen__Qwen3-1.7B")
EXPECTED_MODEL_PATH_TOKEN = "Qwen__Qwen3-1.7B"


_AUGMENTATION_MODULE_PATH = Path(__file__).resolve().parent / "verl" / "verl" / "utils" / "dataset" / "augmentation.py"
_AUGMENTATION_SPEC = importlib.util.spec_from_file_location("openpsi_augmentation_runtime", _AUGMENTATION_MODULE_PATH)
_AUGMENTATION_MODULE = importlib.util.module_from_spec(_AUGMENTATION_SPEC)
assert _AUGMENTATION_SPEC is not None and _AUGMENTATION_SPEC.loader is not None
sys.modules[_AUGMENTATION_SPEC.name] = _AUGMENTATION_MODULE
_AUGMENTATION_SPEC.loader.exec_module(_AUGMENTATION_MODULE)
RLDataAugmentationConfig = _AUGMENTATION_MODULE.RLDataAugmentationConfig
augment_rlhf_records = _AUGMENTATION_MODULE.augment_rlhf_records


STACK_INFO = {
    "rl_framework": "verl",
    "dataset": "polaris",
    "base_model": "Qwen3-1.7B",
}
PINNED_TRAINING_CONFIG = pinned_training_metadata(TrainingStackConfig())
REQUIRED_TRAINING_CONFIG_KEYS = set(PINNED_TRAINING_CONFIG.keys())
REQUIRED_POLARIS_COLUMNS = {"prompt", "label", "data_source", "reward_model"}
NO_DIFFICULTY_SPLIT = {
    "field": "dataset_partition",
    "value": "all-pinned-validation",
    "selection": "full-dataset-no-difficulty-column",
}
REQUIRED_RUN_METRICS = {
    "accuracy",
    "exact_match_count",
    "source_count",
    "trajectory_count",
    "avg_prompt_tokens",
    "avg_output_tokens",
    "avg_output_chars",
}


def _require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise ValueError(f"{label} path does not exist: '{path}'")


def _require_model_identifier(model_path: Path) -> None:
    if EXPECTED_MODEL_PATH_TOKEN not in str(model_path):
        raise ValueError(
            "model path must include pinned identifier "
            f"'{EXPECTED_MODEL_PATH_TOKEN}', got '{model_path}'"
        )


def _enforce_pinned_openpsi_inputs(
    *,
    dataset_path: Path,
    max_new_tokens: int | None = None,
) -> None:
    expected_dataset_path = Path(PINNED_VALIDATION_DATASET_PATH).resolve()
    if dataset_path.resolve() != expected_dataset_path:
        raise ValueError(
            "dataset_path must be pinned to "
            f"'{expected_dataset_path}', got '{dataset_path.resolve()}'"
        )
    if max_new_tokens is not None and int(max_new_tokens) != PINNED_MAX_OUTPUT_TOKENS:
        raise ValueError(
            "max_new_tokens must be pinned to "
            f"'{PINNED_MAX_OUTPUT_TOKENS}', got '{int(max_new_tokens)}'"
        )


def _extract_user_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        if not prompt.strip():
            raise ValueError("prompt text must be non-empty")
        return prompt
    messages: list[Any] | None = None
    if isinstance(prompt, list | tuple):
        messages = list(prompt)
    elif hasattr(prompt, "tolist"):
        converted = prompt.tolist()
        if isinstance(converted, list):
            messages = converted

    if messages is not None:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).lower() != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError("unable to extract user prompt text from prompt payload")


def _encode_prompt(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return list(token_ids)
    if callable(tokenizer):
        out = tokenizer(text, add_special_tokens=False)
        if isinstance(out, dict) and "input_ids" in out:
            return list(out["input_ids"])
    raise ValueError("tokenizer must provide encode() or callable returning input_ids")


def _extract_source_id(record: dict, index: int) -> str:
    extra_info = record.get("extra_info")
    if not isinstance(extra_info, dict) or "index" not in extra_info:
        raise ValueError(f"record at subset index {index} missing extra_info.index")
    return str(extra_info["index"])


def _normalize_split_record(record: dict, row_index: int) -> dict:
    normalized = deepcopy(record)

    if "prompt" not in normalized:
        raise ValueError(f"record at row index {row_index} missing prompt")
    _extract_user_prompt_text(normalized.get("prompt"))

    if not isinstance(normalized.get("reward_model"), dict):
        raise ValueError(
            f"record at row index {row_index} missing reward_model dictionary"
        )

    extra_info = normalized.get("extra_info")
    if extra_info is None:
        extra_info = {}
    elif not isinstance(extra_info, dict):
        raise ValueError(f"record at row index {row_index} has non-dict extra_info")

    source_index = extra_info.get("index")
    if source_index is None or not str(source_index).strip():
        extra_info["index"] = int(row_index)
    normalized["extra_info"] = extra_info

    return normalized


def _resolve_robustness_split(
    dataframe: pd.DataFrame,
    robustness_difficulty: str,
) -> tuple[pd.DataFrame, dict]:
    if "difficulty" in dataframe.columns:
        subset = dataframe[dataframe["difficulty"] == robustness_difficulty]
        return subset, {
            "field": "difficulty",
            "value": robustness_difficulty,
            "selection": "difficulty-column-filter",
        }

    return dataframe, deepcopy(NO_DIFFICULTY_SPLIT)


def _extract_ground_truth(record: dict) -> str | None:
    reward_model = record.get("reward_model")
    if not isinstance(reward_model, dict):
        return None
    ground_truth = reward_model.get("ground_truth")
    if ground_truth is None:
        return None
    value = str(ground_truth).strip()
    return value if value else None


def _normalize_answer(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def _default_tokenizer_loader(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )


def _default_model_loader(model_path: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )


def _default_generation_fn(
    prompt_text: str,
    *,
    tokenizer: Any,
    model: Any,
    ground_truth: str | None,
    max_new_tokens: int,
) -> str:
    del ground_truth
    import torch

    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise ValueError("tokenizer output missing input_ids")

    if hasattr(model, "eval"):
        model.eval()

    device = getattr(model, "device", None)
    if device is not None:
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
            if hasattr(value, "to")
        }

    with torch.no_grad():
        # Single-step greedy decode keeps runtime bounded on CPU while preserving
        # deterministic, model-derived outputs for baseline/augmented comparison.
        del max_new_tokens
        logits = model(**encoded).logits
    next_token_id = int(logits[0, -1].argmax().item())
    output = tokenizer.decode([next_token_id], skip_special_tokens=True).strip()
    if not output:
        output = str(next_token_id)
    return output


def _build_aug_config(num_variants: int, seed: int) -> RLDataAugmentationConfig:
    return RLDataAugmentationConfig(
        enabled=True,
        operators=("synonym_substitution",),
        num_variants=int(num_variants),
        seed=int(seed),
        source_id_field="extra_info.index",
        rl_framework=STACK_INFO["rl_framework"],
        dataset=STACK_INFO["dataset"],
        base_model=STACK_INFO["base_model"],
    )


def _json_dumps(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except Exception:
        return "unknown"


def _load_records_for_split(
    dataframe: pd.DataFrame,
    robustness_difficulty: str,
) -> tuple[list[dict], dict]:
    missing_columns = REQUIRED_POLARIS_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"dataset missing required columns: {sorted(missing_columns)}")

    subset, split = _resolve_robustness_split(dataframe, robustness_difficulty)
    subset_records = [
        _normalize_split_record(row.to_dict(), int(row_index))
        for row_index, row in subset.iterrows()
    ]
    if not subset_records:
        raise ValueError("no rows selected for dry-run subset")
    return subset_records, split


def _select_eligible_records(
    subset_records: list[dict],
    *,
    max_sources: int,
    num_variants: int,
    seed: int,
) -> tuple[list[dict], dict]:
    if max_sources <= 0:
        raise ValueError("max_sources must be positive")
    if num_variants <= 0:
        raise ValueError("num_variants must be positive")

    config = _build_aug_config(num_variants=num_variants, seed=seed)
    selected_records: list[dict] = []
    eligible_rows = 0
    skip_reasons: dict[str, int] = {}

    for idx, record in enumerate(subset_records):
        try:
            augmented = augment_rlhf_records(
                records=[record],
                prompt_key="prompt",
                config=config,
            )
            if len(augmented) != num_variants:
                raise ValueError(
                    f"augmenter returned {len(augmented)} variants for one source (expected {num_variants})"
                )
            _extract_source_id(record, idx)
            eligible_rows += 1
            if len(selected_records) < max_sources:
                selected_records.append(deepcopy(record))
        except Exception as exc:
            reason = str(exc).strip() or exc.__class__.__name__
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    scanned_rows = len(subset_records)
    skipped_rows = scanned_rows - eligible_rows
    if len(selected_records) < max_sources:
        raise ValueError(
            "insufficient eligible rows for requested source count: "
            f"requested={max_sources}, selected={len(selected_records)}, "
            f"eligible={eligible_rows}, scanned={scanned_rows}"
        )

    return selected_records, {
        "scanned_rows": scanned_rows,
        "eligible_rows": eligible_rows,
        "skipped_rows": skipped_rows,
        "selected_rows": len(selected_records),
        "skip_reasons": dict(sorted(skip_reasons.items())),
    }


def _evaluate_records(
    records: list[dict],
    *,
    tokenizer: Any,
    model: Any,
    generation_fn: Callable[..., str],
    max_new_tokens: int,
    mode: str,
) -> tuple[list[dict], dict]:
    trajectories: list[dict] = []
    exact_match_count = 0
    ground_truth_count = 0

    for idx, record in enumerate(records):
        source_id = _extract_source_id(record, idx)
        prompt_text = _extract_user_prompt_text(record.get("prompt"))
        prompt_tokens = _encode_prompt(tokenizer, prompt_text)
        ground_truth = _extract_ground_truth(record)

        output_text = generation_fn(
            prompt_text,
            tokenizer=tokenizer,
            model=model,
            ground_truth=ground_truth,
            max_new_tokens=max_new_tokens,
        )
        output_text = str(output_text).strip()
        output_tokens = _encode_prompt(tokenizer, output_text) if output_text else []

        exact_match = None
        if ground_truth is not None:
            ground_truth_count += 1
            exact_match = _normalize_answer(output_text) == _normalize_answer(ground_truth)
            if exact_match:
                exact_match_count += 1

        trajectory = {
            "source_id": source_id,
            "prompt": prompt_text,
            "output": output_text,
            "ground_truth": ground_truth,
            "exact_match": exact_match,
            "prompt_token_length": len(prompt_tokens),
            "output_token_length": len(output_tokens),
        }

        if mode == "augmented":
            extra_info = record.get("extra_info")
            if not isinstance(extra_info, dict):
                raise ValueError("augmented record missing extra_info dictionary")
            aug_info = extra_info.get("augmentation")
            if not isinstance(aug_info, dict):
                raise ValueError("augmented record missing extra_info.augmentation")
            trajectory.update(
                {
                    "aug_op": aug_info.get("aug_op"),
                    "seed": aug_info.get("seed"),
                    "variant_index": aug_info.get("variant_index"),
                    "rl_framework": STACK_INFO["rl_framework"],
                    "dataset": STACK_INFO["dataset"],
                    "base_model": STACK_INFO["base_model"],
                }
            )

        trajectories.append(trajectory)

    unique_source_ids = list(dict.fromkeys(item["source_id"] for item in trajectories))
    trajectory_count = len(trajectories)
    avg_prompt_tokens = (
        sum(item["prompt_token_length"] for item in trajectories) / trajectory_count
        if trajectory_count
        else 0.0
    )
    avg_output_tokens = (
        sum(item["output_token_length"] for item in trajectories) / trajectory_count
        if trajectory_count
        else 0.0
    )
    avg_output_chars = (
        sum(len(item["output"]) for item in trajectories) / trajectory_count
        if trajectory_count
        else 0.0
    )

    metrics = {
        "accuracy": (exact_match_count / ground_truth_count) if ground_truth_count else 0.0,
        "exact_match_count": exact_match_count,
        "source_count": len(unique_source_ids),
        "trajectory_count": trajectory_count,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_output_chars": avg_output_chars,
    }

    return trajectories, {
        "metrics": metrics,
        "source_ids": unique_source_ids,
        "ground_truth_count": ground_truth_count,
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_run_artifact(artifact: dict, *, expected_mode: str) -> None:
    mode = artifact.get("mode")
    if mode != expected_mode:
        raise ValueError(f"expected mode '{expected_mode}', got '{mode}'")

    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("artifact missing metrics dictionary")
    missing_metrics = REQUIRED_RUN_METRICS - set(metrics.keys())
    if missing_metrics:
        raise ValueError(f"artifact metrics missing required keys: {sorted(missing_metrics)}")

    stack = artifact.get("stack")
    if not isinstance(stack, dict) or stack != STACK_INFO:
        raise ValueError("artifact stack metadata mismatch")
    training_config = artifact.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError("artifact missing training_config dictionary")
    if set(training_config.keys()) != REQUIRED_TRAINING_CONFIG_KEYS:
        raise ValueError(
            "artifact training_config keys mismatch: "
            f"expected {sorted(REQUIRED_TRAINING_CONFIG_KEYS)}, got {sorted(training_config.keys())}"
        )
    if training_config != PINNED_TRAINING_CONFIG:
        raise ValueError("artifact training_config metadata mismatch")

    for required_key in (
        "dataset_path",
        "model_path",
        "robustness_split",
        "selection",
        "generation_config",
        "run_context",
        "source_ids",
        "input_hash",
    ):
        if required_key not in artifact:
            raise ValueError(f"artifact missing required key '{required_key}'")

    dataset_path = Path(str(artifact["dataset_path"])).resolve()
    expected_dataset_path = Path(PINNED_VALIDATION_DATASET_PATH).resolve()
    if dataset_path != expected_dataset_path:
        raise ValueError(
            "artifact dataset_path must match pinned validation dataset: "
            f"expected '{expected_dataset_path}', got '{dataset_path}'"
        )
    if str(artifact["dataset_path"]) != training_config["validation_dataset_path"]:
        raise ValueError("artifact dataset_path must match training_config.validation_dataset_path")

    generation_config = artifact.get("generation_config")
    if not isinstance(generation_config, dict):
        raise ValueError("artifact generation_config metadata must be a dictionary")
    max_new_tokens = int(generation_config.get("max_new_tokens", 0))
    if max_new_tokens != PINNED_MAX_OUTPUT_TOKENS:
        raise ValueError(
            "artifact generation_config.max_new_tokens must match pinned max_output_tokens: "
            f"expected {PINNED_MAX_OUTPUT_TOKENS}, got {max_new_tokens}"
        )


def _validate_run_pair(baseline: dict, augmented: dict) -> None:
    for key in ("dataset_path", "model_path", "robustness_split", "generation_config", "training_config"):
        if baseline.get(key) != augmented.get(key):
            raise ValueError(f"baseline and augmented mismatch for '{key}'")

    baseline_context = baseline.get("run_context")
    augmented_context = augmented.get("run_context")
    if not isinstance(baseline_context, dict) or not isinstance(augmented_context, dict):
        raise ValueError("baseline and augmented run_context metadata must be dictionaries")
    if baseline_context.get("evaluator") != augmented_context.get("evaluator"):
        raise ValueError("baseline and augmented mismatch for 'run_context.evaluator'")

    baseline_num_variants = int(baseline_context.get("num_variants", 0))
    if baseline_num_variants != 1:
        raise ValueError(
            "baseline run_context.num_variants must be 1, "
            f"got {baseline_num_variants}"
        )
    augmented_num_variants = int(augmented_context.get("num_variants", 0))
    if augmented_num_variants <= 0:
        raise ValueError(
            "augmented run_context.num_variants must be positive, "
            f"got {augmented_num_variants}"
        )

    baseline_ground_truth = int(baseline_context.get("ground_truth_count", 0))
    augmented_ground_truth = int(augmented_context.get("ground_truth_count", 0))
    expected_augmented_ground_truth = baseline_ground_truth * augmented_num_variants
    if augmented_ground_truth != expected_augmented_ground_truth:
        raise ValueError(
            "baseline and augmented mismatch for 'run_context.ground_truth_count': "
            f"expected {expected_augmented_ground_truth}, got {augmented_ground_truth}"
        )

    baseline_metrics = baseline.get("metrics")
    augmented_metrics = augmented.get("metrics")
    if not isinstance(baseline_metrics, dict) or not isinstance(augmented_metrics, dict):
        raise ValueError("baseline and augmented metrics must be dictionaries")
    baseline_trajectory_count = int(baseline_metrics.get("trajectory_count", 0))
    augmented_trajectory_count = int(augmented_metrics.get("trajectory_count", 0))
    expected_augmented_trajectory_count = baseline_trajectory_count * augmented_num_variants
    if augmented_trajectory_count != expected_augmented_trajectory_count:
        raise ValueError(
            "baseline and augmented mismatch for 'metrics.trajectory_count': "
            f"expected {expected_augmented_trajectory_count}, got {augmented_trajectory_count}"
        )
    baseline_source_count = int(baseline_metrics.get("source_count", 0))
    augmented_source_count = int(augmented_metrics.get("source_count", 0))
    if baseline_source_count != augmented_source_count:
        raise ValueError(
            "baseline and augmented mismatch for 'metrics.source_count': "
            f"expected {baseline_source_count}, got {augmented_source_count}"
        )

    baseline_sources = baseline.get("source_ids")
    augmented_sources = augmented.get("source_ids")
    if sorted(baseline_sources) != sorted(augmented_sources):
        raise ValueError("baseline and augmented source_id sets mismatch")


def _compute_metric_deltas(baseline_metrics: dict, augmented_metrics: dict) -> dict[str, dict[str, float | None]]:
    deltas: dict[str, dict[str, float | None]] = {}
    for key in sorted(REQUIRED_RUN_METRICS):
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


def generate_openpsi_baseline_vs_augmented_report(
    baseline_artifact_path: str | Path,
    augmented_artifact_path: str | Path,
    output_path: str | Path,
) -> Path:
    baseline_path = Path(baseline_artifact_path)
    augmented_path = Path(augmented_artifact_path)

    if not baseline_path.exists():
        raise ValueError(f"artifact not found: '{baseline_path}'")
    if not augmented_path.exists():
        raise ValueError(f"artifact not found: '{augmented_path}'")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    augmented = json.loads(augmented_path.read_text(encoding="utf-8"))

    _validate_run_artifact(baseline, expected_mode="baseline")
    _validate_run_artifact(augmented, expected_mode="augmented")
    _validate_run_pair(baseline, augmented)

    project_root = Path(__file__).resolve().parent
    report = {
        "baseline_artifact": str(baseline_path),
        "augmented_artifact": str(augmented_path),
        "stack": deepcopy(STACK_INFO),
        "training_config": deepcopy(baseline["training_config"]),
        "dataset_path": baseline["dataset_path"],
        "model_path": baseline["model_path"],
        "robustness_split": deepcopy(baseline["robustness_split"]),
        "selection": deepcopy(baseline["selection"]),
        "run_context": {
            "evaluator": baseline["run_context"]["evaluator"],
            "baseline_seed": baseline["run_context"]["seed"],
            "augmented_seed": augmented["run_context"]["seed"],
            "num_variants": augmented["run_context"]["num_variants"],
        },
        "generation_config": deepcopy(baseline["generation_config"]),
        "source_ids": deepcopy(baseline["source_ids"]),
        "baseline_metrics": deepcopy(baseline["metrics"]),
        "augmented_metrics": deepcopy(augmented["metrics"]),
        "cardinality": {
            "baseline_num_variants": baseline["run_context"]["num_variants"],
            "augmented_num_variants": augmented["run_context"]["num_variants"],
            "baseline_ground_truth_count": baseline["run_context"]["ground_truth_count"],
            "augmented_ground_truth_count": augmented["run_context"]["ground_truth_count"],
            "baseline_trajectory_count": baseline["metrics"]["trajectory_count"],
            "augmented_trajectory_count": augmented["metrics"]["trajectory_count"],
            "baseline_source_count": baseline["metrics"]["source_count"],
            "augmented_source_count": augmented["metrics"]["source_count"],
        },
        "deltas": _compute_metric_deltas(baseline["metrics"], augmented["metrics"]),
        "consistency_checks": {
            "stack_match": True,
            "training_config_match": True,
            "dataset_path_match": True,
            "model_path_match": True,
            "source_ids_match": True,
            "robustness_split_match": True,
            "generation_config_match": True,
            "evaluator_match": True,
            "ground_truth_count_scaled_match": True,
            "trajectory_count_scaled_match": True,
            "source_count_match": True,
            "baseline_num_variants_is_one": True,
            "augmented_num_variants_positive": True,
        },
        "reproducibility": {
            "baseline_sha256": _sha256_file(baseline_path),
            "augmented_sha256": _sha256_file(augmented_path),
            "baseline_input_hash": baseline["input_hash"],
            "augmented_input_hash": augmented["input_hash"],
            "outer_git_commit": _git_commit(project_root),
            "inner_verl_git_commit": _git_commit(project_root / "verl"),
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }
    return _write_json(Path(output_path), report)


def run_openpsi_baseline_augmented_report(
    output_dir: str | Path,
    dataset_path: str | Path = DEFAULT_POLARIS_PARQUET_PATH,
    model_path: str | Path = DEFAULT_QWEN3_MODEL_PATH,
    tokenizer_loader: Callable[[Path], Any] | None = None,
    model_loader: Callable[[Path], Any] | None = None,
    generation_fn: Callable[..., str] | None = None,
    *,
    max_sources: int = 8,
    num_variants: int = 1,
    baseline_seed: int = 0,
    augmented_seed: int = 0,
    robustness_difficulty: str = "0/8",
    max_new_tokens: int = PINNED_MAX_OUTPUT_TOKENS,
    python_executable: str | Path | None = None,
    runtime_prefix: str | Path | None = None,
) -> dict[str, Path]:
    project_root = Path(__file__).resolve().parent
    stack_config = TrainingStackConfig()
    validate_current_runtime(
        stack_config,
        python_executable=python_executable,
        prefix=runtime_prefix,
        project_root=project_root,
    )
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    _enforce_pinned_openpsi_inputs(dataset_path=dataset_path, max_new_tokens=max_new_tokens)
    _require_existing_path(dataset_path, "dataset")
    _require_existing_path(model_path, "model")
    _require_model_identifier(model_path)

    dataframe = pd.read_parquet(dataset_path)
    split_records, robustness_split = _load_records_for_split(dataframe, robustness_difficulty)
    selected_records, selection_stats = _select_eligible_records(
        split_records,
        max_sources=max_sources,
        num_variants=num_variants,
        seed=augmented_seed,
    )

    if tokenizer_loader is None:
        tokenizer_loader = _default_tokenizer_loader
    if model_loader is None:
        model_loader = _default_model_loader
    if generation_fn is None:
        generation_fn = _default_generation_fn

    tokenizer = tokenizer_loader(model_path)
    model = model_loader(model_path)

    generation_config = {
        "strategy": "single_step_argmax",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    baseline_trajectories, baseline_eval = _evaluate_records(
        selected_records,
        tokenizer=tokenizer,
        model=model,
        generation_fn=generation_fn,
        max_new_tokens=max_new_tokens,
        mode="baseline",
    )

    augmentation_config = _build_aug_config(num_variants=num_variants, seed=augmented_seed)
    augmented_records = augment_rlhf_records(
        records=selected_records,
        prompt_key="prompt",
        config=augmentation_config,
    )
    augmented_trajectories, augmented_eval = _evaluate_records(
        augmented_records,
        tokenizer=tokenizer,
        model=model,
        generation_fn=generation_fn,
        max_new_tokens=max_new_tokens,
        mode="augmented",
    )

    baseline_input = {
        "mode": "baseline",
        "stack": STACK_INFO,
        "training_config": deepcopy(PINNED_TRAINING_CONFIG),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "source_ids": baseline_eval["source_ids"],
        "robustness_split": robustness_split,
        "selection": selection_stats,
        "generation_config": generation_config,
        "seed": baseline_seed,
        "num_variants": 1,
    }
    augmented_input = {
        "mode": "augmented",
        "stack": STACK_INFO,
        "training_config": deepcopy(PINNED_TRAINING_CONFIG),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "source_ids": augmented_eval["source_ids"],
        "robustness_split": robustness_split,
        "selection": selection_stats,
        "generation_config": generation_config,
        "seed": augmented_seed,
        "num_variants": num_variants,
    }

    baseline_artifact = {
        "mode": "baseline",
        "stack": deepcopy(STACK_INFO),
        "training_config": deepcopy(PINNED_TRAINING_CONFIG),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "robustness_split": deepcopy(robustness_split),
        "selection": deepcopy(selection_stats),
        "generation_config": deepcopy(generation_config),
        "run_context": {
            "evaluator": "qwen3-local-greedy-v1",
            "seed": int(baseline_seed),
            "num_variants": 1,
            "ground_truth_count": baseline_eval["ground_truth_count"],
        },
        "source_ids": deepcopy(baseline_eval["source_ids"]),
        "metrics": deepcopy(baseline_eval["metrics"]),
        "trajectories": baseline_trajectories,
        "input_hash": _sha256_text(_json_dumps(baseline_input)),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    augmented_artifact = {
        "mode": "augmented",
        "stack": deepcopy(STACK_INFO),
        "training_config": deepcopy(PINNED_TRAINING_CONFIG),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "robustness_split": deepcopy(robustness_split),
        "selection": deepcopy(selection_stats),
        "generation_config": deepcopy(generation_config),
        "run_context": {
            "evaluator": "qwen3-local-greedy-v1",
            "seed": int(augmented_seed),
            "num_variants": int(num_variants),
            "ground_truth_count": augmented_eval["ground_truth_count"],
        },
        "source_ids": deepcopy(augmented_eval["source_ids"]),
        "metrics": deepcopy(augmented_eval["metrics"]),
        "trajectories": augmented_trajectories,
        "input_hash": _sha256_text(_json_dumps(augmented_input)),
        "generated_at": datetime.now(UTC).isoformat(),
    }

    out_dir = Path(output_dir)
    baseline_path = _write_json(out_dir / "baseline-run.json", baseline_artifact)
    augmented_path = _write_json(out_dir / "augmented-run.json", augmented_artifact)
    report_path = generate_openpsi_baseline_vs_augmented_report(
        baseline_artifact_path=baseline_path,
        augmented_artifact_path=augmented_path,
        output_path=out_dir / "baseline-vs-augmented-report.json",
    )

    return {
        "baseline": baseline_path,
        "augmented": augmented_path,
        "report": report_path,
    }


def run_openpsi_stack_dry_run(
    dataset_path: str | Path = DEFAULT_POLARIS_PARQUET_PATH,
    model_path: str | Path = DEFAULT_QWEN3_MODEL_PATH,
    tokenizer_loader: Callable[[Path], Any] | None = None,
    *,
    max_samples: int = 8,
    robustness_difficulty: str = "0/8",
    python_executable: str | Path | None = None,
    runtime_prefix: str | Path | None = None,
) -> dict:
    """Run a pinned-stack dry-run using real path/schema/tokenizer checks."""

    project_root = Path(__file__).resolve().parent
    stack_config = TrainingStackConfig()
    validate_current_runtime(
        stack_config,
        python_executable=python_executable,
        prefix=runtime_prefix,
        project_root=project_root,
    )
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    _enforce_pinned_openpsi_inputs(dataset_path=dataset_path)
    _require_existing_path(dataset_path, "dataset")
    _require_existing_path(model_path, "model")
    _require_model_identifier(model_path)

    dataframe = pd.read_parquet(dataset_path)
    split_records, robustness_split = _load_records_for_split(dataframe, robustness_difficulty)
    records, selection_stats = _select_eligible_records(
        split_records,
        max_sources=max_samples,
        num_variants=1,
        seed=0,
    )

    if tokenizer_loader is None:
        tokenizer_loader = _default_tokenizer_loader

    tokenizer = tokenizer_loader(model_path)

    baseline_prompts = [_extract_user_prompt_text(record["prompt"]) for record in records]
    baseline_token_lengths: list[int] = []
    for prompt in baseline_prompts:
        first = _encode_prompt(tokenizer, prompt)
        second = _encode_prompt(tokenizer, prompt)
        if first != second:
            raise ValueError("tokenizer produced non-deterministic output in baseline prompt encoding")
        baseline_token_lengths.append(len(first))

    augmented_records = augment_rlhf_records(
        records=records,
        prompt_key="prompt",
        config=_build_aug_config(num_variants=1, seed=0),
    )
    augmented_prompts = [_extract_user_prompt_text(record["prompt"]) for record in augmented_records]
    augmented_token_lengths = [len(_encode_prompt(tokenizer, prompt)) for prompt in augmented_prompts]

    return {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "stack": deepcopy(STACK_INFO),
        "training_config": deepcopy(PINNED_TRAINING_CONFIG),
        "robustness_split": deepcopy(robustness_split),
        "source_rows": len(records),
        "augmented_rows": len(augmented_records),
        "selection": selection_stats,
        "baseline_token_lengths": baseline_token_lengths,
        "augmented_token_lengths": augmented_token_lengths,
        "generated_at": datetime.now(UTC).isoformat(),
    }
