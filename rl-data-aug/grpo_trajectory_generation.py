"""GRPO trajectory input construction with optional deterministic augmentation."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from augmentation_pipeline import generate_question_variants, select_augmentation_operator
from training_config import TrainingStackConfig


SUPPORTED_AUG_OPS = {"synonym_substitution", "word_shuffle", "typo_noise"}
REQUIRED_PROVENANCE_FIELDS = {"source_id", "aug_op", "seed"}


def validate_trajectory_record(record: dict) -> None:
    """Validate one trajectory record and its provenance schema."""

    if "source_id" not in record:
        raise ValueError("trajectory record is missing required field: source_id")
    if "prompt" not in record:
        raise ValueError("trajectory record is missing required field: prompt")

    if "augmentation" not in record:
        return

    augmentation = record["augmentation"]
    if not isinstance(augmentation, dict):
        raise ValueError("augmentation field must be a dictionary")

    missing = REQUIRED_PROVENANCE_FIELDS - set(augmentation.keys())
    if missing:
        raise ValueError(f"augmentation metadata missing required fields: {sorted(missing)}")

    if augmentation["aug_op"] not in SUPPORTED_AUG_OPS:
        raise ValueError(f"unsupported augmentation operator: {augmentation['aug_op']}")

    if not isinstance(augmentation["seed"], int):
        raise ValueError("augmentation seed must be an integer")

    if str(augmentation["source_id"]) != str(record["source_id"]):
        raise ValueError("augmentation source_id must match record source_id")


def build_grpo_trajectory_inputs(
    samples: list[dict],
    config: TrainingStackConfig,
    seed: int,
    protected_tokens_by_source: dict[str, Iterable[str]] | None = None,
) -> list[dict]:
    """Build trajectory inputs from source samples with optional augmentation expansion."""

    config.validate()
    protected_tokens_by_source = protected_tokens_by_source or {}
    baseline = deepcopy(samples)

    if not config.augmentation_enabled:
        return baseline

    enabled_ops = set(config.augmentation_operators)
    unsupported = enabled_ops - SUPPORTED_AUG_OPS
    if unsupported:
        raise ValueError(f"unsupported augmentation operators: {sorted(unsupported)}")

    trajectories: list[dict] = []
    for sample_idx, sample in enumerate(samples):
        validate_trajectory_record(sample)

        source_id = str(sample["source_id"])
        prompt = str(sample["prompt"])
        protected_tokens = tuple(protected_tokens_by_source.get(source_id, ()))
        sample_seed = seed + sample_idx * 1000

        variants = generate_question_variants(
            question=prompt,
            num_variants=config.num_variants,
            seed=sample_seed,
            protected_tokens=protected_tokens,
            operators=config.augmentation_operators,
        )

        for variant_idx, variant_prompt in enumerate(variants):
            aug_op = select_augmentation_operator(config.augmentation_operators, variant_idx)
            item = deepcopy(sample)
            item["source_id"] = source_id
            item["prompt"] = variant_prompt
            item["augmentation"] = {
                "enabled": True,
                "source_id": source_id,
                "aug_op": aug_op,
                "seed": sample_seed + variant_idx,
                "variant_index": variant_idx,
                "rl_framework": config.rl_framework,
                "dataset": config.dataset,
                "base_model": config.base_model,
            }
            validate_trajectory_record(item)
            trajectories.append(item)

    return trajectories
