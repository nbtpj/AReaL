"""Polaris dataset normalization utilities for GRPO trajectory construction."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from grpo_trajectory_generation import build_grpo_trajectory_inputs
from training_config import TrainingStackConfig


def load_polaris_records(
    raw_records: Iterable[dict],
    question_field: str = "question",
    source_id_field: str = "id",
) -> list[dict]:
    """Normalize raw polaris records into trajectory-ready sample dictionaries."""

    samples: list[dict] = []
    for idx, record in enumerate(raw_records):
        if question_field not in record:
            raise ValueError(f"polaris record missing question field '{question_field}' at index {idx}")
        if source_id_field not in record:
            raise ValueError(f"polaris record missing source field '{source_id_field}' at index {idx}")

        normalized = deepcopy(record)
        normalized["source_id"] = str(record[source_id_field])
        normalized["prompt"] = str(record[question_field])
        samples.append(normalized)
    return samples


def build_grpo_trajectories_from_polaris(
    raw_records: Iterable[dict],
    config: TrainingStackConfig,
    seed: int,
    question_field: str = "question",
    source_id_field: str = "id",
    protected_tokens_by_source: dict[str, tuple[str, ...]] | None = None,
) -> list[dict]:
    """Load `polaris` records and convert them into GRPO trajectory inputs."""

    samples = load_polaris_records(
        raw_records=raw_records,
        question_field=question_field,
        source_id_field=source_id_field,
    )
    return build_grpo_trajectory_inputs(
        samples=samples,
        config=config,
        seed=seed,
        protected_tokens_by_source=protected_tokens_by_source,
    )
