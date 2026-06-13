"""Validation helpers for Qwen3-1.7B model initialization and tokenization checks."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Callable, Iterable, Sequence

from training_config import TrainingStackConfig


_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Qwen3InitializationSpec:
    """Initialization locations for policy/reference models and tokenizer."""

    policy_model_path: str
    reference_model_path: str
    tokenizer_path: str


def _normalize_locator(locator: str) -> str:
    return _NON_ALNUM_PATTERN.sub("", locator.strip().lower())


def _is_qwen3_1_7b_locator(locator: str) -> bool:
    normalized = _normalize_locator(locator)
    return "qwen3" in normalized and "17b" in normalized


def _validate_qwen_locator(field_name: str, locator: str) -> None:
    if not locator or not locator.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if not _is_qwen3_1_7b_locator(locator):
        raise ValueError(f"{field_name} must resolve to Qwen3-1.7B, got '{locator}'")


def validate_qwen3_initialization_spec(
    spec: Qwen3InitializationSpec,
    *,
    require_existing_paths: bool = False,
    path_exists: Callable[[str], bool] | None = None,
) -> None:
    """Validate policy/reference/tokenizer initialization locations."""

    _validate_qwen_locator("policy_model_path", spec.policy_model_path)
    _validate_qwen_locator("reference_model_path", spec.reference_model_path)
    _validate_qwen_locator("tokenizer_path", spec.tokenizer_path)

    if not require_existing_paths:
        return

    path_exists = path_exists or os.path.exists
    for field_name, locator in (
        ("policy_model_path", spec.policy_model_path),
        ("reference_model_path", spec.reference_model_path),
        ("tokenizer_path", spec.tokenizer_path),
    ):
        if not path_exists(locator):
            raise ValueError(f"{field_name} does not exist: '{locator}'")


def validate_prompt_tokenization(
    prompts: Iterable[str],
    encode: Callable[[str], Sequence[int]],
) -> None:
    """Ensure prompts are tokenizable and stable under repeated encoding."""

    prompt_list = list(prompts)
    if not prompt_list:
        raise ValueError("at least one prompt is required for tokenization validation")

    for idx, prompt in enumerate(prompt_list):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt at index {idx} must be a non-empty string")

        encoded_first = list(encode(prompt))
        encoded_second = list(encode(prompt))
        if encoded_first != encoded_second:
            raise ValueError(f"tokenizer output is non-deterministic for prompt index {idx}")
        if not encoded_first:
            raise ValueError(f"tokenizer produced empty output for prompt index {idx}")
        if not all(isinstance(token, int) and not isinstance(token, bool) for token in encoded_first):
            raise ValueError("tokenizer output must be an integer token sequence")


def validate_qwen3_stack_and_tokenization(
    config: TrainingStackConfig,
    spec: Qwen3InitializationSpec,
    prompts: Iterable[str],
    encode: Callable[[str], Sequence[int]],
    *,
    require_existing_paths: bool = False,
    path_exists: Callable[[str], bool] | None = None,
) -> None:
    """Run a dry-run stack compatibility check for model init and tokenizer behavior."""

    config.validate()
    validate_qwen3_initialization_spec(
        spec=spec,
        require_existing_paths=require_existing_paths,
        path_exists=path_exists,
    )
    validate_prompt_tokenization(prompts=prompts, encode=encode)
