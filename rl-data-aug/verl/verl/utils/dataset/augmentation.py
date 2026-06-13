# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic RL dataset augmentation utilities for text prompts."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any


SUPPORTED_AUGMENTATION_OPERATORS = {"synonym_substitution", "word_shuffle", "typo_noise"}
ALLOWED_RL_FRAMEWORK = "verl"
ALLOWED_DATASET = "polaris"
ALLOWED_BASE_MODEL = "Qwen3-1.7B"
REQUIRED_PROVENANCE_FIELDS = {
    "source_id",
    "aug_op",
    "seed",
    "variant_index",
    "rl_framework",
    "dataset",
    "base_model",
}

WORD_PATTERN = re.compile(r"\w+|\W+")
WORD_ONLY_PATTERN = re.compile(r"\b\w+\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")

DEFAULT_SYNONYM_MAP: dict[str, str | tuple[str, ...]] = {
    "solve": ("answer", "work out"),
    "find": ("locate", "determine"),
    "small": ("little", "compact"),
    "large": ("big", "sizable"),
    "quickly": ("swiftly", "rapidly"),
    "show": ("demonstrate", "present"),
}


@dataclass(frozen=True)
class RLDataAugmentationConfig:
    enabled: bool = False
    operators: tuple[str, ...] = tuple()
    num_variants: int = 1
    seed: int = 0
    source_id_field: str = "extra_info.index"
    protected_fields: tuple[str, ...] = tuple()
    semantic_similarity_threshold: float = 0.5
    rl_framework: str = ALLOWED_RL_FRAMEWORK
    dataset: str = ALLOWED_DATASET
    base_model: str = ALLOWED_BASE_MODEL

    def validate(self) -> None:
        if self.num_variants <= 0:
            raise ValueError(f"num_variants must be positive, got {self.num_variants}")
        if self.semantic_similarity_threshold < 0.0 or self.semantic_similarity_threshold > 1.0:
            raise ValueError(
                "semantic_similarity_threshold must be in [0, 1], "
                f"got {self.semantic_similarity_threshold}"
            )
        if self.rl_framework != ALLOWED_RL_FRAMEWORK:
            raise ValueError(
                f"rl_framework must be '{ALLOWED_RL_FRAMEWORK}', got '{self.rl_framework}'"
            )
        if self.dataset != ALLOWED_DATASET:
            raise ValueError(f"dataset must be '{ALLOWED_DATASET}', got '{self.dataset}'")
        if self.base_model != ALLOWED_BASE_MODEL:
            raise ValueError(f"base_model must be '{ALLOWED_BASE_MODEL}', got '{self.base_model}'")
        if not self.source_id_field.strip():
            raise ValueError("source_id_field must be a non-empty string")
        if not self.enabled:
            return
        if not self.operators:
            raise ValueError("enabled augmentation requires at least one operator")
        unsupported = set(self.operators) - SUPPORTED_AUGMENTATION_OPERATORS
        if unsupported:
            raise ValueError(f"unsupported augmentation operators: {sorted(unsupported)}")


def build_augmentation_config(data_config: Any) -> RLDataAugmentationConfig:
    aug_cfg = data_config.get("augmentation", {})
    config = RLDataAugmentationConfig(
        enabled=bool(aug_cfg.get("enabled", False)),
        operators=tuple(aug_cfg.get("operators", [])),
        num_variants=int(aug_cfg.get("num_variants", 1)),
        seed=int(aug_cfg.get("seed", 0)),
        source_id_field=str(aug_cfg.get("source_id_field", "extra_info.index")),
        protected_fields=tuple(aug_cfg.get("protected_fields", [])),
        semantic_similarity_threshold=float(aug_cfg.get("semantic_similarity_threshold", 0.5)),
        rl_framework=str(aug_cfg.get("rl_framework", ALLOWED_RL_FRAMEWORK)),
        dataset=str(aug_cfg.get("dataset", ALLOWED_DATASET)),
        base_model=str(aug_cfg.get("base_model", ALLOWED_BASE_MODEL)),
    )
    config.validate()
    return config


def _normalize_token(token: str) -> str:
    return token.strip().lower()


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source.istitle():
        return target.title()
    return target.lower()


def _text_token_set(text: str) -> set[str]:
    return {_normalize_token(token) for token in WORD_ONLY_PATTERN.findall(text) if token.strip()}


def _token_jaccard_similarity(source: str, variant: str) -> float:
    source_tokens = _text_token_set(source)
    variant_tokens = _text_token_set(variant)
    if not source_tokens and not variant_tokens:
        return 1.0
    if not source_tokens or not variant_tokens:
        return 0.0
    intersection = source_tokens & variant_tokens
    union = source_tokens | variant_tokens
    return len(intersection) / len(union)


def _protected_token_set(source_text: str, protected_tokens: tuple[str, ...]) -> set[str]:
    explicit = {_normalize_token(token) for token in protected_tokens if token and token.strip()}
    numeric_tokens = {_normalize_token(token) for token in NUMBER_PATTERN.findall(source_text)}
    return explicit | numeric_tokens


def _validate_variant_constraints(
    source_text: str,
    variant_text: str,
    protected_tokens: tuple[str, ...],
    similarity_threshold: float,
) -> None:
    source_numbers = NUMBER_PATTERN.findall(source_text)
    variant_numbers = NUMBER_PATTERN.findall(variant_text)
    if source_numbers != variant_numbers:
        raise ValueError("variant changed numeric constraints")

    source_counts = Counter(_normalize_token(token) for token in WORD_ONLY_PATTERN.findall(source_text))
    variant_counts = Counter(_normalize_token(token) for token in WORD_ONLY_PATTERN.findall(variant_text))
    for token in _protected_token_set(source_text, protected_tokens):
        if source_counts[token] != variant_counts[token]:
            raise ValueError(f"variant changed protected token: '{token}'")

    similarity = _token_jaccard_similarity(source_text, variant_text)
    if similarity < similarity_threshold:
        raise ValueError(
            "variant semantic similarity below threshold: "
            f"{similarity:.4f} < {similarity_threshold:.4f}"
        )


def _choose_replacement(value: str | tuple[str, ...], seed: int) -> str:
    if isinstance(value, tuple):
        if not value:
            raise ValueError("synonym map tuple must not be empty")
        return value[seed % len(value)]
    return value


def _select_operator(operators: tuple[str, ...], variant_index: int) -> str:
    if not operators:
        raise ValueError("at least one augmentation operator is required")
    unsupported = set(operators) - SUPPORTED_AUGMENTATION_OPERATORS
    if unsupported:
        raise ValueError(f"unsupported augmentation operators: {sorted(unsupported)}")
    return operators[variant_index % len(operators)]


def _eligible_word_indices(tokens: list[str], protected_tokens: set[str], *, min_length: int = 1) -> list[int]:
    eligible_indices: list[int] = []
    for idx, token in enumerate(tokens):
        if not token.isalpha():
            continue
        normalized = _normalize_token(token)
        if normalized in protected_tokens:
            continue
        if len(token) < min_length:
            continue
        eligible_indices.append(idx)
    return eligible_indices


def _generate_synonym_variant(
    source_text: str,
    seed: int,
    protected_tokens: tuple[str, ...],
    similarity_threshold: float,
    synonym_map: dict[str, str | tuple[str, ...]] | None = None,
) -> tuple[str, bool]:
    mapping = synonym_map or DEFAULT_SYNONYM_MAP
    tokens = WORD_PATTERN.findall(source_text)
    protected = _protected_token_set(source_text, protected_tokens)

    eligible_indices: list[int] = []
    for idx, token in enumerate(tokens):
        if not token.isalpha():
            continue
        normalized = _normalize_token(token)
        if normalized in protected:
            continue
        if normalized in mapping:
            eligible_indices.append(idx)

    if not eligible_indices:
        return source_text, False

    selected_idx = eligible_indices[seed % len(eligible_indices)]
    selected_token = tokens[selected_idx]
    replacement = _choose_replacement(mapping[_normalize_token(selected_token)], seed)
    tokens[selected_idx] = _match_case(selected_token, replacement)
    variant = "".join(tokens)
    _validate_variant_constraints(
        source_text=source_text,
        variant_text=variant,
        protected_tokens=protected_tokens,
        similarity_threshold=similarity_threshold,
    )
    return variant, variant != source_text


def _generate_word_shuffle_variant(
    source_text: str,
    seed: int,
    protected_tokens: tuple[str, ...],
    similarity_threshold: float,
) -> tuple[str, bool]:
    tokens = WORD_PATTERN.findall(source_text)
    protected = _protected_token_set(source_text, protected_tokens)
    eligible_indices = _eligible_word_indices(tokens, protected)

    if len(eligible_indices) < 2:
        return source_text, False

    first_pos = seed % len(eligible_indices)
    offset = 1 + ((seed // len(eligible_indices)) % (len(eligible_indices) - 1))
    second_pos = (first_pos + offset) % len(eligible_indices)
    first_idx = eligible_indices[first_pos]
    second_idx = eligible_indices[second_pos]
    tokens[first_idx], tokens[second_idx] = tokens[second_idx], tokens[first_idx]

    variant = "".join(tokens)
    _validate_variant_constraints(
        source_text=source_text,
        variant_text=variant,
        protected_tokens=protected_tokens,
        similarity_threshold=similarity_threshold,
    )
    return variant, variant != source_text


def _inject_typo(token: str, seed: int) -> str:
    actions = ("swap", "duplicate", "delete")
    ordered_actions = actions[seed % len(actions) :] + actions[: seed % len(actions)]
    for action in ordered_actions:
        if action == "swap" and len(token) >= 2:
            start = seed % (len(token) - 1)
            for offset in range(len(token) - 1):
                idx = (start + offset) % (len(token) - 1)
                if token[idx] == token[idx + 1]:
                    continue
                chars = list(token)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                typo = "".join(chars)
                if typo != token:
                    return typo
        if action == "duplicate":
            idx = seed % len(token)
            typo = token[:idx] + token[idx] + token[idx:]
            if typo != token:
                return typo
        if action == "delete" and len(token) >= 4:
            idx = seed % len(token)
            typo = token[:idx] + token[idx + 1 :]
            if typo != token:
                return typo
    raise ValueError("typo/noise injection produced no textual change")


def _generate_typo_noise_variant(
    source_text: str,
    seed: int,
    protected_tokens: tuple[str, ...],
    similarity_threshold: float,
) -> tuple[str, bool]:
    tokens = WORD_PATTERN.findall(source_text)
    protected = _protected_token_set(source_text, protected_tokens)
    eligible_indices = _eligible_word_indices(tokens, protected, min_length=2)

    if not eligible_indices:
        return source_text, False

    selected_idx = eligible_indices[seed % len(eligible_indices)]
    tokens[selected_idx] = _inject_typo(tokens[selected_idx], seed)
    variant = "".join(tokens)
    _validate_variant_constraints(
        source_text=source_text,
        variant_text=variant,
        protected_tokens=protected_tokens,
        similarity_threshold=similarity_threshold,
    )
    return variant, variant != source_text


def _generate_variant(
    source_text: str,
    seed: int,
    operator: str,
    protected_tokens: tuple[str, ...],
    similarity_threshold: float,
) -> tuple[str, bool]:
    if operator == "synonym_substitution":
        return _generate_synonym_variant(
            source_text=source_text,
            seed=seed,
            protected_tokens=protected_tokens,
            similarity_threshold=similarity_threshold,
        )
    if operator == "word_shuffle":
        return _generate_word_shuffle_variant(
            source_text=source_text,
            seed=seed,
            protected_tokens=protected_tokens,
            similarity_threshold=similarity_threshold,
        )
    if operator == "typo_noise":
        return _generate_typo_noise_variant(
            source_text=source_text,
            seed=seed,
            protected_tokens=protected_tokens,
            similarity_threshold=similarity_threshold,
        )
    raise ValueError(f"unsupported augmentation operator: {operator}")


def _augment_prompt_field(
    prompt: Any,
    seed: int,
    operator: str,
    protected_tokens: tuple[str, ...],
    similarity_threshold: float,
) -> Any:
    if isinstance(prompt, str):
        variant, applied = _generate_variant(
            source_text=prompt,
            seed=seed,
            operator=operator,
            protected_tokens=protected_tokens,
            similarity_threshold=similarity_threshold,
        )
        if not applied:
            raise ValueError("no eligible token available for augmentation")
        return variant

    messages: list[Any] | None = None
    if isinstance(prompt, list | tuple):
        messages = list(prompt)
    elif hasattr(prompt, "tolist"):
        converted = prompt.tolist()
        if isinstance(converted, list):
            messages = converted
    if messages is None:
        raise ValueError(f"unsupported prompt structure for augmentation: {type(prompt)}")

    augmented_prompt = deepcopy(messages)
    for message in reversed(augmented_prompt):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).lower() != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            variant, applied = _generate_variant(
                source_text=content,
                seed=seed,
                operator=operator,
                protected_tokens=protected_tokens,
                similarity_threshold=similarity_threshold,
            )
            if not applied:
                raise ValueError("no eligible token available in user prompt content")
            message["content"] = variant
            return augmented_prompt
    raise ValueError("no user text content found in prompt structure for augmentation")


def _resolve_nested_field(record: dict, field_path: str) -> Any:
    current: Any = record
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_source_id(record: dict, row_idx: int, source_id_field: str) -> str:
    value = _resolve_nested_field(record, source_id_field)
    if value is not None:
        return str(value)
    if "source_id" in record and record["source_id"] is not None:
        return str(record["source_id"])
    if "id" in record and record["id"] is not None:
        return str(record["id"])
    raise ValueError(f"record missing source identifier field '{source_id_field}' at index {row_idx}")


def _collect_protected_tokens(record: dict, protected_fields: tuple[str, ...]) -> tuple[str, ...]:
    tokens: list[str] = []
    for field in protected_fields:
        value = _resolve_nested_field(record, field)
        if value is None:
            continue
        if isinstance(value, str):
            tokens.append(value)
        elif isinstance(value, list):
            tokens.extend(str(item) for item in value if item is not None)
        else:
            tokens.append(str(value))
    return tuple(tokens)


def validate_augmentation_provenance(augmentation: dict) -> None:
    missing = REQUIRED_PROVENANCE_FIELDS - set(augmentation.keys())
    if missing:
        raise ValueError(f"augmentation metadata missing required fields: {sorted(missing)}")
    if not isinstance(augmentation["seed"], int):
        raise ValueError("augmentation seed must be an integer")
    if not isinstance(augmentation["variant_index"], int):
        raise ValueError("augmentation variant_index must be an integer")
    if augmentation["aug_op"] not in SUPPORTED_AUGMENTATION_OPERATORS:
        raise ValueError(f"unsupported augmentation operator: {augmentation['aug_op']}")
    if augmentation["rl_framework"] != ALLOWED_RL_FRAMEWORK:
        raise ValueError(
            f"augmentation rl_framework must be '{ALLOWED_RL_FRAMEWORK}', "
            f"got '{augmentation['rl_framework']}'"
        )
    if augmentation["dataset"] != ALLOWED_DATASET:
        raise ValueError(
            f"augmentation dataset must be '{ALLOWED_DATASET}', got '{augmentation['dataset']}'"
        )
    if augmentation["base_model"] != ALLOWED_BASE_MODEL:
        raise ValueError(
            f"augmentation base_model must be '{ALLOWED_BASE_MODEL}', got '{augmentation['base_model']}'"
        )


def augment_rlhf_records(
    records: list[dict],
    prompt_key: str,
    config: RLDataAugmentationConfig,
) -> list[dict]:
    config.validate()
    if not config.enabled:
        return deepcopy(records)

    augmented_records: list[dict] = []
    for row_idx, record in enumerate(records):
        source_id = _resolve_source_id(record, row_idx=row_idx, source_id_field=config.source_id_field)
        protected_tokens = _collect_protected_tokens(record, config.protected_fields)

        for variant_idx in range(config.num_variants):
            variant_seed = config.seed + row_idx * 1000 + variant_idx
            variant_operator = _select_operator(config.operators, variant_idx)
            item = deepcopy(record)
            item[prompt_key] = _augment_prompt_field(
                prompt=item[prompt_key],
                seed=variant_seed,
                operator=variant_operator,
                protected_tokens=protected_tokens,
                similarity_threshold=config.semantic_similarity_threshold,
            )
            item["source_id"] = source_id

            augmentation = {
                "source_id": source_id,
                "aug_op": variant_operator,
                "seed": variant_seed,
                "variant_index": variant_idx,
                "rl_framework": config.rl_framework,
                "dataset": config.dataset,
                "base_model": config.base_model,
            }
            validate_augmentation_provenance(augmentation)
            extra_info = item.get("extra_info")
            if not isinstance(extra_info, dict):
                extra_info = {}
            extra_info["augmentation"] = augmentation
            item["extra_info"] = extra_info
            augmented_records.append(item)

    return augmented_records
