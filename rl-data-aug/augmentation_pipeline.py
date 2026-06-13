"""Deterministic question augmentation with protected-token safeguards."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable


WORD_PATTERN = re.compile(r"\w+|\W+")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
WORD_ONLY_PATTERN = re.compile(r"\b\w+\b")


DEFAULT_SYNONYM_MAP: dict[str, str | tuple[str, ...]] = {
    "solve": ("answer", "work out"),
    "find": ("locate", "determine"),
    "small": ("little", "compact"),
    "large": ("big", "sizable"),
    "quickly": ("swiftly", "rapidly"),
    "show": ("demonstrate", "present"),
}
SUPPORTED_AUGMENTATION_OPERATORS = {
    "synonym_substitution",
    "word_shuffle",
    "typo_noise",
}


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source.istitle():
        return target.title()
    return target.lower()


def _normalize_token(token: str) -> str:
    return token.strip().lower()


def _protected_token_set(question: str, protected_tokens: Iterable[str]) -> set[str]:
    explicit = {_normalize_token(token) for token in protected_tokens if token.strip()}
    numerics = {_normalize_token(token) for token in NUMBER_PATTERN.findall(question)}
    return explicit | numerics


def validate_variant_constraints(
    source_question: str, variant_question: str, protected_tokens: Iterable[str] = ()
) -> None:
    """Validate that protected tokens and numeric constraints are unchanged."""

    source_numbers = NUMBER_PATTERN.findall(source_question)
    variant_numbers = NUMBER_PATTERN.findall(variant_question)
    if source_numbers != variant_numbers:
        raise ValueError("variant changed numeric constraints")

    source_counts = Counter(_normalize_token(token) for token in WORD_ONLY_PATTERN.findall(source_question))
    variant_counts = Counter(_normalize_token(token) for token in WORD_ONLY_PATTERN.findall(variant_question))

    protected = _protected_token_set(source_question, protected_tokens)
    for token in protected:
        if source_counts[token] != variant_counts[token]:
            raise ValueError(f"variant changed protected token: '{token}'")


def _choose_replacement(value: str | tuple[str, ...], seed: int) -> str:
    if isinstance(value, tuple):
        if not value:
            raise ValueError("synonym map entry tuple cannot be empty")
        return value[seed % len(value)]
    return value


def _eligible_word_indices(tokens: list[str], protected_tokens: set[str], *, min_length: int = 1) -> list[int]:
    eligible_indices: list[int] = []
    for idx, token in enumerate(tokens):
        normalized = _normalize_token(token)
        if not token.isalpha():
            continue
        if normalized in protected_tokens:
            continue
        if len(token) < min_length:
            continue
        eligible_indices.append(idx)
    return eligible_indices


def _normalize_augmentation_operators(operators: Iterable[str]) -> tuple[str, ...]:
    operator_tuple = tuple(operators)
    if not operator_tuple:
        raise ValueError("at least one augmentation operator is required")
    unsupported = set(operator_tuple) - SUPPORTED_AUGMENTATION_OPERATORS
    if unsupported:
        raise ValueError(f"unsupported augmentation operators: {sorted(unsupported)}")
    return operator_tuple


def select_augmentation_operator(operators: Iterable[str], variant_index: int) -> str:
    """Select the deterministic augmentation operator for a variant index."""

    operator_tuple = _normalize_augmentation_operators(operators)
    return operator_tuple[variant_index % len(operator_tuple)]


def generate_synonym_variant(
    question: str,
    seed: int,
    protected_tokens: Iterable[str] = (),
    synonym_map: dict[str, str | tuple[str, ...]] | None = None,
) -> str:
    """Generate one deterministic synonym-substitution variant."""

    mapping = synonym_map or DEFAULT_SYNONYM_MAP
    tokens = WORD_PATTERN.findall(question)
    protected = _protected_token_set(question, protected_tokens)

    eligible_indices: list[int] = []
    for idx, token in enumerate(tokens):
        normalized = _normalize_token(token)
        if not token.isalpha():
            continue
        if normalized in protected:
            continue
        if normalized in mapping:
            eligible_indices.append(idx)

    if not eligible_indices:
        raise ValueError("no eligible token available for synonym substitution")

    selected_token_idx = eligible_indices[seed % len(eligible_indices)]
    selected_token = tokens[selected_token_idx]
    replacement = _choose_replacement(mapping[_normalize_token(selected_token)], seed)
    tokens[selected_token_idx] = _match_case(selected_token, replacement)

    variant = "".join(tokens)
    if variant == question:
        raise ValueError("synonym substitution produced no textual change")
    validate_variant_constraints(question, variant, protected_tokens=protected_tokens)
    return variant


def generate_word_shuffle_variant(
    question: str,
    seed: int,
    protected_tokens: Iterable[str] = (),
) -> str:
    """Generate one deterministic word-order perturbation variant."""

    tokens = WORD_PATTERN.findall(question)
    protected = _protected_token_set(question, protected_tokens)
    eligible_indices = _eligible_word_indices(tokens, protected)

    if len(eligible_indices) < 2:
        raise ValueError("at least two eligible tokens are required for word shuffle")

    first_pos = seed % len(eligible_indices)
    offset = 1 + ((seed // len(eligible_indices)) % (len(eligible_indices) - 1))
    second_pos = (first_pos + offset) % len(eligible_indices)
    first_idx = eligible_indices[first_pos]
    second_idx = eligible_indices[second_pos]
    tokens[first_idx], tokens[second_idx] = tokens[second_idx], tokens[first_idx]

    variant = "".join(tokens)
    if variant == question:
        raise ValueError("word shuffle produced no textual change")
    validate_variant_constraints(question, variant, protected_tokens=protected_tokens)
    return variant


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


def generate_typo_noise_variant(
    question: str,
    seed: int,
    protected_tokens: Iterable[str] = (),
) -> str:
    """Generate one deterministic typo/noise perturbation variant."""

    tokens = WORD_PATTERN.findall(question)
    protected = _protected_token_set(question, protected_tokens)
    eligible_indices = _eligible_word_indices(tokens, protected, min_length=2)

    if not eligible_indices:
        raise ValueError("no eligible token available for typo/noise injection")

    selected_idx = eligible_indices[seed % len(eligible_indices)]
    tokens[selected_idx] = _inject_typo(tokens[selected_idx], seed)

    variant = "".join(tokens)
    if variant == question:
        raise ValueError("typo/noise injection produced no textual change")
    validate_variant_constraints(question, variant, protected_tokens=protected_tokens)
    return variant


def generate_question_variant(
    question: str,
    seed: int,
    operator: str = "synonym_substitution",
    protected_tokens: Iterable[str] = (),
    synonym_map: dict[str, str | tuple[str, ...]] | None = None,
) -> str:
    """Generate one deterministic question variant for the selected operator."""

    if operator == "synonym_substitution":
        return generate_synonym_variant(
            question=question,
            seed=seed,
            protected_tokens=protected_tokens,
            synonym_map=synonym_map,
        )
    if operator == "word_shuffle":
        return generate_word_shuffle_variant(
            question=question,
            seed=seed,
            protected_tokens=protected_tokens,
        )
    if operator == "typo_noise":
        return generate_typo_noise_variant(
            question=question,
            seed=seed,
            protected_tokens=protected_tokens,
        )
    raise ValueError(f"unsupported augmentation operator: {operator}")


def generate_question_variants(
    question: str,
    num_variants: int,
    seed: int,
    protected_tokens: Iterable[str] = (),
    synonym_map: dict[str, str | tuple[str, ...]] | None = None,
    operators: Iterable[str] = ("synonym_substitution",),
) -> list[str]:
    """Generate a deterministic list of question variants."""

    if num_variants <= 0:
        raise ValueError("num_variants must be positive")

    operator_tuple = _normalize_augmentation_operators(operators)
    variants: list[str] = []
    for variant_idx in range(num_variants):
        variant_seed = seed + variant_idx
        variant = generate_question_variant(
            question=question,
            seed=variant_seed,
            operator=select_augmentation_operator(operator_tuple, variant_idx),
            protected_tokens=protected_tokens,
            synonym_map=synonym_map,
        )
        variants.append(variant)
    return variants
