"""VERL-compatible adapter for Deepscaler math correctness reward."""

from __future__ import annotations

from typing import Any

from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_mathd, grade_answer_sympy


def _ground_truths(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get("answer", value.get("ground_truth"))

    values = value if isinstance(value, (list, tuple)) else [value]
    processed: list[str] = []
    for truth in values:
        if truth is None:
            continue
        truth_str = str(truth)
        if "\\boxed" in truth_str:
            extracted = extract_answer(truth_str)
            if extracted is not None:
                processed.append(extracted)
        else:
            processed.append(truth_str)
    return processed


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    strip_comma_from_answer: bool = True,
    correct_reward: float = 1.0,
    incorrect_reward: float = -1.0,
    **_: Any,
) -> dict[str, Any]:
    """Score a response using Deepscaler's boxed-answer math grader.

    The active VERL reward manager passes only decoded text and ground truth to
    custom reward functions, so this adapter intentionally uses only the
    correctness portion of ThreadWeaver/Deepscaler v2: extract the last boxed
    answer from the full response, normalize, and compare with mathd/sympy.
    """

    del data_source, extra_info

    pred = extract_answer(str(solution_str))
    if strip_comma_from_answer and pred is not None:
        pred = pred.replace(",", "")

    correct = False
    if pred is not None:
        for truth in _ground_truths(ground_truth):
            if grade_answer_mathd(pred, truth) or grade_answer_sympy(pred, truth):
                correct = True
                break

    return {
        "score": float(correct_reward if correct else incorrect_reward),
        "acc": bool(correct),
        "correct": bool(correct),
        "pred": pred if pred is not None else "[INVALID]",
    }
