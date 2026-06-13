#!/usr/bin/env python3
"""Convert OpenR1 Qwen-rendered JSONL into Polaris-style VERL parquet."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "verl"))

from deepscaler.rewards.math_utils.utils import extract_answer  # noqa: E402


CHAT_RE = re.compile(
    r"^\s*<\|im_start\|>user\n(?P<content>.*)<\|im_end\|>\s*<\|im_start\|>assistant\s*$",
    re.DOTALL,
)

BOXED_INSTRUCTION_RE = re.compile(
    r"\s*Please\s+reason\s+step\s+by\s+step,\s+and\s+put\s+your\s+final\s+answer\s+within\s+"
    r"(?:\\+boxed\{\}|\x08oxed\{\})\.?\s*$",
    re.IGNORECASE,
)

SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
        ("difficulty", pa.string()),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([("index", pa.int64()), ("split", pa.string())])),
    ]
)


def extract_user_content(rendered_prompt: str) -> tuple[str, bool]:
    match = CHAT_RE.match(rendered_prompt)
    if match:
        return match.group("content"), True

    content = rendered_prompt
    content = content.removeprefix("<|im_start|>user\n")
    content = content.removesuffix("<|im_start|>assistant")
    content = content.replace("<|im_end|>", "")
    return content, False


def clean_prompt(content: str) -> tuple[str, bool]:
    cleaned, n_subs = BOXED_INSTRUCTION_RE.subn("", content)
    return cleaned.strip(), bool(n_subs)


def first_ground_truth(row: dict[str, Any]) -> str:
    solutions = row.get("solutions") or []
    if not solutions:
        raise ValueError(f"row {row.get('query_id')} has no solutions")

    solution = str(solutions[0])
    extracted = extract_answer(solution)
    return extracted if extracted is not None else solution


def convert(input_path: Path, output_path: Path, system_prompt: str | None = None) -> dict[str, int]:
    records: list[dict[str, Any]] = []
    matched_chat = 0
    stripped_instruction = 0

    with input_path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue

            row = json.loads(line)
            user_content, matched = extract_user_content(str(row["prompt"]))
            clean_content, stripped = clean_prompt(user_content)

            matched_chat += int(matched)
            stripped_instruction += int(stripped)

            messages = []
            if system_prompt:
                messages.append({"content": system_prompt, "role": "system"})
            messages.append({"content": clean_content, "role": "user"})

            records.append(
                {
                    "data_source": "math",
                    "prompt": messages,
                    "difficulty": "unknown",
                    "ability": "math",
                    "reward_model": {"ground_truth": first_ground_truth(row), "style": "rule"},
                    "extra_info": {"index": index, "split": "train"},
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=SCHEMA)
    pq.write_table(table, output_path)

    return {
        "rows": len(records),
        "matched_chat": matched_chat,
        "stripped_instruction": stripped_instruction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/storage/openpsi/users/zzy/zeta/openr1/OpenR1-raw-qwen.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=REPO_ROOT / "rl_data_aug" / "openr1_qwen_no_boxed.parquet",
        type=Path,
    )
    parser.add_argument("--system-prompt", default=None)
    args = parser.parse_args()

    stats = convert(args.input, args.output, args.system_prompt)
    print(f"wrote {args.output}")
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
