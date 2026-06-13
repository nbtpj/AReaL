#!/usr/bin/env python3
"""Keep vanilla.md in sync with validation metrics during the long run."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS = Path(os.environ.get("METRICS_PATH", ROOT / "rl_data_aug" / "vanilla_polaris_qwen3_1_7b.jsonl"))
OUT = Path(os.environ.get("OUT_PATH", ROOT / "vanilla.md"))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", ROOT / "ckpts" / "vanilla"))
TITLE = os.environ.get("RUN_TITLE", "Vanilla Polaris Qwen3-1.7B")
ACC_KEY = "val-core/math_dapo/acc/mean@1"
REWARD_KEY = "val-aux/math_dapo/reward/mean@1"
SCORE_KEY = "val-aux/math_dapo/score/mean@1"
TARGET_STEP = 100


def read_rows() -> tuple[list[dict[str, float]], int]:
    rows: list[dict[str, float]] = []
    max_step = -1
    if not METRICS.exists():
        return rows, max_step

    for line in METRICS.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step = int(record["step"])
        max_step = max(max_step, step)
        data = record.get("data", {})
        if ACC_KEY not in data:
            continue
        rows.append(
            {
                "step": step,
                "accuracy": float(data[ACC_KEY]),
                "reward": float(data.get(REWARD_KEY, data.get(SCORE_KEY, 0.0))),
            }
        )
    return rows, max_step


def checkpoint_steps() -> list[int]:
    if not CKPT_DIR.exists():
        return []
    steps = []
    for path in CKPT_DIR.glob("global_step_*"):
        try:
            steps.append(int(path.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(steps)


def render(rows: list[dict[str, float]], max_step: int) -> str:
    status = "Complete" if max_step >= TARGET_STEP else "Run in progress"
    ckpts = ", ".join(str(step) for step in checkpoint_steps()) or "none yet"
    lines = [
        f"# {TITLE}",
        "",
        f"Status: {status}.",
        f"Last metric step: {max_step if max_step >= 0 else 'none'}.",
        f"Checkpoint steps present: {ckpts}.",
        "",
        "| Step | AIME24 accuracy | Mean reward |",
        "| ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['step']} | {row['accuracy']:.6g} | {row['reward']:.6g} |")
    lines.append("")
    return "\n".join(lines)


def update_once() -> int:
    rows, max_step = read_rows()
    OUT.write_text(render(rows, max_step))
    return max_step


def main() -> None:
    while True:
        max_step = update_once()
        if max_step >= TARGET_STEP:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
