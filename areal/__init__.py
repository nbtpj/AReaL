# SPDX-License-Identifier: Apache-2.0

"""AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"""

from .version import __version__  # noqa

# Names exposed under ``areal.*`` but lazy-loaded so that simply doing
# ``import areal`` (e.g., from the CLI) does not pull in the full infra /
# trainer stack (torch, aiohttp, sglang, ...).
_LAZY_INFRA = {
    "RolloutController",
    "StalenessManager",
    "TrainController",
    "WorkflowExecutor",
    "current_platform",
    "workflow_context",
}

_LAZY_TRAINERS = {"DPOTrainer", "PPOTrainer", "RWTrainer", "SFTTrainer"}


def __getattr__(name: str):
    if name in _LAZY_INFRA:
        from . import infra

        val = getattr(infra, name)
        globals()[name] = val
        return val
    if name in _LAZY_TRAINERS:
        # Eager-load `infra` first. This forces the import order that the old
        # top-level `from .infra import ...` used to guarantee, and sidesteps
        # a circular import between `areal.api.engine_api` and
        # `areal.api.alloc_mode` that surfaces when the trainer stack pulls
        # `areal.api` before `areal.infra` is on the import stack.
        from . import infra  # noqa: F401

        from .trainer import DPOTrainer, PPOTrainer, RWTrainer, SFTTrainer

        _map = {
            "DPOTrainer": DPOTrainer,
            "PPOTrainer": PPOTrainer,
            "RWTrainer": RWTrainer,
            "SFTTrainer": SFTTrainer,
        }
        globals().update(_map)
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DPOTrainer",
    "PPOTrainer",
    "RolloutController",
    "RWTrainer",
    "SFTTrainer",
    "StalenessManager",
    "TrainController",
    "WorkflowExecutor",
    "current_platform",
    "workflow_context",
]
