# SPDX-License-Identifier: Apache-2.0

"""Core components for AREAL.

The submodule imports here used to be *eager* (``from .controller import
RolloutController, ...``). That made any ``from areal.infra.platforms
import current_platform`` — done by ``areal.utils.timeutil`` and
``areal.engine.fsdp_utils.optimizer`` — execute this ``__init__.py`` first,
which then pulled ``controller`` → ``areal.api`` → ``cli_args`` and
deadlocked whenever ``cli_args`` itself was mid-import (e.g. from the
SGLang inference worker subprocess).

PEP 562 ``__getattr__`` defers each load until first access, so submodule
imports like ``from areal.infra.platforms import X`` no longer drag the
controller stack in.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# name -> submodule path containing it
_LAZY_IMPORTS: dict[str, str] = {
    "RolloutController": "areal.infra.controller",
    "TrainController": "areal.infra.controller",
    "LocalLauncher": "areal.infra.launcher",
    "RayLauncher": "areal.infra.launcher",
    "SlurmLauncher": "areal.infra.launcher",
    "SGLangServerWrapper": "areal.infra.launcher",
    "vLLMServerWrapper": "areal.infra.launcher",
    "Platform": "areal.infra.platforms",
    "current_platform": "areal.infra.platforms",
    "is_npu_available": "areal.infra.platforms",
    "RemoteInfBackendProtocol": "areal.infra.remote_inf_engine",
    "RemoteInfEngine": "areal.infra.remote_inf_engine",
    "LocalScheduler": "areal.infra.scheduler",
    "RayScheduler": "areal.infra.scheduler",
    "SlurmScheduler": "areal.infra.scheduler",
    "StalenessManager": "areal.infra.staleness_manager",
    "WorkflowExecutor": "areal.infra.workflow_executor",
    "check_trajectory_format": "areal.infra.workflow_executor",
}

# Submodules exposed as attributes (e.g. ``from areal.infra import workflow_context``)
_LAZY_SUBMODULES: frozenset[str] = frozenset({"workflow_context"})


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(mod, name)
        globals()[name] = val
        return val
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"areal.infra.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'areal.infra' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS) | _LAZY_SUBMODULES)


if TYPE_CHECKING:
    # Help static analyzers/IDE completion without triggering runtime imports.
    from . import workflow_context  # noqa: F401
    from .controller import RolloutController, TrainController  # noqa: F401
    from .launcher import (  # noqa: F401
        LocalLauncher,
        RayLauncher,
        SGLangServerWrapper,
        SlurmLauncher,
        vLLMServerWrapper,
    )
    from .platforms import (  # noqa: F401
        Platform,
        current_platform,
        is_npu_available,
    )
    from .remote_inf_engine import (  # noqa: F401
        RemoteInfBackendProtocol,
        RemoteInfEngine,
    )
    from .scheduler import (  # noqa: F401
        LocalScheduler,
        RayScheduler,
        SlurmScheduler,
    )
    from .staleness_manager import StalenessManager  # noqa: F401
    from .workflow_executor import (  # noqa: F401
        WorkflowExecutor,
        check_trajectory_format,
    )


__all__ = [
    "LocalLauncher",
    "LocalScheduler",
    "Platform",
    "RayLauncher",
    "RayScheduler",
    "RemoteInfBackendProtocol",
    "RemoteInfEngine",
    "RolloutController",
    "SGLangServerWrapper",
    "SlurmLauncher",
    "SlurmScheduler",
    "StalenessManager",
    "TrainController",
    "WorkflowExecutor",
    "check_trajectory_format",
    "current_platform",
    "is_npu_available",
    "vLLMServerWrapper",
    "workflow_context",
]
