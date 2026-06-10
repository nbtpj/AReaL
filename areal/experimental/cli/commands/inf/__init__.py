# SPDX-License-Identifier: Apache-2.0

"""``areal inf`` — Ollama-style inference service operator console.

This namespace owns its OWN service lifecycle (gateway + router +
optional model backends).  Services are standalone — they don't need
a training yaml, an experiment_name, or any cluster scheduler
integration.  A user starts a serving stack, registers external or
internal models, and chats / collects against them.

A separate training surface (``areal train ...``) will land in a
later PR; the two namespaces are intentionally decoupled — training
experiments wrap ``RolloutControllerV2`` in-process, while ``inf``
spawns gateway/router/data-proxy as detached subprocesses owned by
nothing but the kernel.

Phase 1 verbs:
  run     Launch gateway + router (detached).  Optional inline model
          registration (external via --api-url or internal via
          --backend / --model-path).
  stop    Tear down gateway + router (and any tracked model backends).
  status  Health for one service + its components and registered models.
  ps      List locally tracked services.
  logs    Tail gateway / router / model logs.

Future phases:
  register / deregister / models  — standalone model lifecycle verbs.
  chat / collect / interactive shell.

State lives under ``~/.areal/inf/``; see ``state.py`` for layout.
"""

from __future__ import annotations

import click


@click.group(help="Manage inference services and models.")
def inf() -> None:
    pass


# Registering verbs by importing their modules — each imports `inf`
# above and attaches itself via ``@inf.command(...)``.  Keep this at
# the bottom so ``inf`` is fully bound by the time the verbs run their
# decorators.
from areal.experimental.cli.commands.inf import (  # noqa: E402,F401
    logs,
    ps,
    run,
    status,
    stop,
)
