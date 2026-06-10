# SPDX-License-Identifier: Apache-2.0

"""``areal inf logs`` — tail gateway / router / model logs.

Logs live under ``~/.areal/inf/logs/<service>/<component>.log``.
"""

from __future__ import annotations

import os

import click

from areal.experimental.cli.commands.inf import inf
from areal.utils.logging import getLogger

logger = getLogger("InfCli")


@inf.command(name="logs", help="Tail gateway / router / model logs.")
@click.argument("name", required=False)
@click.option(
    "--component",
    default="gateway",
    help="One of `gateway`, `router`, or a model name. "
    "Becomes `<component>.log` under the service log dir.",
)
@click.option("-f", "--follow", is_flag=True, help="Stream new lines.")
@click.option("-n", "--lines", type=int, default=200)
def logs(name: str | None, component: str, follow: bool, lines: int) -> None:
    raise SystemExit(_do_logs(name, component, follow, lines) or 0)


def _do_logs(
    name_arg: str | None, component: str, follow: bool, lines: int
) -> int:
    from areal.experimental.cli.commands.inf.state import (
        resolve_service,
        service_logs_dir,
    )

    name = resolve_service(name_arg)
    log_dir = service_logs_dir(name)
    log_file = log_dir / f"{component}.log"
    if not log_file.exists():
        logger.error("No %s.log at %s.", component, log_file)
        return 1

    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-F")
    cmd.append(str(log_file))
    os.execvp(cmd[0], cmd)
