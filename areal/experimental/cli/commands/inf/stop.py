# SPDX-License-Identifier: Apache-2.0

"""``areal inf stop`` — stop a running inference service.

Sends SIGTERM to the gateway, the router, and any tracked model worker
PIDs (data proxies + inference servers, populated when an internal model
is registered).  After the grace period, escalates to SIGKILL.  State
files are removed unless ``--keep-state`` is passed.
"""

from __future__ import annotations

import time

import click

from areal.experimental.cli.commands.inf import inf
from areal.utils.logging import getLogger

logger = getLogger("InfCli")


@inf.command(name="stop", help="Stop a running inference service.")
@click.argument("name", required=False)
@click.option(
    "--grace-period",
    type=float,
    default=10.0,
    help="Seconds to wait before escalating to SIGKILL.",
)
@click.option(
    "--keep-state",
    is_flag=True,
    help="Keep state files after shutdown (debugging).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Skip confirmations; reserved for future interactive prompts.",
)
def stop(name: str | None, grace_period: float, keep_state: bool, force: bool) -> None:
    raise SystemExit(_do_stop(name, grace_period, keep_state, force) or 0)


def _do_stop(
    name_arg: str | None, grace_period: float, keep_state: bool, force: bool
) -> int:
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayUnreachable,
    )
    from areal.experimental.cli.commands.inf.launcher import kill_pids
    from areal.experimental.cli.commands.inf.state import (
        ServiceModels,
        ServiceState,
        gateway_alive,
        get_current_service,
        models_state_path,
        resolve_service,
        router_alive,
        set_current_service,
    )

    name = resolve_service(name_arg)

    try:
        state = ServiceState.load(name)
    except FileNotFoundError:
        logger.error("No service named %r.", name)
        return 1

    pids: list[int] = [state.gateway_pid, state.router_pid]
    models_path = models_state_path(name)
    if models_path.exists():
        sm = ServiceModels.load(name)
        for m in sm.list_all():
            for pid in m.worker_pids:
                if pid > 0:
                    pids.append(pid)

    alive = gateway_alive(state) or router_alive(state)
    if not alive:
        logger.warning(
            "Service %r is already down (no live gateway/router pid); "
            "cleaning up state.", name,
        )
    else:
        logger.info(
            "Stopping service %r: gateway=%d, router=%d ...",
            name, state.gateway_pid, state.router_pid,
        )
        kill_pids(pids, grace_s=grace_period)

        client = GatewayClient(state.gateway_url, timeout=1.0)
        deadline = time.time() + min(5.0, grace_period)
        while time.time() < deadline:
            try:
                client.health()
                time.sleep(0.3)
            except GatewayUnreachable:
                break

    if not keep_state:
        state.remove()
        if models_path.exists():
            models_path.unlink()
        if get_current_service() == name:
            set_current_service(None)

    logger.info("Service %r stopped.", name)
    return 0
