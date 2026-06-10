# SPDX-License-Identifier: Apache-2.0

"""``areal inf ps`` — list locally tracked services."""

from __future__ import annotations

import json
import time

import click

from areal.experimental.cli.commands.inf import inf
from areal.utils.logging import getLogger

logger = getLogger("InfCli")


@inf.command(name="ps", help="List locally tracked services.")
@click.option(
    "--all", "show_all", is_flag=True, help="Include stale/dead services."
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def ps(show_all: bool, as_json: bool) -> None:
    raise SystemExit(_do_ps(show_all, as_json) or 0)


def _do_ps(show_all: bool, as_json: bool) -> int:
    from areal.experimental.cli.commands.inf.state import (
        ServiceModels,
        ServiceState,
        gateway_alive,
        get_current_service,
        router_alive,
        services_dir,
    )

    current = get_current_service()
    entries: list[dict] = []
    now = time.time()

    for f in sorted(services_dir().glob("*.json")):
        try:
            name = f.stem
            state = ServiceState.load(name)
        except (FileNotFoundError, ValueError, TypeError, KeyError):
            continue

        alive = gateway_alive(state) or router_alive(state)
        if not alive and not show_all:
            continue

        sm = ServiceModels.load(name)
        entries.append(
            {
                "name": name,
                "current": name == current,
                "state": "running" if alive else "dead",
                "gateway": state.gateway_url,
                "router": state.router_url,
                "models": len(sm.models),
                "default_model": sm.default_model,
                "age_s": int(max(0, now - state.created_at)),
            }
        )

    if as_json:
        print(json.dumps(entries, indent=2))
        return 0

    if not entries:
        msg = "No services."
        if not show_all:
            msg += "  (Add --all to include dead ones.)"
        logger.info("%s", msg)
        return 0

    cols = ("CURRENT", "NAME", "STATE", "GATEWAY", "ROUTER", "MODELS", "AGE")
    rows = [
        (
            "*" if e["current"] else "",
            e["name"],
            e["state"],
            e["gateway"],
            e["router"],
            f"{e['models']}"
            + (f" (default={e['default_model']})" if e["default_model"] else ""),
            f"{e['age_s']}s",
        )
        for e in entries
    ]
    widths = [max(len(r[i]) for r in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    for r in rows:
        print(fmt.format(*r))
    return 0
