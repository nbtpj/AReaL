# SPDX-License-Identifier: Apache-2.0

"""``areal inf status`` — health for one service and its components.

Composed from local state + gateway HTTP — no aggregated status endpoint
on the gateway.
"""

from __future__ import annotations

import json
import sys
import time

import click

from areal.experimental.cli.commands.inf import inf
from areal.utils.logging import getLogger

logger = getLogger("InfCli")


@inf.command(name="status", help="Show service / component health.")
@click.argument("name", required=False)
@click.option("--watch", is_flag=True, help="Refresh until interrupted.")
@click.option("--interval", type=float, default=2.0)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def status(name: str | None, watch: bool, interval: float, as_json: bool) -> None:
    raise SystemExit(_do_status(name, watch, interval, as_json) or 0)


def _collect(name: str) -> dict:
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayUnreachable,
    )
    from areal.experimental.cli.commands.inf.state import (
        ServiceModels,
        ServiceState,
        gateway_alive,
        router_alive,
    )

    state = ServiceState.load(name)
    rows: list[dict] = []

    g_alive = gateway_alive(state)
    r_alive = router_alive(state)

    client = GatewayClient(
        state.gateway_url, admin_api_key=state.admin_api_key, timeout=2.0
    )
    gateway_status = "down"
    gateway_models_count = 0
    if g_alive:
        try:
            client.health()
            gateway_status = "ok"
            try:
                gw_models = client.list_models()
                if isinstance(gw_models, dict):
                    items = gw_models.get("data") or gw_models.get("models") or []
                    gateway_models_count = len(items) if isinstance(items, list) else 0
                elif isinstance(gw_models, list):
                    gateway_models_count = len(gw_models)
            except GatewayUnreachable:
                pass
        except GatewayUnreachable:
            gateway_status = "unreachable"

    router_status = "ok" if r_alive else "down"

    rows.append(
        {
            "service": name,
            "component": "gateway",
            "status": gateway_status,
            "addr": f"{state.gateway_host}:{state.gateway_port}",
            "details": f"models={gateway_models_count}",
        }
    )
    rows.append(
        {
            "service": name,
            "component": "router",
            "status": router_status,
            "addr": f"{state.router_host}:{state.router_port}",
            "details": "",
        }
    )

    sm = ServiceModels.load(name)
    for m in sm.list_all():
        details_parts = [f"kind={m.kind}"]
        if m.kind == "internal" and m.backend_spec:
            details_parts.append(f"backend={m.backend_spec}")
        if m.kind == "external" and m.api_url:
            details_parts.append(f"upstream={m.api_url}")
        if sm.default_model == m.name:
            details_parts.append("default")
        rows.append(
            {
                "service": name,
                "component": m.name,
                "status": "registered",
                "addr": "internal" if m.kind == "internal" else "external",
                "details": " ".join(details_parts),
            }
        )

    return {
        "service": name,
        "rows": rows,
        "default_model": sm.default_model,
    }


def _print_table(snap: dict) -> None:
    cols = ("SERVICE", "COMPONENT", "STATUS", "ADDR", "DETAILS")
    rows = [
        (r["service"], r["component"], r["status"], r["addr"], r["details"])
        for r in snap["rows"]
    ]
    if not rows:
        return
    widths = [max(len(r[i]) for r in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    for r in rows:
        print(fmt.format(*r))


def _do_status(
    name_arg: str | None, watch: bool, interval: float, as_json: bool
) -> int:
    from areal.experimental.cli.commands.inf.state import resolve_service

    name = resolve_service(name_arg)
    try:
        if not watch:
            snap = _collect(name)
            if as_json:
                print(json.dumps(snap, indent=2))
            else:
                _print_table(snap)
            return 0
        while True:
            snap = _collect(name)
            if as_json:
                print(json.dumps(snap, indent=2))
            else:
                sys.stdout.write("\033[2J\033[H")
                _print_table(snap)
                sys.stdout.flush()
            time.sleep(interval)
    except FileNotFoundError:
        logger.error("No service named %r.", name)
        return 1
    except KeyboardInterrupt:
        return 0
