# SPDX-License-Identifier: Apache-2.0

"""``areal inf run`` — launch the inference service (detached).

Spawns a gateway and a router as detached subprocesses, polls the gateway's
``/health`` until it answers, optionally registers a model inline (external
API or local sglang/vllm), persists ``ServiceState`` (and ``ModelState`` if
a model was registered) under ``~/.areal/inf/``, and exits.

There is **no supervisor process**.  Subsequent commands (``stop``,
``status``, ``ps``, ``logs``) reconcile via ``pid_alive`` + gateway
``/health``.
"""

from __future__ import annotations

import os
import shlex
import time
from pathlib import Path

import click

from areal.experimental.cli.commands.inf import inf
from areal.utils.logging import getLogger

logger = getLogger("InfCli")


@inf.command(name="run", help="Launch the inference service (detached).")
# Service flags
@click.option("--service", default="default", help="Service instance name.")
@click.option("--gateway-host", default="127.0.0.1")
@click.option("--gateway-port", type=int, default=8080)
@click.option("--router-host", default="127.0.0.1")
@click.option("--router-port", type=int, default=8081)
@click.option("--admin-api-key", default="areal-admin-key")
@click.option(
    "--routing-strategy",
    type=click.Choice(["round_robin", "least_busy"]),
    default="round_robin",
)
@click.option("--poll-interval", type=float, default=5.0)
@click.option("--router-timeout", type=float, default=2.0)
@click.option("--forward-timeout", type=float, default=120.0)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="info",
)
@click.option("--launch-timeout", type=float, default=30.0)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional TOML override file merged over ~/.areal/inf/config.toml.",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Stop an existing healthy service with the same name first.",
)
# Inline model registration
@click.option(
    "--model",
    default=None,
    help="Model name to register at startup. Triggers inline registration.",
)
# External-model flags
@click.option(
    "--api-url",
    default=None,
    help="External provider URL.  Presence marks the model as external.",
)
@click.option("--provider-api-key", default=None)
@click.option(
    "--provider-api-key-env",
    default=None,
    help="Name of an environment variable holding the provider API key.",
)
@click.option(
    "--provider-model",
    default=None,
    help="Upstream model name to send to the provider (defaults to --model).",
)
# Internal-model flags
@click.option(
    "--backend",
    default=None,
    help="Backend spec for internal model, e.g. 'sglang', 'sglang:tp=2', "
    "'vllm:tp=2,dp=2'.",
)
@click.option(
    "--model-path",
    default=None,
    help="HuggingFace or local path to weights (internal models).",
)
@click.option(
    "--tokenizer-path",
    default=None,
    help="Tokenizer path for the data-proxy.  Defaults to --model-path.",
)
@click.option(
    "--model-health-timeout",
    type=float,
    default=600.0,
    help="Seconds to wait for an internal model server to become healthy.",
)
@click.option(
    "--engine-args",
    default="",
    help="Extra args forwarded verbatim to the sglang / vllm process. "
    "Shell-style string, e.g. '--mem-fraction-static 0.85'.",
)
@click.option(
    "--proxy-args",
    default="",
    help="Extra args forwarded verbatim to the data-proxy process. "
    "Shell-style string, e.g. '--tool-call-parser qwen'.",
)
def run(**opts) -> None:
    """Click entry point: delegate to _do_run, exit with its return code."""
    raise SystemExit(_do_run(opts) or 0)


# ---- helpers --------------------------------------------------------------


def _resolve_provider_api_key(opts: dict) -> str:
    if opts["provider_api_key"]:
        return opts["provider_api_key"]
    env_name = opts["provider_api_key_env"]
    if env_name:
        v = os.environ.get(env_name)
        if not v:
            raise SystemExit(
                f"--provider-api-key-env={env_name!r} is not set in the environment."
            )
        return v
    raise SystemExit(
        "External model registration requires either --provider-api-key or "
        "--provider-api-key-env."
    )


def _refuse_or_replace(name: str, force: bool) -> None:
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayUnreachable,
    )
    from areal.experimental.cli.commands.inf.launcher import kill_pids
    from areal.experimental.cli.commands.inf.state import (
        ServiceModels,
        ServiceState,
        gateway_alive,
        models_state_path,
        router_alive,
        service_state_path,
    )

    p = service_state_path(name)
    if not p.exists():
        return
    try:
        existing = ServiceState.load(name)
    except (FileNotFoundError, ValueError, TypeError):
        p.unlink()
        return

    pid_says_alive = gateway_alive(existing) or router_alive(existing)
    healthy = False
    if pid_says_alive:
        try:
            GatewayClient(
                existing.gateway_url,
                admin_api_key=existing.admin_api_key,
                timeout=1.0,
            ).health()
            healthy = True
        except GatewayUnreachable:
            healthy = False

    if healthy and not force:
        raise SystemExit(
            f"Service {name!r} is already running "
            f"(gateway pid={existing.gateway_pid}, router pid={existing.router_pid}). "
            f"Use --force to replace it, or `areal inf stop {name}` first."
        )
    if not healthy and pid_says_alive:
        logger.warning(
            "Service %r has live pids (gateway=%d, router=%d) but gateway "
            "is unreachable; treating as stale and reclaiming.",
            name,
            existing.gateway_pid,
            existing.router_pid,
        )
    if healthy or pid_says_alive:
        worker_pids: list[int] = []
        mp = models_state_path(name)
        if mp.exists():
            sm = ServiceModels.load(name)
            for m in sm.list_all():
                worker_pids.extend(m.worker_pids)
        kill_pids(
            [existing.gateway_pid, existing.router_pid, *worker_pids],
            grace_s=10.0,
        )

    existing.remove()
    mp = models_state_path(name)
    if mp.exists():
        mp.unlink()


def _wait_health(client, supervisor_pids: list[int], deadline: float) -> None:
    from areal.experimental.cli.commands.inf.gateway_client import GatewayUnreachable
    from areal.experimental.cli.state import pid_alive

    last_err: Exception | None = None
    while time.time() < deadline:
        if not all(pid_alive(p) for p in supervisor_pids):
            raise SystemExit(
                "Gateway or router subprocess died during startup."
            )
        try:
            client.health()
            return
        except GatewayUnreachable as e:
            last_err = e
            time.sleep(0.5)
    raise SystemExit(
        f"Service did not become healthy within timeout. Last error: {last_err}"
    )


def _register_external_inline(
    *, opts: dict, service_name: str, gateway_url: str
) -> None:
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayHTTPError,
        GatewayUnreachable,
    )
    from areal.experimental.cli.commands.inf.state import ModelState, ServiceModels

    api_key = _resolve_provider_api_key(opts)
    payload = {
        "model": opts["model"],
        "url": opts["api_url"],
        "api_key": api_key,
        "data_proxy_addrs": [],
    }
    if opts["provider_model"]:
        payload["provider_model"] = opts["provider_model"]

    client = GatewayClient(
        gateway_url, admin_api_key=opts["admin_api_key"], timeout=10.0
    )
    try:
        client.register_model(payload)
    except (GatewayUnreachable, GatewayHTTPError) as e:
        raise SystemExit(
            f"Inline register of model {opts['model']!r} failed: {e}"
        ) from e

    models = ServiceModels.load(service_name)
    models.add(
        ModelState(
            name=opts["model"],
            kind="external",
            api_url=opts["api_url"],
            provider_model=opts["provider_model"] or opts["model"],
            registered_at=time.time(),
        )
    )
    models.save()


def _register_internal_inline(
    *, opts: dict, service_name: str,
    gateway_url: str, router_url: str, log_dir: Path,
) -> None:
    from areal.experimental.cli.commands.inf.register_helper import (
        InternalRegisterArgs,
        register_internal_model,
    )
    from areal.experimental.cli.commands.inf.state import ModelState, ServiceModels

    if not opts["model_path"]:
        raise SystemExit(
            "--model-path is required for internal model registration."
        )

    result = register_internal_model(
        InternalRegisterArgs(
            model_name=opts["model"],
            backend_spec=opts["backend"],
            model_path=opts["model_path"],
            tokenizer_path=opts["tokenizer_path"] or opts["model_path"],
            log_dir=log_dir,
            admin_api_key=opts["admin_api_key"],
            log_level=opts["log_level"],
            health_timeout=opts["model_health_timeout"],
            engine_extra_args=shlex.split(opts["engine_args"]) if opts["engine_args"] else [],
            proxy_extra_args=shlex.split(opts["proxy_args"]) if opts["proxy_args"] else [],
        ),
        gateway_url=gateway_url,
        router_url=router_url,
    )

    models = ServiceModels.load(service_name)
    models.add(
        ModelState(
            name=opts["model"],
            kind="internal",
            backend_spec=opts["backend"],
            data_proxy_addrs=result.data_proxy_addrs,
            inference_server_addrs=result.inference_server_addrs,
            worker_pids=result.worker_pids,
            registered_at=time.time(),
        )
    )
    models.save()


# ---- main entry ----------------------------------------------------------


def _do_run(opts: dict) -> int:
    # Sanity-check model flags up front.
    if opts["api_url"] and not opts["model"]:
        raise SystemExit("--api-url requires --model.")
    if opts["backend"] and not opts["model"]:
        raise SystemExit("--backend requires --model.")
    if opts["model"] and opts["api_url"] and opts["backend"]:
        raise SystemExit(
            "Specify either --api-url (external) OR --backend (internal), not both."
        )
    if opts["model"] and not (opts["api_url"] or opts["backend"]):
        raise SystemExit(
            "--model requires either --api-url <url> (external) or "
            "--backend <spec> --model-path <path> (internal)."
        )

    from areal.experimental.cli.commands.inf.gateway_client import GatewayClient
    from areal.experimental.cli.commands.inf.launcher import (
        kill_pids,
        spawn_gateway,
        spawn_router,
    )
    from areal.experimental.cli.commands.inf.state import (
        ServiceState,
        get_current_service,
        service_logs_dir,
        set_current_service,
    )

    service = opts["service"]
    _refuse_or_replace(service, force=opts["force"])

    logs = service_logs_dir(service)
    logger.info("Starting service %r (logs: %s)", service, logs)

    router_pid = spawn_router(
        host=opts["router_host"],
        port=opts["router_port"],
        admin_api_key=opts["admin_api_key"],
        poll_interval=opts["poll_interval"],
        routing_strategy=opts["routing_strategy"],
        log_level=opts["log_level"],
        log_file=logs / "router.log",
    )
    logger.info("Spawned router pid=%d", router_pid)

    time.sleep(0.3)

    gateway_pid = spawn_gateway(
        host=opts["gateway_host"],
        port=opts["gateway_port"],
        admin_api_key=opts["admin_api_key"],
        router_host=opts["router_host"],
        router_port=opts["router_port"],
        router_timeout=opts["router_timeout"],
        forward_timeout=opts["forward_timeout"],
        log_level=opts["log_level"],
        log_file=logs / "gateway.log",
    )
    logger.info("Spawned gateway pid=%d", gateway_pid)

    state = ServiceState(
        name=service,
        gateway_host=opts["gateway_host"],
        gateway_port=opts["gateway_port"],
        router_host=opts["router_host"],
        router_port=opts["router_port"],
        gateway_pid=gateway_pid,
        router_pid=router_pid,
        admin_api_key=opts["admin_api_key"],
        routing_strategy=opts["routing_strategy"],
        log_level=opts["log_level"],
        created_at=time.time(),
    )

    client = GatewayClient(
        state.gateway_url, admin_api_key=opts["admin_api_key"], timeout=2.0
    )
    try:
        _wait_health(
            client,
            [router_pid, gateway_pid],
            deadline=time.time() + opts["launch_timeout"],
        )
    except SystemExit:
        kill_pids([gateway_pid, router_pid], grace_s=5.0)
        raise

    state.save()

    if opts["model"]:
        try:
            if opts["api_url"]:
                _register_external_inline(
                    opts=opts, service_name=service, gateway_url=state.gateway_url
                )
            else:
                _register_internal_inline(
                    opts=opts, service_name=service,
                    gateway_url=state.gateway_url,
                    router_url=state.router_url,
                    log_dir=logs,
                )
        except SystemExit:
            kill_pids([gateway_pid, router_pid], grace_s=5.0)
            state.remove()
            raise

    if get_current_service() is None:
        set_current_service(service)

    logger.info("Service %r ready.", service)
    logger.info("  gateway: %s", state.gateway_url)
    logger.info("  router:  %s", state.router_url)
    logger.info("  pids:    gateway=%d, router=%d", gateway_pid, router_pid)
    if opts["model"]:
        kind = "external" if opts["api_url"] else f"internal ({opts['backend']})"
        logger.info("  default model: %s (%s)", opts["model"], kind)
    logger.info("  log dir: %s", logs)
    return 0
