# SPDX-License-Identifier: Apache-2.0

"""On-disk state primitives shared across service-style CLIs.

Two layers live here:

1. **Utility functions** — ``areal_home``, ``atomic_write_json``, and the
   namespace-aware path helpers (``services_dir``, ``logs_dir``, etc.).
   Every subcommand CLI reads its on-disk state from
   ``$AREAL_HOME/<namespace>/...``; passing the namespace explicitly lets
   the same helpers serve every CLI from one place.

2. **Contract types** — ``SupportsComponentProbe`` (Protocol) and
   ``ServiceStateBase`` (ABC). Subcommand CLIs implement their own
   ServiceState dataclass that satisfies the protocol / base class, and
   in return get to plug into scaffold's ``ServiceLifecycle`` /
   ``StatusReporter`` / etc. without further glue.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

DEFAULT_SERVICE = "default"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def areal_home() -> Path:
    """Return the AReaL CLI home directory.

    Resolves ``$AREAL_HOME`` if set, otherwise ``~/.areal``. Created on
    first access so callers can mkdir-then-write subdirs without an
    explicit setup step.
    """

    env = os.environ.get("AREAL_HOME")
    root = Path(env).expanduser() if env else Path.home() / ".areal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Atomically write ``data`` as JSON to ``path``.

    Writes to a unique tempfile in ``path``'s directory, fsync()s it to
    disk, then renames into place. ``NamedTemporaryFile(delete=False)``
    gives us a fresh name per call so concurrent writers do not stomp on
    each other's tempfiles, and the tempfile is unlinked on serialization
    or rename failure so half-written state never lingers on disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".tmp"
    ) as f:
        tmp_path = Path(f.name)
        try:
            json.dump(data, f, indent=indent, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Namespace-aware path helpers
# ---------------------------------------------------------------------------


def namespace_root(namespace: str) -> Path:
    d = areal_home() / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d


def services_dir(namespace: str) -> Path:
    d = namespace_root(namespace) / "services"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_root(namespace: str) -> Path:
    d = namespace_root(namespace) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir(namespace: str, service: str) -> Path:
    d = logs_root(namespace) / service
    d.mkdir(parents=True, exist_ok=True)
    return d


def service_state_path(namespace: str, service: str) -> Path:
    return services_dir(namespace) / f"{service}.json"


def service_lock_path(namespace: str, service: str) -> Path:
    return services_dir(namespace) / f"{service}.lock"


def current_service_path(namespace: str) -> Path:
    return namespace_root(namespace) / "current-service"


def config_path(namespace: str) -> Path:
    return namespace_root(namespace) / "config.toml"


def list_service_names(namespace: str) -> list[str]:
    return sorted(p.stem for p in services_dir(namespace).glob("*.json"))


def set_current_service(namespace: str, service: str) -> None:
    current_service_path(namespace).write_text(service + "\n")


def clear_current_service(namespace: str, service: str) -> None:
    path = current_service_path(namespace)
    if path.exists() and path.read_text().strip() == service:
        path.unlink()


def resolve_service_name(
    namespace: str,
    explicit: str | None = None,
    *,
    fallback: str = DEFAULT_SERVICE,
) -> str:
    """Resolve the active service name for a CLI call.

    Order: ``--service`` flag > current-service pointer file > the single
    running service (if exactly one) > ``fallback``.
    """

    if explicit:
        return explicit
    pointer = current_service_path(namespace)
    if pointer.exists():
        value = pointer.read_text().strip()
        if value:
            return value
    running = list_service_names(namespace)
    if len(running) == 1:
        return running[0]
    return fallback


# ---------------------------------------------------------------------------
# Best-effort orphan PID recovery
# ---------------------------------------------------------------------------


def recover_pids_from_raw_state(namespace: str, service: str) -> list[int]:
    """Walk the on-disk state files for ``service`` and pull any
    ``pid`` / ``pids`` numbers.

    Used by ``run --force`` to clean up children when the dataclass
    parse fails (state file from an older / corrupted schema).
    """

    pids: list[int] = []
    pid_keys = {"pid", "pids"}

    def add(value) -> None:
        if isinstance(value, int) and value > 0:
            pids.append(value)
        elif isinstance(value, list):
            for item in value:
                add(item)

    def walk(value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in pid_keys:
                    add(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    candidate_files = [
        service_state_path(namespace, service),
        # Subcommand CLIs may also keep secondary state files (e.g. inf's
        # models/<svc>.json). Walking namespace_root/* once would catch
        # them, but we keep this targeted to the service-state file to
        # avoid surprising file enumeration.
    ]
    for path in candidate_files:
        if not path.exists():
            continue
        with open(path) as f:
            walk(json.load(f))

    seen: set[int] = set()
    unique: list[int] = []
    for pid in pids:
        if pid not in seen:
            seen.add(pid)
            unique.append(pid)
    return unique


# ---------------------------------------------------------------------------
# Contract types
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsComponentProbe(Protocol):
    """Minimum surface a subcommand's component handle must expose so
    scaffold helpers can probe it and identify it.

    Each subcommand CLI's handle type — inf's ``TaskHandle``, agent's
    ``ProcessState``, etc. — already provides ``addr`` (HTTP base for
    ``/health`` probes) and ``pid`` (for local liveness / kill paths).
    No inheritance needed: structural duck typing.
    """

    @property
    def addr(self) -> str: ...

    @property
    def pid(self) -> int: ...


class ServiceStateBase(ABC):
    """Abstract base every subcommand CLI's ServiceState should satisfy.

    The base nails down the universal fields and the two methods
    (``gateway_alive`` / ``components``) that scaffold's lifecycle and
    status reporters rely on. Subclasses are free to add backend-specific
    fields (engine handles, model registries, agent pair configs, etc.).
    """

    service: str
    admin_api_key: str
    launch_mode: str
    started_at: float

    @abstractmethod
    def gateway_alive(self) -> bool:
        """Return True iff the service's central entry point is reachable.

        ``ServiceLifecycle`` uses this to decide "running" — every CLI
        must define what alive means for its own architecture (local PID
        alive / k8s pod ready / slurm job state).
        """

    @abstractmethod
    def components(self) -> Iterable[tuple[str, SupportsComponentProbe]]:
        """Yield ``(label, handle)`` for every component of this service.

        Used by ``StatusReporter`` to enumerate rows; the order is the
        order rows appear in the table. Labels are display-only strings
        (e.g. ``"gateway"``, ``"worker[qwen/0]"``).
        """
