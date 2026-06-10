# SPDX-License-Identifier: Apache-2.0

"""Top-level entry point for the ``areal`` console-script.

This baseline only wires the ``inf`` namespace.  Other top-level
surfaces (``train``, ``agent``, ``weight-update``) will land in
follow-up PRs reusing the same click group registration shape.

Heavy imports (torch / sglang / vllm / fastapi / ...) must stay out
of this module and out of any ``@click.command`` callback's import
preamble.  Each verb's body is the only place that may pull in those
packages.
"""

from __future__ import annotations

import click

from areal.experimental.cli.commands.inf import inf
from areal.version import __version__


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="AReaL operator CLI for the v2 microservice architecture.",
)
@click.version_option(__version__, prog_name="areal")
def cli() -> None:
    pass


cli.add_command(inf)


if __name__ == "__main__":
    cli()
