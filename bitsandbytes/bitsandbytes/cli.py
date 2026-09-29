"""Compatibility shim. The reference itself lives in `bitsandbytes_cpu_cli`.

Why it is not here: `bitsandbytes/__init__.py` imports torch, and torch is deliberately
not a dependency of this distribution. So this module can only be imported on a machine
that already has torch -- which is not the machine whose user needs the help. The console
script therefore points at the top-level `bitsandbytes_cpu_cli`, which uses nothing but
the standard library.

This file is kept so that `python -m bitsandbytes.cli help` and code that imports the
tables from here keep working once torch is installed.
"""
from __future__ import annotations

import sys

from bitsandbytes_cpu_cli import (  # noqa: F401
    EXAMPLES,
    PY,
    SECTIONS,
    SH,
    USAGE,
    WIDTH,
    _row,
    _select,
    _wrap,
    cmd_detect,
    cmd_doctor,
    cmd_help,
    cmd_selftest,
    cmd_version,
    main,
    render_markdown,
    render_text,
)

__all__ = [
    "EXAMPLES",
    "PY",
    "SECTIONS",
    "SH",
    "USAGE",
    "WIDTH",
    "main",
    "render_markdown",
    "render_text",
]


if __name__ == "__main__":
    sys.exit(main())
