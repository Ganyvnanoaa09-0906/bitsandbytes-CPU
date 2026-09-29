# -*- coding: utf-8 -*-
"""Compatibility shim — the real detector now lives at bitsandbytes/detect_cpu.py.

Run it either way; both work once the package is installed:

    python -m bitsandbytes.detect_cpu
    python detect_cpu.py            # this file, from a checkout

It moved into the package for the same reason torch_cpu_kit did: while it sat at the
repository root a `pip install` user could not reach it, so the documentation was
describing a tool they did not have.
"""
import sys

from bitsandbytes.detect_cpu import detect, main  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
