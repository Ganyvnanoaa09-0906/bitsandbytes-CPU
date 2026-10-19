# -*- coding: utf-8 -*-
"""Backward-compatible import location for EFST.

The implementation ships inside the package as ``bitsandbytes.efst`` so that
``pip install bitsandbytes-cpu-fork`` users get it too. This module keeps
``from efst import ...`` working for the scripts in this repository, including when
they run from the repository root before the package is installed.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.join(_HERE, "bitsandbytes")
if os.path.isdir(os.path.join(_PKG_ROOT, "bitsandbytes")):
    # Move to the front even when it is already on sys.path (an editable install may have
    # appended it behind a stale installed package, and we want the checked-out sources).
    while _PKG_ROOT in sys.path:
        sys.path.remove(_PKG_ROOT)
    sys.path.insert(0, _PKG_ROOT)

from bitsandbytes.efst import *  # noqa: E402,F401,F403
from bitsandbytes.efst import __all__  # noqa: E402,F401
