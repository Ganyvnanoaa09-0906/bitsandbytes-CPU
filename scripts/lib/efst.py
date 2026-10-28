# -*- coding: utf-8 -*-
"""Backward-compatible import location for EFST.

The implementation ships inside the package as ``bitsandbytes.efst`` so that
``pip install bitsandbytes-cpu-fork`` users get it too. This module keeps
``from efst import ...`` working for the scripts in this repository, including when
they run from the repository root before the package is installed.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

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
