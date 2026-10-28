r"""Compatibility shim — the real module now lives at bitsandbytes/disk_balancer.py.

It moved into the package so that a `pip install` user gets it. From the workspace it was
reachable only because this directory happened to be on sys.path, which is exactly the
kind of thing that works for the author and fails for everyone else.

Both of these work once the package is installed:

    from bitsandbytes.disk_balancer import DiskLoadBalancer, DiskBalancerConfig
    from disk_balancer import DiskLoadBalancer, DiskBalancerConfig   # this file
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
from bitsandbytes.disk_balancer import *  # noqa: F401,F403
from bitsandbytes.disk_balancer import (  # noqa: F401
    DiskBalancerConfig,
    DiskInfo,
    DiskLoadBalancer,
    add_flash_args,
    detect_disks,
    get_disk_activity,
    parse_flash_args,
)
