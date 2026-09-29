r"""Compatibility shim — the real module now lives at bitsandbytes/disk_balancer.py.

It moved into the package so that a `pip install` user gets it. From the workspace it was
reachable only because this directory happened to be on sys.path, which is exactly the
kind of thing that works for the author and fails for everyone else.

Both of these work once the package is installed:

    from bitsandbytes.disk_balancer import DiskLoadBalancer, DiskBalancerConfig
    from disk_balancer import DiskLoadBalancer, DiskBalancerConfig   # this file
"""
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
