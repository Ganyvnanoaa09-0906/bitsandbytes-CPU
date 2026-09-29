"""Compatibility shim — the real module now lives at bitsandbytes/torch_cpu_kit.py.

It moved into the package so that a `pip install` user gets it. While it sat at the
repository root it was reachable only from a checkout, which meant the documentation
could describe it while an installed user could not import it.

Keeping this file means `import torch_cpu_kit` still works for scripts written against
the old layout. New code should use either of these, both of which work installed:

    import bitsandbytes.torch_cpu_kit as tck
    from bitsandbytes import torch_cpu_kit as tck
"""
from bitsandbytes.torch_cpu_kit import *  # noqa: F401,F403
from bitsandbytes.torch_cpu_kit import (  # noqa: F401
    apply_env,
    autocast,
    fast_loader,
    init,
    make_contiguous_,
    mem_report,
    physical_cores,
    recommended_autocast,
    report,
    start_mem_monitor,
    suspend_mem_monitor,
)
