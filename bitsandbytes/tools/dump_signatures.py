"""Every signature the reference prints, read back out of the library.

`bitsandbytes-cpu help` publishes hand-written signatures in its REFERENCE sections. A
signature that has drifted from the code is worse than no signature: the reader trusts
it. This prints the real ones so the two can be compared in one screen.
"""
import inspect
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bitsandbytes as bnb  # noqa: E402
from bitsandbytes import functional as F  # noqa: E402
import bitsandbytes.torch_cpu_kit as tck  # noqa: E402
from bitsandbytes.detect_cpu import detect  # noqa: E402
from bitsandbytes import gdn_cpu, disk_balancer as db  # noqa: E402


def sig(obj, name=None):
    try:
        return f"{name or obj.__name__}{inspect.signature(obj)}"
    except Exception as exc:  # noqa: BLE001
        return f"{name or obj}: <{type(exc).__name__}: {exc}>"


print("-- layers --")
for cls in (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, bnb.nn.Embedding4bit,
            bnb.nn.Embedding8bit, bnb.nn.Params4bit, bnb.nn.Int8Params):
    print("  " + sig(cls))
print("  " + sig(bnb.nn.OutlierAwareLinear))

print("-- optimizers --")
print("  " + sig(bnb.optim.AdamW8bit))
print("  " + sig(bnb.optim.SGD8bit))
print("  " + sig(bnb.optim.Lion8bit))

print("-- functional --")
for fn in (F.quantize_blockwise, F.dequantize_blockwise, F.quantize_4bit,
           F.dequantize_4bit, F.gemv_4bit, F.int8_linear_matmul,
           F.int8_vectorwise_quant, F.fused_dequant_linear_8bit,
           F.optimizer_update_8bit_blockwise, F.QuantState.from_dict,
           F.create_linear_map, F.create_dynamic_map, F.create_normal_map,
           F.create_fp8_map, F.has_avx512bf16, F.quantize_nf4):
    print("  " + sig(fn))

print("-- torch_cpu_kit --")
for fn in (tck.apply_env, tck.init, tck.physical_cores, tck.mem_report,
           tck.autocast, tck.fast_loader, tck.make_contiguous_,
           tck.start_mem_monitor, tck.suspend_mem_monitor, tck.report,
           tck.recommended_autocast):
    print("  " + sig(fn))

print("-- others --")
print("  " + sig(detect))
print("  " + sig(gdn_cpu.load_native))
print("  " + sig(gdn_cpu.patch_transformers))
print("  " + sig(gdn_cpu.patch_fla))
print("  " + sig(db.add_flash_args))
print("  " + sig(db.parse_flash_args))
print("  " + sig(db.DiskLoadBalancer.__init__, "DiskLoadBalancer"))
for m in ("attach_model", "start", "update_step", "put_cold", "get_cold", "stats"):
    print("    ." + sig(getattr(db.DiskLoadBalancer, m), m))
