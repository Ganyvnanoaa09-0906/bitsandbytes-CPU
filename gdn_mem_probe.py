"""Multi-layer GDN memory probe: the real training pattern.

L GDN layers run their forward (all checkpoints stay alive, exactly like a
backward through a deep model), then backward runs layer by layer. Reports
RSS after the forward stack (the number that decides swap death on a
12-16 GB box) for auto chunk vs C=64, at T=4096/8192.

Run each config in a fresh process: python gdn_mem_probe.py
"""
import importlib.util
import os
import sys


def rss_mb():
    if os.name == "nt":  # Windows: working set via psapi (no /proc)
        try:
            import ctypes
            import ctypes.wintypes as wt

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t)] + \
                           [(n, ctypes.c_size_t) for n in (
                               "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                               "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                               "PagefileUsage", "PeakPagefileUsage")]

            pmc = _PMC(cb=ctypes.sizeof(_PMC))
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return float(pmc.WorkingSetSize) / (1 << 20)
        except Exception:
            pass
        return -1.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return -1.0


def load_gdn():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "gdn_cpu", os.path.join(here, "bitsandbytes", "bitsandbytes", "gdn_cpu.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    import torch
    chunk = os.environ.get("GDN_CPU_CHUNK")  # unset -> auto
    T = int(os.environ.get("PROBE_T", "8192"))
    L = int(os.environ.get("PROBE_L", "8"))
    B, H, K, V = 1, 16, 128, 128

    gc = load_gdn()
    fused = gc.fused_recurrent_gated_delta_rule
    torch.manual_seed(0)

    q = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_(True)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_(True)
    v = (torch.randn(B, T, H, V) * 0.5).requires_grad_(True)
    base = rss_mb()

    # forward stack: outputs kept alive by the autograd graph, checkpoints
    # kept alive by each layer's ctx - the training worst case
    x = q
    outs = []
    for _ in range(L):
        beta = (torch.rand(B, T, H) * 0.9 + 0.05).requires_grad_(True)
        g = (-torch.rand(B, T, H) * 0.1).requires_grad_(True)
        o = fused(x, k, v, beta, g)[0]
        x = torch.nn.functional.normalize(o, dim=-1)  # cheap stand-in mixing
        outs.append((o, beta, g))
    r_fwd = rss_mb()

    (x.square().mean()).backward()
    r_bwd = rss_mb()

    _C = gc._chunk_len(B, H, T, K, V)
    ck_mb = B * H * ((T + _C - 1) // _C) * K * V * 4 / 1e6
    tag = f"C={chunk}" if chunk else "auto"
    print(f"T={T} L={L} {tag:>7}: after-fwd +{r_fwd-base:7.1f} MB  after-bwd +{r_bwd-base:7.1f} MB"
          f"   ckpt/layer {ck_mb:6.1f} MB x{L} = {ck_mb*L:7.1f} MB")


if __name__ == "__main__":
    main()
