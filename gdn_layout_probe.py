"""Layout-1 zero-copy + auto-chunk validation.

1. bitwise: layout 1 ([B,T,H,D] contiguous, zero-copy) vs layout 0 (strided
   inputs -> per-head copies) must produce identical outputs AND grads
2. bitwise: results are invariant to the checkpoint chunk length C
3. memory: peak RSS of one fwd+bwd at T=8192 with contiguous inputs
   (zero-copy path + auto chunk) vs pinned C=64 (old default)

Run: PYTHONPATH=/workspace/bitsandbytes python gdn_layout_probe.py
"""
import os
import sys
import time

import torch

import bitsandbytes.gdn_cpu as gc
from bitsandbytes.gdn_cpu import fused_recurrent_gated_delta_rule

_HERE = os.path.dirname(os.path.abspath(__file__))
_GDN_PY = os.path.join(_HERE, "bitsandbytes", "bitsandbytes", "gdn_cpu.py")


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


def make(B=1, T=2048, H=16, K=128, V=128, seed=0, strided=False):
    torch.manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1)
    v = torch.randn(B, T, H, V) * 0.5
    beta = torch.rand(B, T, H) * 0.9 + 0.05
    g = -torch.rand(B, T, H) * 0.1
    if strided:  # force the per-head-copy path (layout 0)
        qb = torch.zeros(B, H, T * 2, K); qb[:, :, ::2] = q.permute(0, 2, 1, 3)
        kb = torch.zeros(B, H, T * 2, K); kb[:, :, ::2] = k.permute(0, 2, 1, 3)
        vb = torch.zeros(B, H, T * 2, V); vb[:, :, ::2] = v.permute(0, 2, 1, 3)
        q, k, v = (t[:, :, ::2].permute(0, 2, 1, 3) for t in (qb, kb, vb))
        assert not q.is_contiguous() and not k.is_contiguous()
    return q, k, v, beta, g


def run(strided=False, chunk_env=None):
    if chunk_env is None:
        os.environ.pop("GDN_CPU_CHUNK", None)
    else:
        os.environ["GDN_CPU_CHUNK"] = str(chunk_env)
    q, k, v, beta, g = make(strided=strided)
    for t in (q, k, v, beta, g):
        t.requires_grad_(True)
    t0 = time.perf_counter()
    o, sf = fused_recurrent_gated_delta_rule(q, k, v, beta, g, output_final_state=True)
    loss = o.square().mean() + sf.square().mean()
    loss.backward()
    dt = time.perf_counter() - t0
    C = gc._chunk_len(1, 16, 2048, 128, 128)
    return o.detach(), sf.detach(), q.grad, k.grad, v.grad, beta.grad, g.grad, dt, C


fails = 0

# --- 1: layout 1 (zero-copy) vs layout 0 (copies), same values -------------
o1, s1, *g1, _, _, C1 = run(strided=False, chunk_env=64)
o0, s0, *g0, _, _, C0 = run(strided=True, chunk_env=64)
same = all(torch.equal(a, b) for a, b in zip((o1, s1) + tuple(g1), (o0, s0) + tuple(g0)))
print(f"layout1 vs layout0 bitwise: {'OK' if same else 'MISMATCH'}  (C={C1}/{C0})")
fails += not same

# --- 2: chunk-length invariance (auto vs 64 vs 512) -------------------------
oa, sa, *ga, _, dta, Ca = run(strided=False, chunk_env=None)
o5, s5, *g5, _, dt5, C5 = run(strided=False, chunk_env=512)
same_a = all(torch.equal(a, b) for a, b in zip((oa, sa) + tuple(ga), (o1, s1) + tuple(g1)))
same_5 = all(torch.equal(a, b) for a, b in zip((o5, s5) + tuple(g5), (o1, s1) + tuple(g1)))
print(f"C-invariance: auto C={Ca} vs 64 {'OK' if same_a else 'MISMATCH'}, "
      f"C=512 vs 64 {'OK' if same_5 else 'MISMATCH'}  ({dta*1e3:.0f}ms / {dt5*1e3:.0f}ms)")
fails += not (same_a and same_5)

# --- 3: peak RSS, T=8192, contiguous inputs (isolated subprocess each) ------
B, T, H, K, V = 1, 8192, 16, 128, 128
for label, chunk in (("auto", None), ("C=64 (old default)", 64)):
    import subprocess
    code = (
        "import os,importlib.util,time,torch\n"
        + (f"os.environ['GDN_CPU_CHUNK'] = {str(chunk)!r}\n" if chunk else "")
        + "# load gdn_cpu straight from its file: skips the full bnb package\n"
        "# import whose allocator churn makes RSS deltas meaningless\n"
        f"spec=importlib.util.spec_from_file_location('gdn_cpu',{_GDN_PY!r})\n"
        "gc=importlib.util.module_from_spec(spec); spec.loader.exec_module(gc)\n"
        "fused=gc.fused_recurrent_gated_delta_rule\n"
        "def rss():\n"
        "    if os.name=='nt':\n"
        "        import ctypes\n"
        "        import ctypes.wintypes as wt\n"
        "        class P(ctypes.Structure):\n"
        "            _fields_=[('cb',wt.DWORD),('PageFaultCount',wt.DWORD),"
        "('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t)]\n"
        "        pm=P(cb=ctypes.sizeof(P))\n"
        "        h=ctypes.windll.kernel32.GetCurrentProcess()\n"
        "        ctypes.windll.psapi.GetProcessMemoryInfo(h,ctypes.byref(pm),pm.cb)\n"
        "        return pm.WorkingSetSize/1048576\n"
        "    for l in open('/proc/self/status'):\n"
        "        if l.startswith('VmRSS'): return int(l.split()[1])/1024\n"
        "torch.manual_seed(0)\n"
        f"B,T,H,K,V={B},{T},{H},{K},{V}\n"
        "q=torch.nn.functional.normalize(torch.randn(B,T,H,K),dim=-1).requires_grad_(True)\n"
        "k=torch.nn.functional.normalize(torch.randn(B,T,H,K),dim=-1).requires_grad_(True)\n"
        "v=(torch.randn(B,T,H,V)*0.5).requires_grad_(True)\n"
        "beta=(torch.rand(B,T,H)*0.9+0.05).requires_grad_(True)\n"
        "g=(-torch.rand(B,T,H)*0.1).requires_grad_(True)\n"
        "base=rss(); t0=time.perf_counter()\n"
        "o=fused(q,k,v,beta,g)[0]; r1=rss()\n"
        "o.square().mean().backward(); r2=rss()\n"
        "dt=time.perf_counter()-t0\n"
        "print(f'{max(r1,r2)-base:.1f} {dt*1e3:.0f} {gc._chunk_len(B,H,T,K,V)} {base:.0f} {max(r1,r2):.0f}')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={k: v for k, v in os.environ.items() if k != "GDN_CPU_CHUNK"}
        | {"PYTHONPATH": os.path.join(_HERE, "bitsandbytes")},
    )
    if out.returncode != 0:
        print(f"{label}: child failed:\n{out.stderr[-500:]}")
        raise SystemExit(1)
    parts = out.stdout.strip().split()
    d_rss, dt_ms, c = float(parts[0]), float(parts[1]), int(parts[2])
    print(f"peak RSS {label:>18}: +{d_rss:7.1f} MB   fwd+bwd {dt_ms:7.0f} ms   C={c}   "
          f"(base {parts[3]} MB -> peak {parts[4]} MB)")

print("LAYOUT PROBE", "PASSED" if fails == 0 else f"FAILED ({fails})")
raise SystemExit(1 if fails else 0)
