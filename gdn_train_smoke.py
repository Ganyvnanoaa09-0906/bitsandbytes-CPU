"""End-to-end trainability check: GDN layer + AdamW8bit on CPU.

Compares the fused kernel path against the naive per-timestep eager loop
(the original slow-backward pathology) for speed and peak RSS, then trains
a tiny GDN layer to fit a signal with the 8-bit optimizer.
"""
import os
import time

import torch

from bitsandbytes.gdn_cpu import fused_recurrent_gated_delta_rule


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


def naive_gdn(q, k, v, beta, g):
    """The pathological path: per-timestep eager loop, T-deep autograd graph."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    S = torch.zeros(B, H, K, V, dtype=q.dtype, device=q.device)
    outs = []
    for t in range(T):
        S = S * torch.exp(g[:, t]).unsqueeze(-1).unsqueeze(-1)
        u = v[:, t] - torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], beta[:, t].unsqueeze(-1) * u)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
    return torch.stack(outs, 1)


B, H, T, K, V = 2, 4, 1024, 64, 64
torch.manual_seed(0)
# l2-normalized q/k keep the delta-rule recurrence contractive (raw randn k
# explodes over T=1024 -> inf/nan in BOTH the fused and the naive path)
q = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1)
k = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1)
v = torch.randn(B, T, H, V) * 0.5
beta = torch.rand(B, T, H) * 0.9 + 0.05
g = -torch.rand(B, T, H) * 0.1
target = torch.randn(B, T, H, V)

# --- speed + memory: fused vs naive, one fwd+bwd (grad-enabled inputs)
qg, kg, vg, bg, gg, tg = (t.clone().requires_grad_(True) for t in (q, k, v, beta, g, target))
for name, fn in (("fused", lambda: fused_recurrent_gated_delta_rule(qg, kg, vg, bg, gg)[0]),
                 ("naive", lambda: naive_gdn(qg, kg, vg, bg, gg))):
    base = rss_mb()
    t0 = time.perf_counter()
    out = fn()
    loss = ((out - tg) ** 2).mean()
    loss.backward()
    dt = time.perf_counter() - t0
    print(f"{name:6s} fwd+bwd {dt*1000:8.1f} ms   dRSS {rss_mb()-base:+8.1f} MB   loss {loss.item():.4f}")
    del out, loss
    qg.grad = kg.grad = vg.grad = bg.grad = gg.grad = None
    time.sleep(0.2)

# --- trainability: learn a fixed projection inside the GDN layer
import bitsandbytes as bnb  # noqa: E402

w_true = torch.randn(K, K) * 0.2
with torch.no_grad():
    q_t = torch.nn.functional.normalize(q @ w_true, dim=-1)
    target_o = fused_recurrent_gated_delta_rule(q_t, k, v, beta, g)[0]

wq = torch.nn.Parameter(torch.randn(K, K) * 0.2)
opt = bnb.optim.AdamW8bit([wq], lr=1e-2)
losses = []
for step in range(150):
    qh = torch.nn.functional.normalize(q @ wq, dim=-1)
    o = fused_recurrent_gated_delta_rule(qh, k, v, beta, g)[0]
    loss = ((o - target_o) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
print(f"train loss: {losses[0]:.4f} -> {losses[-1]:.4f} (min {min(losses):.4f})")
assert losses[-1] < losses[0] * 0.5, "GDN layer is not training"
print("GDN + AdamW8bit TRAINING OK")
