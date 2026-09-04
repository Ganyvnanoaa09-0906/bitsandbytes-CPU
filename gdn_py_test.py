"""Python e2e validation of gdn_cpu.fused_recurrent_gated_delta_rule.

- forward output & final state vs float64 per-timestep reference
- all gradients (dq,dk,dv,dbeta,dg,dS0) vs autograd through the reference
- layouts: head_first, non-contiguous inputs, initial_state, g=None, scale
- patch_fla/patch_transformers graceful no-op when targets missing
"""
import torch

from bitsandbytes.gdn_cpu import fused_recurrent_gated_delta_rule, patch_fla, patch_transformers

torch.manual_seed(0)


def reference(q, k, v, beta, g, s0=None, scale=None, l2norm=False):
    """fla-semantics float64 reference with autograd.

    Returns grads at the ORIGINAL (pre-scale/pre-l2norm) leaves, matching the
    wrapper's user-facing gradients (the wrapper folds scale into torch ops,
    so the chain rule through them is exact).
    """
    q_leaf = q.double().clone().requires_grad_(True)
    k_leaf = k.double().clone().requires_grad_(True)
    v = v.double().clone().requires_grad_(True)
    beta = beta.double().clone().requires_grad_(True)
    g = None if g is None else g.double().clone().requires_grad_(True)
    s0 = None if s0 is None else s0.double().clone().requires_grad_(True)
    q, k = q_leaf, k_leaf
    if scale is not None:
        q = q * scale
    if l2norm:
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
    B, T, H, K = q.shape
    V = v.shape[-1]
    S = torch.zeros(B, H, K, V, dtype=torch.float64) if s0 is None else s0.clone()
    outs = []
    for t in range(T):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]        # [B,H,K]
        bt = beta[:, t]                                # [B,H]
        if g is not None:
            S = S * torch.exp(g[:, t]).unsqueeze(-1).unsqueeze(-1)
        u = vt - torch.einsum("bhk,bhk->bhv", S, kt) if False else vt - torch.einsum("bhkv,bhk->bhv", S, kt)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, bt.unsqueeze(-1) * u)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, qt))
    o = torch.stack(outs, dim=1)                      # [B,T,H,V]
    return o, S, (q_leaf, k_leaf, v, beta, g, s0)


def max_rel(a, b):
    d = (a.double() - b.double()).abs().max().item()
    s = b.double().abs().max().item()
    return d / max(s, 1e-9)


def check_case(name, B, T, H, K, V, dtype, with_g=True, with_s0=True, scale=None, **kw):
    # l2-normalized q/k (as in the real Qwen3-Next layer) keep the delta-rule
    # recurrence contractive; raw randn k makes S explode (fp16 then overflows
    # its 65504 max - a property of the math, not a kernel bug).
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.float32), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.float32), dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, dtype=dtype) * 0.5
    beta = torch.rand(B, T, H, dtype=dtype) * 0.9 + 0.05
    g = -torch.rand(B, T, H, dtype=dtype) * 0.1 if with_g else None  # decay <= 1
    s0 = torch.randn(B, H, K, V, dtype=dtype) * 0.1 if with_s0 else None

    qc, kc, vc, bc, gc, sc = (
        t.clone().requires_grad_(True) if torch.is_tensor(t) else t
        for t in (q, k, v, beta, g, s0)
    )
    o, sf = fused_recurrent_gated_delta_rule(
        qc, kc, vc, bc, gc, scale=scale, initial_state=sc, output_final_state=True, **kw
    )
    assert o.dtype == dtype

    ro, rS, rinputs = reference(q, k, v, beta, g, s0, scale=scale)
    e_o = max_rel(o, ro)
    e_s = max_rel(sf.float(), rS)

    # grads: seed with a fixed pattern on o and S
    go = torch.randn_like(o.float())
    gs = torch.randn(B, H, K, V)
    named = {"q": qc, "k": kc, "v": vc, "beta": bc, "g": gc, "s0": sc}
    inputs = [t for t in named.values() if torch.is_tensor(t)]
    names = [n_ for n_, t in named.items() if torch.is_tensor(t)]
    grads = torch.autograd.grad((o.float(), sf.float()), inputs, grad_outputs=(go, gs))
    gmap = dict(zip(names, grads))
    torch.autograd.backward(ro, grad_tensors=go.double(), retain_graph=True)
    torch.autograd.backward(rS, grad_tensors=gs.double())
    rq, rk, rv, rbeta, rg, rs0 = rinputs

    errs = {
        "o": e_o, "S": e_s,
        "dq": max_rel(gmap["q"], rq.grad), "dk": max_rel(gmap["k"], rk.grad),
        "dv": max_rel(gmap["v"], rv.grad), "dbeta": max_rel(gmap["beta"], rbeta.grad),
    }
    if "g" in gmap and rg is not None:
        errs["dg"] = max_rel(gmap["g"], rg.grad)
    if "s0" in gmap and rs0 is not None:
        errs["ds0"] = max_rel(gmap["s0"], rs0.grad)
    worst = max(errs.values())
    print(f"{name:34s} worst_rel={worst:.2e}  " + " ".join(f"{k_}={v_:.1e}" for k_, v_ in errs.items()))
    assert worst < 5e-2, f"{name}: errors too large"


check_case("fp32 full", 2, 64, 3, 32, 40, torch.float32)
check_case("bf16 full", 2, 48, 2, 32, 32, torch.bfloat16)
check_case("fp16 full", 1, 33, 2, 16, 24, torch.float16)
check_case("no g", 2, 40, 2, 16, 16, torch.float32, with_g=False)
check_case("no s0", 2, 40, 2, 16, 16, torch.float32, with_s0=False)
check_case("scale", 2, 40, 2, 16, 16, torch.float32, scale=0.25)
check_case("scale minimal", 2, 40, 2, 16, 16, torch.float32, scale=0.25, with_g=False, with_s0=False)

# head_first layout
q = torch.randn(2, 3, 50, 16)  # [B,H,T,K]
k = torch.randn(2, 3, 50, 16)
v = torch.randn(2, 3, 50, 16)
beta = torch.rand(2, 3, 50)
g = torch.randn(2, 3, 50) * 0.05
o_hf, s_hf = fused_recurrent_gated_delta_rule(
    q, k, v, beta, g, output_final_state=True, head_first=True
)
o_ref, S_ref, _ = reference(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    beta.transpose(1, 2), g.transpose(1, 2),
)
o_ref = o_ref.transpose(1, 2)  # back to [B,H,T,V]
assert o_hf.shape == (2, 3, 50, 16)
print(f"head_first                         worst_rel={max_rel(o_hf, o_ref):.2e}")
assert max_rel(o_hf, o_ref) < 5e-2

# non-contiguous inputs (transposed views)
qb = torch.randn(2, 3, 60, 16).transpose(1, 2)   # [B,T,H,K] strided
kb = torch.randn(2, 3, 60, 16).transpose(1, 2)
vb = torch.randn(2, 3, 60, 24).transpose(1, 2)
bb = torch.rand(2, 3, 60).transpose(1, 2)
gb = torch.randn(2, 3, 60).transpose(1, 2) * 0.05
o_nc, _ = fused_recurrent_gated_delta_rule(qb, kb, vb, bb, gb)
o_ref2, _, _ = reference(qb.contiguous(), kb.contiguous(), vb.contiguous(), bb.contiguous(), gb.contiguous())
assert max_rel(o_nc, o_ref2) < 5e-2
print(f"non-contiguous                     worst_rel={max_rel(o_nc, o_ref2):.2e}")

# grad flows through non-contiguous too
qg = qb.clone().transpose(1, 2).transpose(1, 2).requires_grad_(True)
o_g, _ = fused_recurrent_gated_delta_rule(qg, kb, vb, bb, gb)
o_g.sum().backward()
assert qg.grad is not None and torch.isfinite(qg.grad).all()
print("grad through strided input          ok")

# patch entries: fla/transformers not installed here -> graceful False
print(f"patch_fla={patch_fla()} patch_transformers={patch_transformers()} (both should be False or no-crash)")

# scale=None path returns final_state None by default
o3, s3 = fused_recurrent_gated_delta_rule(qb, kb, vb, bb, gb)
assert s3 is None
print("default final_state=None            ok")

print("ALL GDN PYTHON TESTS PASSED")
