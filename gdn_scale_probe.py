import torch
from bitsandbytes.gdn_cpu import fused_recurrent_gated_delta_rule

torch.manual_seed(0)
B, T, H, K, V = 2, 40, 2, 16, 16
q = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1)
k = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1)
v = torch.randn(B, T, H, V) * 0.5
beta = torch.rand(B, T, H) * 0.9 + 0.05
g = -torch.rand(B, T, H) * 0.1

for scale in (None, 0.25, 2.0):
    qc = q.clone().requires_grad_(True)
    o, sf = fused_recurrent_gated_delta_rule(qc, k, v, beta, g, scale=scale, output_final_state=True)
    go, gs = torch.ones_like(o), torch.zeros(B, H, K, V)
    dq = torch.autograd.grad((o, sf), (qc,), grad_outputs=(go, gs))[0]
    # reference: scale folded manually
    qd = (q * (scale or 1.0)).double().clone().requires_grad_(True)
    kd, vd = k.double(), v.double()
    betad, gd = beta.double(), g.double()
    S = torch.zeros(B, H, K, V, dtype=torch.float64)
    outs = []
    for t in range(T):
        qt, kt, vt = qd[:, t], kd[:, t], vd[:, t]
        bt = betad[:, t]
        S = S * torch.exp(gd[:, t]).unsqueeze(-1).unsqueeze(-1)
        u = vt - torch.einsum("bhkv,bhk->bhv", S, kt)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, bt.unsqueeze(-1) * u)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, qt))
    ro = torch.stack(outs, 1)
    ro.backward(torch.ones_like(ro))
    print(f"scale={scale}: dq_sum={dq.sum().item():+.6e} ref={qd.grad.sum().item():+.6e} "
          f"ratio={(dq.sum()/qd.grad.sum()).item():.4f} "
          f"maxrel={((dq - qd.grad.float()).abs().max() / qd.grad.abs().max()).item():.3e}")
