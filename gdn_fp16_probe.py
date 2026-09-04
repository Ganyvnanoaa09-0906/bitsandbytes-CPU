import torch
from bitsandbytes.gdn_cpu import fused_recurrent_gated_delta_rule, load_native

torch.manual_seed(0)
B, T, H, K, V = 1, 33, 2, 16, 24
for dtype in (torch.float32, torch.bfloat16, torch.float16):
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype) * 0.9 + 0.05
    g = torch.randn(B, T, H, dtype=dtype) * 0.05
    with torch.no_grad():
        o, sf = fused_recurrent_gated_delta_rule(q, k, v, beta, g, output_final_state=True)
    print(f"{dtype}: o finite={torch.isfinite(o.float()).all().item()} "
          f"max|o|={o.float().abs().max().item():.3f} "
          f"sf finite={torch.isfinite(sf.float()).all().item()} "
          f"nan_o={torch.isnan(o.float()).sum().item()} inf_o={torch.isinf(o.float()).sum().item()}")
    # bit pattern of first bad element
    bad = ~torch.isfinite(o.float())
    if bad.any():
        idx = bad.nonzero()[0].tolist()
        print("  first bad idx:", idx, "raw:", o.float()[tuple(idx)].item())
        print("  q/k/v at that t:", q[idx[0], idx[1], idx[2], :3].tolist())
