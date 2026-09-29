"""End-to-end validation of the CPU training path (fused 8-bit optimizer + GDN).

Run: python -m tests.test_cpu_e2e   (from the bitsandbytes repo root)
"""
import time

import torch

torch.manual_seed(0)


def train(optimizer_factory, steps=150, seed=123):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 256), torch.nn.GELU(),
        torch.nn.Linear(256, 256), torch.nn.GELU(),
        torch.nn.Linear(256, 1),
    )
    opt = optimizer_factory(model.parameters())
    x = torch.randn(1024, 64)
    w = torch.randn(64)
    y = x @ w + 0.3
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x).squeeze(-1), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, model, opt


def main() -> int:
    import bitsandbytes as bnb
    from bitsandbytes.cextension import lib

    fused_available = hasattr(lib, "coptimizer_update_8bit_blockwise_cpu")
    print(f"native lib: {type(lib).__name__}  fused kernel exported: {fused_available}")
    assert fused_available, "fused 8-bit optimizer entry point missing from native lib"

    t0 = time.perf_counter()
    ref_losses, _, _ = train(lambda p: torch.optim.AdamW(p, lr=1e-3))
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    bnb_losses, model, opt = train(lambda p: bnb.optim.AdamW8bit(p, lr=1e-3), steps=150)
    t_bnb = time.perf_counter() - t0

    print(f"loss  step0   torch {ref_losses[0]:.4f}  bnb8bit {bnb_losses[0]:.4f}")
    print(f"loss  final   torch {ref_losses[-1]:.4f} bnb8bit {bnb_losses[-1]:.4f}")
    print(f"time  torch {t_ref*1000:.0f} ms   bnb8bit {t_bnb*1000:.0f} ms")
    assert bnb_losses[-1] < bnb_losses[0] * 0.2, "8-bit AdamW is not converging"
    gap = abs(bnb_losses[-1] - ref_losses[-1])
    print(f"final-loss gap vs fp32 AdamW: {gap:.4f}")
    assert gap < 0.05, f"8-bit path diverged from reference (gap {gap:.4f})"

    # state bytes: uint8 codes + fp32 absmax per 256-block, x2 states.
    # tensors below min_8bit_size keep fp32 states (no absmax) - count those too.
    bytes_8bit = sum(
        s["state1"].numel() + s["absmax1"].numel() * 4 +
        (s["state2"].numel() + s["absmax2"].numel() * 4 if "state2" in s else 0)
        if "absmax1" in s
        else s["state1"].numel() * 4 + (s["state2"].numel() * 4 if "state2" in s else 0)
        for s in opt.state.values() if "state1" in s
    )
    n_param = sum(p.numel() for g in opt.param_groups for p in g["params"])
    print(f"8-bit state: {bytes_8bit/1e6:.1f} MB  vs fp32 Adam states {n_param*8/1e6:.1f} MB "
          f"({n_param*8/max(bytes_8bit, 1):.1f}x smaller)")

    # non-contiguous grad must not corrupt states (falls back to composite path)
    torch.manual_seed(7)
    g_val = torch.randn(64, 64)
    buf = torch.zeros(64, 128)
    buf[:, :64] = g_val
    pa = torch.nn.Parameter(torch.zeros(64, 64))
    pb = torch.nn.Parameter(torch.zeros(64, 64))
    oa = bnb.optim.AdamW8bit([pa], lr=1e-2)
    ob = bnb.optim.AdamW8bit([pb], lr=1e-2)
    for _ in range(3):
        oa.zero_grad(); ob.zero_grad()
        pa.grad = g_val.clone()
        pb.grad = buf[:, :64]  # same values, strides (128,1) -> non-contiguous
        assert not pb.grad.is_contiguous()
        oa.step(); ob.step()
    diff = (pa - pb).abs().max().item()
    print(f"non-contiguous grad update diff: {diff:.2e}")
    assert diff < 1e-6, "non-contiguous grad corrupted the update"

    fused_contracts(bnb)
    four_bit_reload(bnb)

    print("ALL E2E PASSED")
    return 0


def fused_contracts(bnb) -> None:
    """The kernels called directly, against a reference that is obviously right.

    These are the contracts the reference documents (`bitsandbytes-cpu help kernels`), so
    they are checked here rather than only in prose: a documented call that has quietly
    stopped being correct is the failure this file exists to catch.
    """
    N = K = 512
    BLOCK = 64
    torch.manual_seed(0)
    w = torch.randn(N, K) * 0.02
    a = torch.randn(4, K)

    code = bnb.functional.create_linear_map()
    wq, state = bnb.functional.quantize_blockwise(w, code=code, blocksize=BLOCK)
    out = bnb.functional.fused_dequant_linear_8bit(a, wq, state.absmax, BLOCK)
    ref = a @ w.t()
    rel = ((out - ref).norm() / ref.norm()).item()
    print(f"fused_dequant_linear_8bit   rel {rel:.5f} (8-bit weights, expect <0.02)")
    assert out.shape == (4, N)
    assert rel < 0.02, f"fused 8-bit linear drifted from the reference: {rel}"

    # Leading dimensions, which the video path depends on.
    v = torch.randn(2, 3, K)
    lead = bnb.functional.fused_dequant_linear_8bit(v, wq, state.absmax, BLOCK)
    ref_lead = v.reshape(-1, K) @ w.t()
    rel_lead = ((lead.reshape(-1, N) - ref_lead).norm() / ref_lead.norm()).item()
    print(f"  with leading dims (2,3,K) rel {rel_lead:.5f}, shape {tuple(lead.shape)}")
    assert lead.shape == (2, 3, N) and rel_lead < 0.02

    # K not a multiple of blocksize must raise, not read the wrong row.
    w_bad = torch.randn(64, 500) * 0.02
    wq_bad, st_bad = bnb.functional.quantize_blockwise(w_bad, code=code, blocksize=BLOCK)
    try:
        bnb.functional.fused_dequant_linear_8bit(torch.randn(2, 500), wq_bad,
                                                 st_bad.absmax, BLOCK)
    except ValueError as exc:
        print(f"  K=500 refused: {str(exc)[:60]}...")
    else:
        raise AssertionError("fused_dequant_linear_8bit accepted K not divisible "
                             "by blocksize")

    # The fused optimizer step, driven by hand, against torch's own AdamW.
    n = 4096
    p = torch.nn.Parameter(torch.randn(n) * 0.1)
    g = torch.randn(n) * 0.05
    p_ref = torch.nn.Parameter(p.detach().clone())
    p_ref.grad = g.clone()
    torch.optim.AdamW([p_ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8).step()

    qmap = bnb.functional.create_dynamic_map(signed=True, total_bits=8)
    bnb.functional.optimizer_update_8bit_blockwise(
        "adam", g, p,
        torch.zeros(n, dtype=torch.uint8), torch.zeros(n, dtype=torch.uint8),
        0.9, 0.999, 0.0, 1.0, 1e-8, 1, 1e-3,
        qmap, qmap.clone(),
        torch.zeros(n // BLOCK), torch.zeros(n // BLOCK),
        weight_decay=0.0, gnorm_scale=1.0)
    step_rel = ((p.detach() - p_ref.detach()).norm() / p_ref.detach().norm()).item()
    print(f"optimizer_update_8bit_blockwise vs torch AdamW, 1 step: rel {step_rel:.2e}")
    assert step_rel < 1e-3, f"fused optimizer step differs from AdamW by {step_rel}"

    ai = torch.randint(-100, 100, (8, 64), dtype=torch.int8)
    bi = torch.randint(-100, 100, (16, 64), dtype=torch.int8)
    exact = torch.equal(bnb.functional.int8_linear_matmul(ai, bi),
                        ai.to(torch.int32) @ bi.to(torch.int32).t())
    print(f"int8_linear_matmul exact against int32 matmul: {exact}")
    assert exact, "int8_linear_matmul is not exact"


def four_bit_reload(bnb) -> None:
    """A 4-bit layer must survive save -> load through Params4bit(bnb_quantized=True).

    The obvious route -- a fresh Linear4bit and load_state_dict() -- does not work, and
    the reference says so; this pins the route that does, because a quantised checkpoint
    that cannot be reloaded is unrecoverable work.
    """
    torch.manual_seed(0)
    src = torch.nn.Linear(512, 256, bias=False)
    q = bnb.nn.Linear4bit(512, 256, bias=False, compute_dtype=torch.float32,
                          quant_type="nf4", compress_statistics=True)
    q.weight = bnb.nn.Params4bit(src.weight.data, requires_grad=False,
                                 quant_type="nf4", compress_statistics=True)
    q = q.to("cpu")
    x = torch.randn(4, 512)
    y = q(x)

    sd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in q.state_dict().items()}
    state = bnb.functional.QuantState.from_dict(
        {k[len("weight."):]: v for k, v in sd.items() if k.startswith("weight.")},
        device=torch.device("cpu"))
    fresh = bnb.nn.Linear4bit(512, 256, bias=False, compute_dtype=torch.float32,
                              quant_type="nf4")
    pw = bnb.nn.Params4bit(sd["weight"], requires_grad=False,
                           quant_type=state.quant_type, blocksize=state.blocksize,
                           bnb_quantized=True, quant_storage=sd["weight"].dtype)
    pw.quant_state = state
    fresh.weight = pw

    diff = (fresh(x) - y).abs().max().item()
    print(f"4-bit reload: max|y_reloaded - y| {diff:.2e}, "
          f"{len(sd)} tensors in the state dict")
    assert diff < 1e-5, f"reloaded 4-bit layer does not reproduce its output ({diff})"


if __name__ == "__main__":
    raise SystemExit(main())
