"""termux_train.py -- training smoke that needs ONLY torch and bitsandbytes.

WHY NOT transformers
--------------------
The first version imported Qwen3Next and died on the phone with
"ModuleNotFoundError: No module named 'transformers'", and transformers cannot be
pip-installed on Android: its dependencies tokenizers and safetensors are Rust
extensions, regex and pyyaml are C extensions, and every aarch64 wheel on PyPI is
manylinux (glibc) while Android uses bionic. Termux's own repo has none of them.

Chasing that dependency would mean compiling a Rust toolchain on the phone to
test an ORTHOGONAL thing. The question actually being asked is "does the 8-bit
training path descend on ARM64", and that needs neither transformers nor a
pretrained config -- it needs nn.Module, autograd, and the bnb kernels.

So this builds the model out of `bnb.nn.Linear8bitLt`, which is the QUANTIZED
weight path, and optimizes it with `bnb.optim.AdamW8bit`. Progressing the loss
therefore means the quantize/dequantize kernels ran inside a real forward AND
that gradients flowed back through them.

Measured on x86 with probe_quant_layers.py before writing this:
    nn.Linear8bitLt   forward OK, backward OK, |w.grad|max = 4.0000e+00
    nn.Linear4bit     forward FAILS: "FP4 quantization state not initialized.
                      Please call .cuda()"  -- CUDA-only, a design boundary
                      rather than a Termux problem, so 4-bit layers are not used.

PASS CRITERIA (deliberately more than "the last number is smaller")
------------------------------------------------------------------
  * every loss finite          -- a broken 8-bit state shows up as NaN first
  * final < first              -- the headline requirement
  * descent past the start     -- best of the last fifth beats best of the first
                                  fifth, so a single lucky step cannot pass it
  * gradients non-zero         -- an all-zero grad path would "converge" by
                                  doing nothing
  * 8-bit optimizer state allocated -- if the kernel silently no-opped, the
                                  state dict would be empty and weights frozen
  * the 8-bit LINEAR layers actually quantized -- checked via their own
                                  state/weight bookkeeping, not assumed

Usage:
    python termux_train.py [--steps 50] [--threads 4] [--out result.json]
"""
import argparse
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

R = []


def check(name, ok, detail=""):
    R.append({"name": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    return ok


class Block(nn.Module):
    """Pre-norm transformer block whose projections are 8-bit quantized layers."""

    def __init__(self, d_model, n_heads, ff, quant):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = quant(d_model, 3 * d_model)
        self.proj = quant(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = quant(d_model, ff)
        self.fc2 = quant(ff, d_model)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        h = self.ln2(x)
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(h)))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d_model=128, n_heads=4, n_layers=2, ff=256,
                 seq=48, quant=None):
        super().__init__()
        quant = quant or (lambda i, o: nn.Linear(i, o))
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, ff, quant) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    print("=" * 70)
    print(f"Termux {a.steps}-step training smoke (torch + bitsandbytes only)")
    print("=" * 70)
    print(f"  host    : {platform.node()}")
    print(f"  machine : {platform.machine()}")
    print(f"  python  : {platform.python_version()}")
    print(f"  torch   : {torch.__version__}")
    print(f"  threads : {torch.get_num_threads()}")
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"  memory  : {vm.total/1e9:.1f} GB total, {vm.available/1e9:.1f} GB available")
    except Exception:  # noqa: BLE001
        pass

    import bitsandbytes as bnb

    # ---------------------------------------------------------------- model
    vocab, seq = 256, 48
    n_quant = 0

    def quant(i, o):
        nonlocal n_quant
        n_quant += 1
        return bnb.nn.Linear8bitLt(i, o, has_fp16_weights=False)

    print()
    print("building TinyLM with 8-bit quantized projections ...")
    t0 = time.perf_counter()
    model = TinyLM(vocab=vocab, quant=quant)
    model.train()
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  {n_all:,} params, {n_quant} Linear8bitLt layers, "
          f"built in {time.perf_counter()-t0:.2f}s")

    if n_quant == 0:
        check("quantized layers were used", False, "no Linear8bitLt constructed")
        return finish(a, model=None)

    # A forward pass on its own proves the quantize path runs; do it once and
    # assert it is finite before starting the loop, so a numeric failure is
    # reported as such rather than as a mysterious loss curve.
    x0 = torch.randint(0, vocab, (2, seq))
    with torch.no_grad():
        y0 = model(x0)
    check("first forward is finite (quantize path runs)",
          bool(torch.isfinite(y0).all()), f"out {tuple(y0.shape)}")

    # ---------------------------------------------------------------- train
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(trainable, lr=3e-3)
    x = torch.randint(0, vocab, (2, seq))

    print()
    print(f"training {a.steps} steps ...")
    losses, gradnorms = [], []
    t0 = time.perf_counter()
    for step in range(a.steps):
        opt.zero_grad()
        out = model(x)
        loss = nn.functional.cross_entropy(
            out.view(-1, vocab), x.view(-1))
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1e9))
        opt.step()
        losses.append(float(loss.detach()))
        gradnorms.append(gn)
        if step == 0 or (step + 1) % 10 == 0:
            print(f"  step {step+1:3d}  loss {losses[-1]:.4f}  grad_norm {gn:.3e}"
                  f"  {1000*(time.perf_counter()-t0)/(step+1):.0f} ms/step", flush=True)
    dt = time.perf_counter() - t0
    print(f"  {a.steps} steps in {dt:.1f}s ({1000*dt/a.steps:.0f} ms/step)")

    # ---------------------------------------------------------------- checks
    print()
    print("=" * 70)
    print("CHECKS")
    print("=" * 70)

    check("all losses finite",
          all(v == v and abs(v) != float("inf") for v in losses),
          f"min {min(losses):.4f} max {max(losses):.4f}")

    drop = 100.0 * (1.0 - losses[-1] / losses[0])
    check("loss decreased", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}  ({drop:+.2f}%)")

    k = max(1, a.steps // 5)
    early_best, late_best = min(losses[:k]), min(losses[-k:])
    check("descent continued past the start", late_best < early_best,
          f"best early {early_best:.4f} -> best late {late_best:.4f}")

    check("gradients non-zero", all(g > 0 for g in gradnorms),
          f"grad_norm {gradnorms[0]:.3e} -> {gradnorms[-1]:.3e}")

    n_state = sum(1 for g in opt.state.values()
                  for kk in ("state1", "state2") if kk in g)
    check("8-bit optimizer state allocated", n_state > 0, f"{n_state} state entries")

    # Did the quantized layers keep their quantized weight, or silently fall back
    # to a plain float matmul? Linear8bitLt stores the weight in CB/SCB after the
    # first forward; check that bookkeeping exists rather than assuming it.
    lts = [m for m in model.modules() if isinstance(m, bnb.nn.Linear8bitLt)]
    with_state = sum(1 for m in lts if getattr(m, "state", None) is not None
                     and getattr(m.state, "CB", None) is not None)
    check("quantized weights materialised (state.CB present)", with_state > 0,
          f"{with_state}/{len(lts)} Linear8bitLt layers have quantized state")

    return finish(a, losses, gradnorms, dt, n_all, n_quant, len(lts), with_state)


def finish(a, losses=None, gradnorms=None, dt=None, n_all=0, n_quant=0,
           n_lts=0, n_with_state=0, model=None):
    npass = sum(1 for r in R if r["pass"])
    print()
    print("=" * 70)
    print(f"RESULT: {npass}/{len(R)} checks passed")
    for r in R:
        if not r["pass"]:
            print(f"  FAIL: {r['name']}  {r['detail']}")
    print("=" * 70)
    if losses:
        print(f"  loss  : {losses[0]:.4f} -> {losses[-1]:.4f}")
        stride = max(1, len(losses) // 10)
        print(f"  curve : {[round(v, 4) for v in losses[::stride]]}")
    ok = bool(R) and npass == len(R)
    print("TERMUX-TRAIN " + ("PASSED" if ok else "FAILED"))

    payload = {
        "host": platform.node(), "machine": platform.machine(),
        "python": platform.python_version(), "torch": torch.__version__,
        "steps": a.steps, "threads": a.threads,
        "params": n_all, "quant_layers": n_quant, "linear8bitlt": n_lts,
        "linear8bitlt_with_state": n_with_state,
        "wall_s": dt, "ms_per_step": (1000 * dt / a.steps) if dt else None,
        "losses": losses, "gradnorms": gradnorms,
        "checks": R, "passed": ok,
    }
    out = a.out or os.path.join(HERE, "termux_train_result.json")
    targets = [out]
    # /data/local/tmp is used to get the result off the device, but it is NOT
    # writable by the Termux uid (the adb shell owns what it creates there), so
    # it is tried second and a failure is expected rather than reported as a
    # problem. The copy inside the working directory is the real artefact.
    if "/data/local/tmp" not in out:
        targets.append("/data/local/tmp/termux_train_result.json")
    for target in targets:
        try:
            with open(target, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            print(f"  wrote {target}")
        except OSError as e:
            # Not fatal: on Android the shared path is usually not writable, and
            # saying so quietly is better than a scary traceback at the end of a
            # run that actually passed.
            print(f"  (skipped {target}: {type(e).__name__})")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
