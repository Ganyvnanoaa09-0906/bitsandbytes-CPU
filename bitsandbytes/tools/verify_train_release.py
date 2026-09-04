"""Offline smoke-train: verify the shipped libbitsandbytes_cpu.dll actually does real
CPU training with an 8-bit optimizer on a LOCAL model (no internet, no dataset download).

Works even when the model dir has NO tokenizer (we build random input_ids from vocab),
so it can validate the kernel on a plain decoder-only model like deepseek-coder-1.3b-base.

Run with the release package on PYTHONPATH, e.g.:
    $env:PYTHONPATH="D:\work\那很有乐子了~\bitsandbytes\dist_windows\bitsandbytes-cpu-win_0.50.2.dev0"
    py -3.11 tools\verify_train_release.py --model D:\work\textmodel\deepseek-coder-1.3b-base --steps 2

If the DLL is broken this fails (import / backward / step errors). If it works you'll see
the loss descend over a couple of steps and the native lib path that was used.
"""
import argparse
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import bitsandbytes as bnb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local model dir (offline).")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    from bitsandbytes.cextension import get_native_library
    lib = get_native_library()
    print(f"[*] native lib: {lib}  (compiled_with_cuda={lib.compiled_with_cuda})")
    print(f"[*] bnb version: {bnb.__version__}")
    print(f"[*] torch: {torch.__version__}  dtype={args.dtype}")

    print(f"[*] loading local model ... {args.model}")
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)

    # Build a tiny LOCAL tokenizer-free input set from the vocab (no internet).
    vocab = cfg.vocab_size
    ids = torch.randint(0, vocab, (args.batch, args.maxlen))
    labels = ids.clone()

    print(f"[*] optimizer: bnb.optim.AdamW8bit  lr={args.lr}")
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)

    model.train()
    if args.steps < 1:
        raise ValueError(f"--steps 必须 >=1，得到 {args.steps}")
    if args.batch < 1:
        raise ValueError(f"--batch 必须 >=1，得到 {args.batch}")
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        optimizer.zero_grad()
        out = model(input_ids=ids, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  step {step:2d} | loss {loss.item():.4f} | elap {time.time()-t0:.1f}s")

    t = time.time() - t0
    print("\n--- Results ---")
    if not losses:
        print("WARNING: 0 steps -- no losses recorded (did you pass --steps >= 1?).")
        return
    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (delta {losses[-1]-losses[0]:+.4f})")
    print(f"Total time: {t:.1f}s  ({args.steps/t:.2f} steps/s)")
    if losses[-1] < losses[0]:
        print("OK: loss decreased -- CPU kernel training works with the shipped DLL.")
    else:
        print("WARNING: loss did not decrease; investigate.")


if __name__ == "__main__":
    main()
