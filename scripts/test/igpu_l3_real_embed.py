# -*- coding: utf-8 -*-
"""L3 on real 30M-AR token embeddings (non-AR layout smoke).

x[B,T,d] = token_emb[tokens], target = token_emb[next tokens], MSE, 4-layer Stack.
Compares the same 20-step device (fp16-safe) curve against torch fp32.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import gt1030.igpu_block as B  # noqa: E402

REF = os.path.join(ROOT, "scripts", "test", "igpu_l3_reference.py")
spec = importlib.util.spec_from_file_location("ref_l3", REF)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"F:\weibo_cos\_from76000\ckpt_76400.pt")
    ap.add_argument("--tokens", default=r"F:\weibo_cos\tokens_32x32_weibo.pt")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    emb = ck["model"]["token_emb.weight"].float().numpy()
    toks = torch.load(a.tokens, map_location="cpu", weights_only=False).numpy()
    val = toks[-2:]  # (B,T_full)
    cfg = dict(ref.DEFAULTS)
    cfg["steps"] = a.steps
    Bn, T = 2, cfg["t"]
    tok_x = val[:, :T]
    tok_y = val[:, 1 : T + 1]
    x_np = emb[tok_x].astype(np.float16)  # (B,T,d)
    target_np = emb[tok_y].astype(np.float32)
    rows, d = Bn * T, cfg["d"]

    model = ref.build_model(cfg, cfg["seed"])
    st = B.Stack(
        rows,
        d,
        cfg["ffn"],
        cfg["nh"],
        cfg["hd"],
        T,
        cfg["n_layers"],
        seed=cfg["seed"],
        final_norm=False,
        per_layer=True,
    )
    for rb, db in zip(model.blocks, st.blocks):
        db.activate()
        db.norm1.set_weight(rb.n1.w.detach().numpy())
        db.norm2.set_weight(rb.n2.w.detach().numpy())
        db.attn.qkv.set_weight(rb.attn.qkv.weight.detach().numpy())
        db.attn.wo.set_weight(rb.attn.out.weight.detach().numpy())
        w12 = rb.mlp.w12.weight.detach().numpy()
        db.mlp.gate.set_weight(w12[: cfg["ffn"]])
        db.mlp.up.set_weight(w12[cfg["ffn"] :])
        db.mlp.down.set_weight(rb.mlp.w3.weight.detach().numpy())

    # Align fp16 device weights into the torch reference.
    with torch.no_grad():
        for rb, db in zip(model.blocks, st.blocks):
            db.activate()
            rb.n1.w.copy_(
                torch.from_numpy(B._read(db.norm1.w, (d,)).astype(np.float32))
            )
            rb.n2.w.copy_(
                torch.from_numpy(B._read(db.norm2.w, (d,)).astype(np.float32))
            )
            rb.attn.qkv.weight.copy_(
                torch.from_numpy(B._read(db.attn.qkv.w, (3 * d, d)).astype(np.float32))
            )
            rb.attn.out.weight.copy_(
                torch.from_numpy(B._read(db.attn.wo.w, (d, d)).astype(np.float32))
            )
            g = B._read(db.mlp.gate.w, (cfg["ffn"], d)).astype(np.float32)
            u = B._read(db.mlp.up.w, (cfg["ffn"], d)).astype(np.float32)
            rb.mlp.w12.weight.copy_(torch.from_numpy(np.concatenate([g, u], 0)))
            rb.mlp.w3.weight.copy_(
                torch.from_numpy(
                    B._read(db.mlp.down.w, (d, cfg["ffn"])).astype(np.float32)
                )
            )

    x_t = torch.from_numpy(x_np.astype(np.float32)).reshape(Bn, T, d)
    t_t = torch.from_numpy(target_np).reshape(Bn, T, d)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0)
    G = 65536.0
    dev_losses, ref_losses = [], []
    for step in range(1, cfg["steps"] + 1):
        yh = st.forward_np(x_np.reshape(rows, d))
        dl = float(((yh - target_np.reshape(rows, d)) ** 2).mean())
        dy = (2.0 * (yh - target_np.reshape(rows, d)) / (rows * d) * G).astype(
            np.float16
        )
        st.backward_np(dy)
        st.step(cfg["lr"], step, eps=1e-8 * G)
        dev_losses.append(dl)

        model.zero_grad(set_to_none=True)
        yt = model(x_t)
        rt = ((yt - t_t) ** 2).mean()
        rt.backward()
        opt.step()
        ref_losses.append(float(rt))

    print(
        "device %.8f -> %.8f | torch %.8f -> %.8f"
        % (dev_losses[0], dev_losses[-1], ref_losses[0], ref_losses[-1])
    )
    print(
        "final rel %.4f | worst rel %.4f"
        % (
            abs(dev_losses[-1] - ref_losses[-1]) / ref_losses[-1],
            max(abs(a - b) / b for a, b in zip(dev_losses, ref_losses)),
        )
    )
    if a.out:
        json.dump(
            {"device": dev_losses, "torch": ref_losses}, open(a.out, "w"), indent=2
        )
        print("wrote", a.out)


if __name__ == "__main__":
    main()
