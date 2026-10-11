
import os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, r"D:\\work\\bitsandbytes-CPU")
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

FUSE = os.environ.get("FUSE") == "1"
T, B = 256, 8
torch.set_num_threads(6)
torch.manual_seed(1234)

cfg = SmallImageConfigV2(
    vocab_size=1024, total_tokens=T, grid=16,
    d_model=512, n_layer=10, n_head=8, n_kv_head=4, ffn_dim=2048,
    head_mode="flat", cond_mode="none", pos_mode="2d", token_order="raster",
    loop_times=1, fuse_qkv=FUSE,
)
m = SmallARImageModelV2(cfg)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
x = torch.randint(0, 1024, (B, T))
y = torch.randint(0, 1024, (B, T))


def step():
    out = m.forward_logits(x)
    gen = out[1]
    logits = gen[0] if isinstance(gen, tuple) else gen
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss)


l0 = step()
for _ in range(2):
    step()
t0 = time.perf_counter()
for _ in range(4):
    step()
dt = (time.perf_counter() - t0) / 4
nparam = sum(p.numel() for p in m.parameters())
tag = "FUSED" if FUSE else "SEP"
print(tag, l0, dt, nparam / 1e6)
