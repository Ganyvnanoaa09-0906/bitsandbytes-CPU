import argparse
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes.fused as F

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['full', 'chunk'], required=True)
a = ap.parse_args()
torch.set_num_threads(6)
M, D, V = 1024, 2048, 151936
h = torch.randn(M, D)
w = torch.randn(V, D)
lab = torch.randint(0, V, (M,))
t = time.perf_counter()
if a.mode == 'full':
    logits = h @ w.t()
    loss = torch.nn.functional.cross_entropy(logits, lab)
else:
    loss = F.linear_cross_entropy(h, w, lab, chunk=1024)
print('RESULT', a.mode, 'ms', round((time.perf_counter() - t) * 1e3, 1), 'loss', float(loss), flush=True)
