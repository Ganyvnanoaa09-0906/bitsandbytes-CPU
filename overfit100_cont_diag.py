"""Why did prefix-conditioned continuation score 0/128 while teacher-forced top-1 is 43%?

Two candidates: (a) exposure bias -- one sampled misstep takes the context off-distribution and an
overfitted model collapses; (b) something is wrong with the continuation path itself (alignment,
sampler, position handling). Separate them: with the SAME 896-token true prefix, continue greedily
(argmax) and with top-k sampling, on one row, and also check the step-0 prediction directly.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from probe_exposure import load_ar, topk_sample  # noqa: E402

model, cfg = load_ar(r'D:\work\bitsandbytes-CPU\i5build\ar_overfit100_v3\ckpt_last.pt')
T = cfg.total_tokens
tokens = torch.load(r'D:\work\bitsandbytes-CPU\tokens_overfit100.pt', map_location='cpu',
                    weights_only=False).long()
true = tokens[0]
P = 896


def lg(seq):
    with torch.no_grad():
        _th, gen = model.forward_logits(seq, None)
    if isinstance(gen, (tuple, list)):
        cl, cd = gen
        return model.merge(cl, cd, None)
    return gen


# step-0 prediction from the true prefix: is the FIRST continuation token right?
full = lg(true[:P].unsqueeze(0))[0]
probs = torch.softmax(full[-1], -1)
tru = int(true[P])
print('first continuation token: true=%d  argmax=%d (prob %.3f)  top1 acc at this position: %s'
      % (tru, int(full[-1].argmax()), float(probs[full[-1].argmax()]),
         int(true[P]) == int(full[-1].argmax())))
print('  true token prob %.4f · rank %d' % (float(probs[tru]),
                                             int((probs > probs[tru]).sum()) + 1))


def run(mode, P=896, n_steps=128):
    seq = true[:P].unsqueeze(0).clone()
    for _ in range(n_steps):
        l = lg(seq)
        if mode == 'greedy':
            nxt = l[:, -1].argmax(-1)
        else:
            nxt = topk_sample(l[:, -1], 100, 1.0)
        seq = torch.cat([seq, nxt.unsqueeze(1)], 1)
    gen_t = seq[0, P:]
    truth = true[P:P + n_steps]
    agree = float((gen_t == truth).float().mean())
    # how coherent is the generated tail itself: entropy and distinct count
    used = int(torch.unique(gen_t).numel())
    print('  %-7s continuation agreement %.1f%%  distinct tokens %d/%d  first 8 gen=%s true=%s'
          % (mode, agree * 100, used, n_steps, gen_t[:8].tolist(), truth[:8].tolist()))
    return agree


print()
print('continuation from 896 true tokens:')
for P in (896, 512):
    print(' prefix =', P)
    run('greedy', P)
    run('topk', P)
