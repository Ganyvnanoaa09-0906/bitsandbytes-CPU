# -*- coding: utf-8 -*-
"""做一个 n_layer=5 的 ckpt，用来测试『设备占用减半后驱动是否就稳了』。"""
import sys
import torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')

src = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
dst = r'F:\weibo_cos\ckpt_5l.pt'
ck = torch.load(src, map_location='cpu', weights_only=False)
N = 5
new = {}
for k, v in ck['model'].items():
    if k.startswith('blocks.'):
        i = int(k.split('.')[1])
        if i >= N:
            continue
        nk = 'blocks.%d.%s' % (i, k.split('.', 2)[2])
    else:
        nk = k
    new[nk] = v
cfg = dict(ck['cfg'])
cfg['n_layer'] = N
cfg['loop_end'] = 3
out = {'model': new, 'step': ck['step'], 'cfg': cfg}
if 'cluster_of' in ck:
    out['cluster_of'] = ck['cluster_of']
out['offset_of'] = ck.get('offset_of')
torch.save(out, dst)
print('saved', dst, 'params %.2fM' % (sum(v.numel() for v in new.values()) / 1e6))
