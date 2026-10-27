# -*- coding: utf-8 -*-
"""按层数切片生成 ckpt：python gt1030/_mk_ckptNl.py N"""
import sys
import torch

src = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
N = int(sys.argv[1])
dst = r'F:\weibo_cos\ckpt_%dl.pt' % N
ck = torch.load(src, map_location='cpu', weights_only=False)
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
cfg['loop_end'] = min(3, N - 1)
out = {'model': new, 'step': ck['step'], 'cfg': cfg}
if 'cluster_of' in ck:
    out['cluster_of'] = ck['cluster_of']
out['offset_of'] = ck.get('offset_of')
torch.save(out, dst)
print('saved', dst, 'params %.2fM' % (sum(v.numel() for v in new.values()) / 1e6))
