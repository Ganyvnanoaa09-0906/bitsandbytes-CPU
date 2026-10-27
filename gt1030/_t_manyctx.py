# -*- coding: utf-8 -*-
"""探针 33：一层一个 context，逐个建 block，看第几个开始上传校验失败（context 数量上限？）"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import psutil
import torch
import gt1030.ar_v2_multi as MM

ck = torch.load(r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt', map_location='cpu', weights_only=False)
sd = ck['model']
keep = []
for i in range(10):
    try:
        keep.append(MM.Block(sd, i))
        vm = psutil.virtual_memory()
        print('block %d ok   host_avail=%.2fGB' % (i, vm.available / 1e9))
    except Exception as e:
        print('block %d FAILED: %r' % (i, e))
        break
