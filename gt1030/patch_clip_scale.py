# -*- coding: utf-8 -*-
"""patch_clip_scale.py — 修"梯度以 ×CE_SCALE 存进 fp16、但 clip 用未缩放范数"的 1024 倍错配。

证据（gt1030/probe_cpu_gnorm2.py，ckpt_46000、同批 token）：
    A = CPU fp32 真梯度范数        = 16.09
    B = 核显 fp16 缓冲里的范数     = 16375.17   （B/A = 1018 ≈ CE_SCALE 1024）
    C = 核显 grad_norm() 上报值    = 16.41      （= B/CE_SCALE，上报值本身是对的）

而 clip 用的是 **C**：  f = max_norm / C = 1/16.4
但被乘的梯度是 **B 尺度**：  结果范数 = B * f = 16375/16.4 ≈ 1000 ≫ max_norm=1
⇒ **梯度裁剪实际被削弱 1024 倍**；同理 `--spike` 判据比较的也是 C，
真实 1024 倍的爆炸在日志里只显示成"正常"的 16。

修法（最小侵入，环境变量开关，默认不开，便于 A/B）：
    AR_CLIP_TRUE=1  ⇒  clip 的 f 用 B = C*CE_SCALE 计算，spike 判据也换成 B 尺度。
"""
import os

import numpy as np

REPO = r"D:\work\bitsandbytes-CPU"
TRAIN = os.path.join(REPO, "gt1030", "ar_v2_train.py")
MODEL = os.path.join(REPO, "gt1030", "ar_v2_multi.py")

# 1) Model.opt_step：clip 用真尺度
m = open(MODEL, encoding="utf-8").read()
if "AR_CLIP_TRUE" in m:
    print("[skip] ar_v2_multi 已改")
else:
    OLD = """    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, max_norm=1.0):
        gn = self.grad_norm()"""
    NEW = """    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, max_norm=1.0):
        gn = self.grad_norm()
        # 2026-11-06：grad 缓冲是【×CE_SCALE 缩放后】的 fp16（实测 B/A≈1024），
        # 而 grad_norm() 返回的是除以 CE_SCALE 的值 ⇒ clip 的 f 必须用真尺度，否则弱 1024 倍。
        gn_clip = gn * (CE_SCALE if os.environ.get('AR_CLIP_TRUE') == '1' else 1.0)"""
    assert OLD in m, "opt_step 锚点没命中"
    m = m.replace(OLD, NEW, 1)
    OLD2 = """        if max_norm and gn > max_norm:
            self.clip_grads(np.float32(max_norm / (gn + 1e-8)))"""
    NEW2 = """        if max_norm and gn_clip > max_norm:
            self.clip_grads(np.float32(max_norm / (gn_clip + 1e-8)))"""
    assert OLD2 in m, "clip 锚点没命中"
    m = m.replace(OLD2, NEW2, 1)
    open(MODEL, "w", encoding="utf-8", newline="\n").write(m)
    print("[ok] ar_v2_multi.opt_step：clip 用真尺度（AR_CLIP_TRUE=1 生效）")

# 2) 训练器的 spike 判据也用真尺度
t = open(TRAIN, encoding="utf-8").read()
if "AR_CLIP_TRUE" in t:
    print("[skip] ar_v2_train 已改")
else:
    OLD = """            gpre = model.grad_norm()
            if not np.isfinite(gpre) or gpre > args.spike:"""
    NEW = """            gpre = model.grad_norm()
            if os.environ.get('AR_CLIP_TRUE') == '1':
                gpre = gpre * MM.CE_SCALE      # 换算回 fp16 缓冲的真实尺度
            if not np.isfinite(gpre) or gpre > args.spike:"""
    assert OLD in t, "spike 锚点没命中"
    t = t.replace(OLD, NEW, 1)
    open(TRAIN, "w", encoding="utf-8", newline="\n").write(t)
    print("[ok] ar_v2_train：spike 判据用真尺度（AR_CLIP_TRUE=1 生效）")
