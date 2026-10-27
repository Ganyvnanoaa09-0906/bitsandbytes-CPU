# -*- coding: utf-8 -*-
"""CE_SCALE 的干净实现：kernel 增加 gscale 参数，dlogits 乘 gscale，loss_row 不乘。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

# 1) kernel 签名 + 用法
s = s.replace("""__kernel void ce_softmax(__global const half* lg, __global const int* tgt,
                         __global const float* mask, __global half* dlg,
                         __global float* loss_row, int V, float inv_cnt) {""",
"""__kernel void ce_softmax(__global const half* lg, __global const int* tgt,
                         __global const float* mask, __global half* dlg,
                         __global float* loss_row, int V, float inv_cnt, float gscale) {""")
s = s.replace("""        out[k] = (half)(mk * (p - ((k == t) ? 1.f : 0.f)) * inv_cnt);""",
              """        out[k] = (half)(mk * (p - ((k == t) ? 1.f : 0.f)) * inv_cnt * gscale);""")

# 2) 去掉两次调用，只调一次
s = s.replace("""        d.K['ce_softmax'](d.q, (M * 256,), (256,), self.logits, self.tgt, self.msk,
                          self.dlogits, self.loss_row, np.int32(self.V),
                          np.float32(inv_cnt))
        # 重新跑一遍 CE，但 dlogits 乘 CE_SCALE（loss_row 已经除过 inv_cnt，这里不受影响）
        d.K['ce_softmax'](d.q, (M * 256,), (256,), self.logits, self.tgt, self.msk,
                          self.dlogits, self.loss_row, np.int32(self.V),
                          np.float32(inv_cnt * CE_SCALE))""",
"""        d.K['ce_softmax'](d.q, (M * 256,), (256,), self.logits, self.tgt, self.msk,
                          self.dlogits, self.loss_row, np.int32(self.V),
                          np.float32(inv_cnt), np.float32(CE_SCALE))""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched gscale')
