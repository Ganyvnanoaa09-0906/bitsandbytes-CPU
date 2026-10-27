# -*- coding: utf-8 -*-
"""修精度：CE 的 dlogits 量级只有 1/(B*(T-1)) ≈ 5e-4，反向越往下越小，dg 掉到 1e-6 ——
那已经进入 fp16 次正规区（最小正规 6.1e-5，次正规只有 6e-8~6e-5），有效位数只剩几位，
于是 dg/da2/dW12 误差 50%+。

对策：把 CE 梯度整体乘 CE_SCALE（AdamW 对梯度尺度不敏感，更新方向不变；eps 用 1e-8 足够小），
算 loss / 算 grad-norm 时再除回来。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""scale = L.scale""", """scale = L.scale

# CE 梯度放大倍数：避免反向中间量掉进 fp16 次正规区（见文件头注释）
CE_SCALE = float(os.environ.get('AR_CE_SCALE', '1024'))""")

# Head.loss: 放大 dlogits
s = s.replace("""        inv_cnt = 1.0 / float(msk.sum())""",
              """        inv_cnt = 1.0 / float(msk.sum())""")
s = s.replace("""                          np.float32(inv_cnt))
        d.finish()
        return float(d.down(self.loss_row, (M,), np.float32).sum())""",
              """                          np.float32(inv_cnt))
        # 重新跑一遍 CE，但 dlogits 乘 CE_SCALE（loss_row 已经除过 inv_cnt，这里不受影响）
        d.K['ce_softmax'](d.q, (M * 256,), (256,), self.logits, self.tgt, self.msk,
                          self.dlogits, self.loss_row, np.int32(self.V),
                          np.float32(inv_cnt * CE_SCALE))
        d.finish()
        return float(d.down(self.loss_row, (M,), np.float32).sum())""")

# Model.grad_norm: 除以 CE_SCALE 得到真实范数
s = s.replace("""        # embedding（host）
        for gg in (self.dtok, self.drow, self.dcol):
            tot += float((gg.astype(np.float32) ** 2).sum())
        return float(np.sqrt(tot))""",
"""        # embedding（host）
        for gg in (self.dtok, self.drow, self.dcol):
            tot += float((gg.astype(np.float32) ** 2).sum())
        return float(np.sqrt(tot)) / CE_SCALE""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched CE_SCALE')
