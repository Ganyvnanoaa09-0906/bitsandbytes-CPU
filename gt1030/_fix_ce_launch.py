# -*- coding: utf-8 -*-
"""修 CE 启动尺寸：ce_softmax 用 get_group_id(0) 当行号，所以 global size 必须是 M*lsz，
   否则只处理 8 行（M=2048 / 256），loss 被严重低估（32.1 vs 6.17）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()
old = """        K['ce_softmax'](d.q, (M,), (256,), self.sc['logits'], self.tgt, self.msk,
                        self.sc['dlogits'], self.loss_row, np.int32(self.V), np.float32(inv_cnt))"""
new = """        # ⚠️ 必须 M*256：kernel 里用 get_group_id(0) 当行号，(M,) 只有 M/256 个工作组，
        # 也就是只会算前 8 行（loss 少了两个数量级，而且 dlogits 大部分是 0）。
        K['ce_softmax'](d.q, (M * 256,), (256,), self.sc['logits'], self.tgt, self.msk,
                        self.sc['dlogits'], self.loss_row, np.int32(self.V), np.float32(inv_cnt))"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched ce_softmax launch size')
