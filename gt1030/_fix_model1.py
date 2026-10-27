# -*- coding: utf-8 -*-
"""Patch ar_v2_model.py: wire up embed grads -> host AdamW, drop unused tok_dev upload."""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""        self.tok_dev = d.up(self.tok_tab.astype(np.float16))       # (V,MD) 每步重传
""", "")

s = s.replace("""        self.tok_dev_grad = dtok
        return dtok, drow, dcol""",
"""        self.dtok_last, self.drow_last, self.dcol_last = dtok, drow, dcol
        return dtok, drow, dcol""")

s = s.replace("""        for tab, g, m, v in ((self.tok_tab, self.tok_dev_grad, self.tok_m, self.tok_v),""",
"""        for tab, g, m, v in ((self.tok_tab, self.dtok_last, self.tok_m, self.tok_v),""")

s = s.replace("""            tab -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps) - lr * wd * tab
        self.dev.finish()""",
"""            tab -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps) - lr * wd * tab
        # 位置表由 row/col 表推出，更新后必须重算（否则下一步嵌入用的是旧位置）
        self.pos_tab = self.row_tab[self.row_idx] + self.col_tab[self.col_idx]
        self.dev.finish()""")

io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched; has dtok_last:', 'dtok_last' in s, '| has tok_dev:', 'tok_dev' in s)
