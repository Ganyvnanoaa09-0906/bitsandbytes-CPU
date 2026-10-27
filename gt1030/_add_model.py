# -*- coding: utf-8 -*-
"""给 ar_v2_multi 追加：scale_half kernel + Model（embed / 10 层 / head / 裁剪 / AdamW）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

# 1) 增加 scale_half kernel
s = s.replace("""    if (lid == 0) loss_row[r] = red[0] * mk * inv_cnt;
}
\"\"\"""",
"""    if (lid == 0) loss_row[r] = red[0] * mk * inv_cnt;
}
__kernel void scale_half(__global half* g, float f, int n) {
    int i = get_global_id(0); if (i >= n) return;
    g[i] = (half)((float)g[i] * f);
}
\"\"\"""")
s = s.replace("""          'ce_softmax')""", """          'ce_softmax', 'scale_half')""")

# 2) 追加 Model
s += '''

class Model(object):
    """整机：host 侧 embedding（token/2D pos）+ 每层一个 context + head context。"""

    def __init__(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = ck['cfg']
        assert cfg['cond_mode'] == 'none' and cfg['pos_mode'] == '2d'
        assert cfg['head_mode'] == 'flat' and not cfg['use_bos'] and cfg['token_order'] == 'raster'
        assert cfg['grid'] * cfg['grid'] == T, 'grid %d != T %d' % (cfg['grid'], T)
        sd = ck['model']
        self.cfg = cfg
        self.n_layer = int(cfg['n_layer'])
        self.blocks = [Block(sd, i) for i in range(self.n_layer)]
        self.head = Head(sd)
        V = self.head.V
        g = cfg['grid']
        self.tok_tab = sd['token_emb.weight'].numpy().astype(np.float32)
        self.row_tab = sd['row_emb'].numpy().astype(np.float32)
        self.col_tab = sd['col_emb'].numpy().astype(np.float32)
        ordr = sd['order'].numpy() if 'order' in sd else np.arange(g * g)
        self.row_idx = (ordr // g).astype(np.int64)[:T]
        self.col_idx = (ordr % g).astype(np.int64)[:T]
        self.pos_tab = (self.row_tab[self.row_idx] + self.col_tab[self.col_idx])
        self.row_idx_full = np.tile(self.row_idx, B)
        self.col_idx_full = np.tile(self.col_idx, B)
        self.tok_m = np.zeros_like(self.tok_tab); self.tok_v = np.zeros_like(self.tok_tab)
        self.row_m = np.zeros_like(self.row_tab); self.row_v = np.zeros_like(self.row_tab)
        self.col_m = np.zeros_like(self.col_tab); self.col_v = np.zeros_like(self.col_tab)
        self.last_loss = None

    # ---------------- 前向 ----------------
    def embed(self, tokens):
        ids = tokens.reshape(-1)
        return (self.tok_tab[ids] + np.tile(self.pos_tab, (B, 1))).astype(np.float16)

    def forward(self, tokens):
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk.forward(x)
        self.head.forward(x)
        self.last_loss = self.head.loss(tokens)
        self.x_last = x
        return self.last_loss

    # ---------------- 反向 ----------------
    def backward(self, tokens):
        dx = self.head.backward()
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        # embedding 梯度
        d = dx.astype(np.float32)
        ids = tokens.reshape(-1)
        self.dtok = np.zeros_like(self.tok_tab); np.add.at(self.dtok, ids, d)
        self.drow = np.zeros_like(self.row_tab); np.add.at(self.drow, self.row_idx_full, d)
        self.dcol = np.zeros_like(self.col_tab); np.add.at(self.dcol, self.col_idx_full, d)
        return dx

    # ---------------- 优化器 ----------------
    def grad_norm(self):
        tot = self.head.dev_gnorm() if hasattr(self.head, 'dev_gnorm') else 0.0
        d = self.head.dev
        g = d.down(self.head.W.grad, (self.head.W.n,), np.float16).astype(np.float32)
        tot += float((g * g).sum())
        for blk in self.blocks:
            tot += blk.grad_norm()
        # embedding（host）
        for gg in (self.dtok, self.drow, self.dcol):
            tot += float((gg.astype(np.float32) ** 2).sum())
        return float(np.sqrt(tot))

    def clip_grads(self, f):
        for blk in self.blocks:
            d = blk.dev
            d.activate()
            for k in ('Wqkv', 'Wo', 'W12', 'W3'):
                p = blk.W[k]
                d.K['scale_half'](d.q, (p.n,), None, p.grad, np.float32(f), np.int32(p.n))
            d.finish()
        d = self.head.dev
        d.activate()
        p = self.head.W
        d.K['scale_half'](d.q, (p.n,), None, p.grad, np.float32(f), np.int32(p.n))
        d.finish()
        for gg in (self.dtok, self.drow, self.dcol):
            gg *= f

    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, max_norm=1.0):
        gn = self.grad_norm()
        if max_norm and gn > max_norm:
            self.clip_grads(np.float32(max_norm / (gn + 1e-8)))
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        for blk in self.blocks:
            blk.opt_step(lr, t, b1, b2, eps, wd)
        self.head.opt_step(lr, t, b1, b2, eps, wd)
        # embedding：host numpy AdamW
        for tab, g, m, v in ((self.tok_tab, self.dtok, self.tok_m, self.tok_v),
                             (self.row_tab, self.drow, self.row_m, self.row_v),
                             (self.col_tab, self.dcol, self.col_m, self.col_v)):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            tab -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps) - lr * wd * tab
        self.pos_tab = self.row_tab[self.row_idx] + self.col_tab[self.col_idx]
        return gn
'''
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('appended Model')
