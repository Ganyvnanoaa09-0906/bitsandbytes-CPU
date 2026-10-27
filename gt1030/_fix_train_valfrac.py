# -*- coding: utf-8 -*-
"""ar_v2_train.py 增加 --val-frac（默认 0.02，和 CPU trainer 一致：尾部 2% 不参与训练）。

顺带确认：CPU trainer 的 --hflip 只在 token_order='zorder' 时生效（flip_seq 只在那个分支里算），
我们这个 ckpt 是 raster ⇒ hflip 实际不起作用 ⇒ iGPU 侧不需要实现翻转增强。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_train.py'
s = io.open(p, encoding='utf-8').read()
s = s.replace("""    ap.add_argument('--shuffle', type=int, default=1)""",
"""    ap.add_argument('--shuffle', type=int, default=1)
    ap.add_argument('--val-frac', type=float, default=0.02,
                    help='尾部这么多比例的图不参与训练（与 CPU trainer 的 val_frac 一致）')""")
s = s.replace("""    toks = load_tokens(args.tokens)
    n_img = toks.shape[0]""",
"""    toks = load_tokens(args.tokens)
    n_all = toks.shape[0]
    n_train = int(round(n_all * (1.0 - args.val_frac)))
    n_img = n_train                     # 只用前 n_train 张；尾部就是 CPU 的验证集""")
s = s.replace("""    print('[igpu] ckpt=%s start_step=%d  数据 %d 张 x %d token'
          % (os.path.basename(args.ckpt), start_step, n_img, MM.T), flush=True)""",
"""    print('[igpu] ckpt=%s start_step=%d  训练用 %d 张（共 %d，尾部 %.1f%% 是 val）x %d token'
          % (os.path.basename(args.ckpt), start_step, n_img, n_all,
             100.0 * args.val_frac, MM.T), flush=True)""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched val-frac')
