# -*- coding: utf-8 -*-
"""merge_captions.py — 把 6 个目录的 caption 合成一份，并把 path 改写成【token 文件里的原样路径】。

⚠️ 为什么必须改写（这是个会重演"条件静默失效"的坑）：
  `train_ar_v2.load_caption_cond` 的匹配键是 `os.path.abspath(p)`：
      by  = {abspath(c['path']): caption}
      text= by.get(abspath(token_path), 'a person')
  - auto_caption 写出的是【相对路径】（如 `.autopilot\\clean_tu\\images\\x.jpg`）；
  - token 文件的 paths.json 是【Windows 绝对路径】（`D:\\work\\...`）；
  - 而训练跑在 Linux 云端。
  ⇒ 两边 abspath 的结果毫无交集，**全部落到默认值 'a person'**，
    条件会变成"所有图片同一个常量" —— 又是"实现了但静默失效"。
  修法：按 basename 匹配，把 caption 的 path **改写成 token 文件里的那条原样字符串**，
  这样无论在哪个平台，两边 abspath 的输入字符串完全相同。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import json, os, sys, collections

TOK_PATHS = r'.autopilot\tokens_16x16_full.pt.paths.json'
CAPS = [r'.autopilot\captions\cap_%s.jsonl' % d for d in
        ('clean_nf', 'clean_dl', 'clean_char2_1024', 'clean_webp', 'clean_tu', 'clean_za')]
OUT = r'.autopilot\captions\captions_merged.jsonl'


def key_of(p):
    """取 .autopilot 之后的路径后缀做匹配键（两端都能算出来，且不受平台/绝对前缀影响）。

    只用 basename 会有 168 个重名冲突（不同目录下有同名文件），
    会让那些图拿到【别人的 caption】—— 这是"看起来对了"的静默错误。
    """
    s = p.replace('\\', '/')
    i = s.find('autopilot/')
    if i >= 0:
        return s[i + len('autopilot/'):]
    # 云端 token 路径可能是 /workspace/... 形式，退化到后两段
    return '/'.join(s.split('/')[-3:])


def main():
    tok = json.load(open(TOK_PATHS, encoding='utf-8'))
    print('[merge] token 行数 %d' % len(tok))

    by_key = {}
    dup = collections.Counter()
    for p in tok:
        k = key_of(p)
        dup[k] += 1
        by_key.setdefault(k, p)
    collide = [k for k, n in dup.items() if n > 1]
    print('[merge] 路径后缀冲突 %d 个%s'
          % (len(collide), ('（例：%s）' % collide[:3]) if collide else ''))

    cap_by_key = {}
    total = 0
    for f in CAPS:
        if not os.path.exists(f):
            print('  [warn] 缺 %s' % f); continue
        n = 0
        for line in open(f, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cap_by_key[key_of(r['path'])] = r
            n += 1
        total += n
        print('  %-46s %6d 条' % (os.path.basename(f), n))
    print('[merge] caption 合计 %d 条, 唯一键 %d' % (total, len(cap_by_key)))

    hit, miss, empty = 0, 0, 0
    with open(OUT, 'w', encoding='utf-8') as f:
        for p in tok:
            r = cap_by_key.get(key_of(p))
            if r is None:
                miss += 1
                cap = 'a person'
            else:
                hit += 1
                cap = r.get('caption', '') or 'a person'
                if not r.get('caption'):
                    empty += 1
            f.write(json.dumps({'path': p, 'caption': cap}, ensure_ascii=False) + '\n')
    print()
    print('[merge] ✅ 命中 %d / %d  (%.1f%%)   未命中 %d   空 caption %d'
          % (hit, len(tok), 100.0 * hit / len(tok), miss, empty))
    print('[merge] -> %s' % OUT)

    # 覆盖率不够高的话，条件训练等于在噪声上训 —— 必须显式告警
    if hit < 0.95 * len(tok):
        print('[merge] ⚠️ 命中率不足 95%%，先别用这份 caption 训条件模型！')
    # 顺便看 caption 是否退化成一个常量（另一类静默失效）
    u = collections.Counter()
    for line in open(OUT, encoding='utf-8'):
        u[json.loads(line)['caption']] += 1
    print('[merge] 不同 caption 数 = %d；最高频占 %.1f%%（若接近 100%% 说明等于无条件）'
          % (len(u), 100.0 * u.most_common(1)[0][1] / len(tok)))


if __name__ == '__main__':
    main()
