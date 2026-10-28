# -*- coding: utf-8 -*-
"""clip_score_eval.py - 图片-文本对齐度（CLIP score）批量评估。

优先用真 CLIP（openai/clip-vit-base-patch32，已缓存在 <repo>/.hf_home），
不可用时自动退回轻量代理分数，保证脚本永远能跑通。

CLIP score 约定：`100 * cos(image_emb, text_emb)`，越高越对齐（常见范围 10~35）。

用法
----
  # 1) 图片目录 + prompts.json（{文件名: {prompt: ...}} 或 {文件名: "..."}）
  python clip_score_eval.py --images D:\\work\\lora_char2\\images ^
      --prompts D:\\work\\lora_char2\\prompts.json --out scores.jsonl

  # 2) 清洗流水线产出的 metadata.jsonl（用其中的 image 字段）
  python clip_score_eval.py --metadata <clean_out>\\metadata.jsonl --prompt "1girl, anime style"

  # 3) 单个提示词跑整个目录
  python clip_score_eval.py --images <dir> --prompt "1girl, silver hair, anime, portrait"

  # 4) 强制走代理分数（无 CLIP 环境自检）
  python clip_score_eval.py --images <dir> --prompt "..." --no-clip

输出
----
  scores.jsonl  每行 {image, prompt, clip_score, mode}
  report.json   mean / median / p10 / p90 / 每条 prompt 的均分 / 耗时
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse, json, os, statistics, sys, time

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
_REPO = os.path.dirname(os.path.abspath(__file__))
# CLIP 权重缓存在工作区内（默认的 C:\Users\...\.cache 在沙箱里不可写）
if not os.environ.get("HF_HOME") and os.path.isdir(os.path.join(_REPO, ".hf_home")):
    os.environ["HF_HOME"] = os.path.join(_REPO, ".hf_home")


def list_images(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def load_prompts(path):
    """支持 {name: {prompt: str}} 与 {name: str} 两种格式。"""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    m = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            m[k] = v.get('prompt') or v.get('caption') or ''
        else:
            m[k] = str(v)
    return m


def build_pairs(a):
    """返回 [(image_path, prompt), ...]"""
    pairs = []
    if a.metadata:
        with open(a.metadata, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                img = r.get('image') or r.get('src')
                if not img:
                    continue
                if not os.path.isabs(img):
                    # clean_pipeline 写的是相对路径，相对仓库根解析
                    cand = os.path.join(_REPO, img)
                    img = cand if os.path.exists(cand) else os.path.abspath(img)
                if not os.path.exists(img):
                    continue
                pairs.append((img, r.get('prompt') or r.get('caption') or a.prompt or ''))
        return pairs

    imgs = list_images(a.images)
    if a.limit:
        imgs = imgs[:a.limit]
    pmap = load_prompts(a.prompts) if a.prompts else {}
    for p in imgs:
        pr = pmap.get(os.path.basename(p)) or a.prompt or ''
        pairs.append((p, pr))
    return pairs


# --------------------------- 代理分数（无 CLIP 兜底） ---------------------------
def proxy_score(prompt, path, size=256):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    im = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    r, g, b = x[0], x[1], x[2]
    feat = dict(
        skin=float(((r > 0.45) & (g > 0.25) & (b > 0.15) & (r > g * 1.05) & (r > b * 1.15)).float().mean()),
        green=float(((g > r * 1.10) & (g > b * 1.10)).float().mean()),
        blue=float(((b > r * 1.05) & (b > g * 1.05)).float().mean()),
        warm=float(((r > g * 1.08) & (r > b * 1.15)).float().mean()),
    )
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    feat['edge'] = float(torch.sqrt(F.conv2d(gray[None, None], k, padding=1) ** 2 + 1e-8).mean())
    feat['clarity'] = float(gray.std())
    p = prompt.lower()
    s = w = 0.0

    def add(kw, val, weight):
        nonlocal s, w
        if kw in p:
            s += val * weight; w += weight

    for kw in ('1girl', 'girl', 'woman', 'person', 'solo', 'face', 'portrait', 'close-up', 'upper body', 'full body'):
        add(kw, feat['skin'], 1.0)
    for kw in ('forest', 'tree', 'grass', 'nature'):
        add(kw, feat['green'], 1.0)
    for kw in ('sky', 'ocean', 'water', 'blue'):
        add(kw, feat['blue'], 1.0)
    add('detailed', feat['clarity'] * 3, 1.0)
    if w == 0:
        return 0.0
    return max(0.0, min(1.0, s / w)) * 100.0


# --------------------------- 真 CLIP ---------------------------
def _feat(out):
    """兼容不同 transformers 版本：可能是 Tensor，也可能是带 image_embeds/text_embeds 的输出对象。"""
    if hasattr(out, 'image_embeds') and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, 'text_embeds') and out.text_embeds is not None:
        return out.text_embeds
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def clip_feat(model, out, kind):
    """取 CLIP 对比空间里的嵌入，兼容 transformers 5.x。

    ⚠️ 为什么需要、以及为什么【不能再投影一次】（2026-09-19 实测）：

    (1) transformers 5.x 的 `CLIPModel.get_*_features()` **不再返回张量**，
        而是返回 `BaseModelOutputWithPooling`（只有 last_hidden_state / pooler_output）：
            out type: BaseModelOutputWithPooling
            last_hidden_state (1, 7, 512)
            pooler_output     (1, 512)      <- 没有 text_embeds / image_embeds
        ⇒ 旧代码 `model.get_text_features(**inp).float()` 直接抛
          `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'float'`，
          而 `train_ar_v2.load_caption_cond` 把它 try/except 吞掉后返回 None
          ⇒ **条件训练静默退化成无条件**（"条件注入太弱"的真正原因）。

    (2) **不要再过一遍投影层**。我一开始想当然地写
        `text_projection(pooler_output)`，结果匹配对的余弦只有 **0.0434**；
        直接取 `pooler_output` 是 **0.2888**（CLIP ViT-B/32 的正常量级）。
        实测（probe_clip_variants.py，同一张图 vs 匹配文本）：
            pool+pool          0.2888   ✅
            proj(pool)+pool    0.0434   ❌ 重复投影
            lhs[:,0]+pool     -0.0460
            proj(lhs.mean)+pool 0.2559
        ⇒ **这个版本返回的 pooler_output 已经是投影后的嵌入**。

    kind 仅用于在存在 text_embeds/image_embeds 的旧版本里优先取对应字段。
    """
    # ⚠️ 本模块【不在顶层 import torch】（CLIP 是可选依赖），所以这里必须惰性导入。
    import torch
    if torch.is_tensor(out):
        return out
    name = 'text_embeds' if kind == 'text' else 'image_embeds'
    v = getattr(out, name, None)
    if torch.is_tensor(v):
        return v
    pooled = getattr(out, 'pooler_output', None)
    if pooled is not None:
        return pooled                # ★ 实测：已经是投影后的嵌入，不要再投影
    if isinstance(out, (tuple, list)) and len(out) and torch.is_tensor(out[0]):
        return out[0]
    return out


def find_local_snapshot(model_id):
    """在 <repo>/.hf_home/hub 下找模型的本地 snapshot 目录，找不到返回 None。

    直接按路径加载比依赖 HF_HOME/HF_HUB_OFFLINE 环境变量更可靠
    （transformers 4.38 里离线标志的读取时机容易踩坑）。
    """
    hub = os.path.join(_REPO, '.hf_home', 'hub')
    d = os.path.join(hub, 'models--' + model_id.replace('/', '--'), 'snapshots')
    if not os.path.isdir(d):
        return None
    snaps = [os.path.join(d, x) for x in os.listdir(d)]
    snaps = [s for s in snaps if os.path.isfile(os.path.join(s, 'config.json'))]
    return max(snaps, key=os.path.getmtime) if snaps else None


def load_clip(model_id, offline=True):
    """优先从本地 snapshot 直接加载，失败再退回 HF 联网（默认 hf-mirror 镜像）。"""
    from transformers import CLIPModel, CLIPProcessor
    src = None
    if offline:
        src = find_local_snapshot(model_id)
        if src:
            try:
                m = CLIPModel.from_pretrained(src, local_files_only=True).eval()
                p = CLIPProcessor.from_pretrained(src, local_files_only=True)
                print(f'[clip_eval] loaded from local snapshot: {src}', flush=True)
                return m, p
            except Exception as e:
                print(f'[clip_eval] local snapshot load failed ({type(e).__name__}: {e}) -> online', flush=True)
        else:
            print('[clip_eval] no local snapshot found -> online', flush=True)
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    m = CLIPModel.from_pretrained(model_id).eval()
    p = CLIPProcessor.from_pretrained(model_id)
    return m, p


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--images', help='图片目录（递归）')
    src.add_argument('--metadata', help='metadata.jsonl（清洗产出）')
    ap.add_argument('--prompts', help='prompts.json：{文件名: prompt} 或 {文件名: {prompt: ...}}')
    ap.add_argument('--prompt', default='', help='统一提示词（未在 prompts.json 中命中时使用）')
    ap.add_argument('--out', default='', help='输出 jsonl（默认 <images>/clip_scores.jsonl）')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--model', default='openai/clip-vit-base-patch32')
    ap.add_argument('--template', default='none', choices=['none', 'photo'],
                    help="none=原样送 CLIP；photo=包一层 'a photo of {}'（对自然图更准）")
    ap.add_argument('--no-clip', dest='use_clip', action='store_false', help='强制走代理分数')
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)

    pairs = build_pairs(a)
    if not pairs:
        raise SystemExit('no (image, prompt) pairs found')
    print(f'[clip_eval] pairs={len(pairs)}', flush=True)

    mode = 'proxy'
    model = proc = None
    if a.use_clip:
        t = time.perf_counter()
        try:
            model, proc = load_clip(a.model)
            mode = 'clip'
            print(f'[clip_eval] real CLIP ready ({a.model}) in {time.perf_counter()-t:.1f}s', flush=True)
        except Exception as e:
            print(f'[clip_eval] CLIP unavailable ({type(e).__name__}: {e}) -> proxy fallback', flush=True)

    from PIL import Image
    import torch.nn.functional as F

    def wrap(p):
        return f'a photo of {p}' if a.template == 'photo' and p else p

    results = []
    t0 = time.perf_counter()
    for s in range(0, len(pairs), a.batch):
        chunk = pairs[s:s + a.batch]
        imgs, ok = [], []
        for p, _ in chunk:
            try:
                imgs.append(Image.open(p).convert('RGB')); ok.append(True)
            except Exception:
                imgs.append(Image.new('RGB', (224, 224))); ok.append(False)
        texts = [wrap(pr) or 'an image' for _, pr in chunk]
        if mode == 'clip':
            with torch.no_grad():
                ii = proc(images=imgs, return_tensors='pt')
                ti = proc(text=texts, return_tensors='pt', padding=True, truncation=True, max_length=77)
                imf = F.normalize(_feat(model.get_image_features(**ii)).float(), dim=-1)
                txf = F.normalize(_feat(model.get_text_features(**ti)).float(), dim=-1)
                sims = (imf * txf).sum(dim=-1)
            scores = [float(x) * 100.0 for x in sims]
        else:
            scores = [proxy_score(pr, p) if ok_ else 0.0 for (p, pr), ok_ in zip(chunk, ok)]
        for (p, pr), sc, ok_ in zip(chunk, scores, ok):
            results.append(dict(image=p, prompt=pr, clip_score=round(sc, 4),
                                mode=mode, ok=ok_))
        if (s // a.batch) % 10 == 0:
            el = time.perf_counter() - t0
            print(f'[clip_eval] {min(s+a.batch,len(pairs))}/{len(pairs)} {el:.1f}s', flush=True)

    dt = time.perf_counter() - t0
    vals = [r['clip_score'] for r in results if r['ok']]
    vals_sorted = sorted(vals)

    def pct(q):
        return vals_sorted[min(len(vals_sorted) - 1, int(q * len(vals_sorted)))] if vals_sorted else 0.0

    # 按 prompt 分组：能看出哪组 caption 没对齐（多图共享同一 prompt 时）
    groups = {}
    for r in results:
        if r['ok']:
            groups.setdefault(r['prompt'], []).append(r['clip_score'])
    by_prompt = sorted((dict(prompt=k, n=len(v), mean=round(sum(v) / len(v), 3))
                        for k, v in groups.items()), key=lambda x: x['mean'])

    outpath = a.out or (os.path.join(a.images, 'clip_scores.jsonl') if a.images
                        else os.path.join(os.path.dirname(a.metadata), 'clip_scores.jsonl'))
    with open(outpath, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    report = dict(mode=mode, model=a.model if mode == 'clip' else None,
                  n=len(results), n_ok=len(vals),
                  mean=round(sum(vals) / len(vals), 3) if vals else None,
                  median=round(statistics.median(vals), 3) if vals else None,
                  p10=round(pct(0.10), 3) if vals else None,
                  p90=round(pct(0.90), 3) if vals else None,
                  min=round(min(vals), 3) if vals else None,
                  max=round(max(vals), 3) if vals else None,
                  elapsed_s=round(dt, 2),
                  ms_per_img=round(dt / max(1, len(results)) * 1000, 1),
                  n_prompt_groups=len(by_prompt),
                  worst_prompts=by_prompt[:5], best_prompts=by_prompt[-5:],
                  out=outpath)
    rp = os.path.splitext(outpath)[0] + '_report.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 文本直方图（matplotlib 本环境没装，用 ASCII 也能看分布）
    if vals:
        lo, hi = min(vals), max(vals)
        nb = 20
        w = (hi - lo) / nb or 1.0
        bins = [0] * nb
        for v in vals:
            bins[min(nb - 1, int((v - lo) / w))] += 1
        mx = max(bins)
        print('[clip_eval] 分数分布')
        for i, c in enumerate(bins):
            print(f'  {lo+i*w:7.2f} |{"#" * int(40*c/mx)}{" " * (40-int(40*c/mx))}| {c}')
    print('[clip_eval] ' + json.dumps({k: v for k, v in report.items()
                                       if k not in ('worst_prompts', 'best_prompts')},
                                      ensure_ascii=False), flush=True)
    print(f'[clip_eval] report -> {rp}', flush=True)


if __name__ == '__main__':
    main()
