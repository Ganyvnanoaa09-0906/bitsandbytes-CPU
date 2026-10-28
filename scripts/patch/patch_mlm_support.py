"""Patch the universal trainer to support masked-LM (encoder) models in addition to causal LM.

Auto-detect: try AutoModelForCausalLM; on failure fall back to AutoModelForMaskedLM with synthetic MLM
labels (15% random masking). Everything else (patching, parity check, save, --data) stays the same.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import io

P = r'D:\work\bitsandbytes-CPU\gt1030\igpu_train.py'
t = io.open(P, encoding='utf-8').read()

# ---- 1. data: task-aware ----
old = """def make_data(args, model, tokenizer=None):
    \"\"\"Synthetic fixed batch (default) or random windows from --data text.\"\"\"
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    vocab = getattr(cfg, 'vocab_size', None) or getattr(getattr(cfg, 'text_config', cfg), 'vocab_size', 32000)
    if not args.data:
        g = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, min(vocab, 4000), (args.batch, args.tokens + 1), generator=g)
        return ids[:, :-1], ids[:, 1:]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    text = open(args.data, encoding='utf-8', errors='replace').read()
    toks = tok(text)['input_ids']
    if len(toks) < args.tokens + 2:
        raise SystemExit('data file too short after tokenisation: %d tokens' % len(toks))
    g = np.random.default_rng(1234)
    win = args.tokens + 1
    batch = []
    for _ in range(args.batch):
        s = int(g.integers(0, len(toks) - win))
        batch.append(toks[s:s + win])
    ids = torch.tensor(batch, dtype=torch.long)
    return ids[:, :-1], ids[:, 1:]
"""
new = """def _load_tokenizer(path):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:
        return None


def make_data(args, model):
    \"\"\"Task-aware batch: causal LM (input/target shifted) or masked LM (15% random masks).\"\"\"
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    vocab = getattr(cfg, 'vocab_size', None) or getattr(getattr(cfg, 'text_config', cfg), 'vocab_size', 32000)
    tok = _load_tokenizer(args.model_path) if (args.data or args.task == 'mlm') else None
    if args.data:
        text = open(args.data, encoding='utf-8', errors='replace').read()
        toks = tok(text)['input_ids']
        if len(toks) < args.tokens + 2:
            raise SystemExit('data file too short after tokenisation: %d tokens' % len(toks))
        g = np.random.default_rng(1234)
        win = args.tokens + 1
        seqs = [toks[int(g.integers(0, len(toks) - win)):][:win] for _ in range(args.batch)]
        ids = torch.tensor(seqs, dtype=torch.long)
    else:
        g = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, min(vocab, 4000), (args.batch, args.tokens + 1), generator=g)
    if args.task == 'mlm':
        input_ids = ids[:, :-1].clone()
        labels = input_ids.clone()
        mask_id = getattr(tok, 'mask_token_id', None) if tok is not None else None
        if mask_id is None:
            mask_id = 103
        rng = np.random.default_rng(1234)
        prob = rng.random(input_ids.shape)
        mask_pos = prob < 0.15
        input_ids[mask_pos] = mask_id
        labels[~mask_pos] = -100
        attn = torch.ones_like(input_ids)
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attn}
    return {'input_ids': ids[:, :-1], 'labels': ids[:, 1:]}


def run_loss(model, task, batch):
    if task == 'mlm':
        return model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                     labels=batch['labels']).loss
    return model(batch['input_ids'], labels=batch['labels'], use_cache=False).loss
"""
assert t.count(old) == 1
t = t.replace(old, new, 1)

# ---- 2. model load with task fallback ----
old2 = """    from transformers import AutoModelForCausalLM
    torch.set_num_threads(6)
    path = resolve_model(args.model)
    args.model_path = path
    print('model dir:', path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                                 local_files_only=True)
"""
new2 = """    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
    torch.set_num_threads(6)
    path = resolve_model(args.model)
    args.model_path = path
    print('model dir:', path)
    task = args.task
    model = None
    if task in ('auto', 'clm'):
        try:
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                                         local_files_only=True)
            task = 'clm'
        except Exception as e:
            if args.task == 'clm':
                raise
            print('  causal-LM load failed (%s); trying masked-LM' % str(e)[:80])
    if model is None:
        model = AutoModelForMaskedLM.from_pretrained(path, torch_dtype=torch.float32,
                                                     local_files_only=True)
        task = 'mlm'
    args.task = task
    print('  task:', task)
"""
assert t.count(old2) == 1
t = t.replace(old2, new2, 1)

# ---- 3. batch creation / reference loss / step forward ----
t = t.replace("""    inp, tgt = make_data(args, model)
    # CPU reference on the same batch, before patching: the iGPU first-step loss must match it
    with torch.no_grad():
        try:
            base_loss = float(model(inp, labels=tgt, use_cache=False).loss)
        except Exception:
            base_loss = float('nan')
""", """    batch = make_data(args, model)
    # CPU reference on the same batch, before patching: the iGPU first-step loss must match it
    with torch.no_grad():
        try:
            base_loss = float(run_loss(model, task, batch))
        except Exception:
            base_loss = float('nan')
""")
t = t.replace("        out = model(inp, labels=tgt, use_cache=False)\n",
              "        out = run_loss(model, task, batch)\n")
t = t.replace("        loss = out.loss\n", "        loss = out\n")
# CLI --task
t = t.replace("    ap.add_argument('--mode', choices=['lora', 'full'], default='lora')\n",
              "    ap.add_argument('--mode', choices=['lora', 'full'], default='lora')\n"
              "    ap.add_argument('--task', choices=['auto', 'clm', 'mlm'], default='auto')\n")
io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('masked-LM task support added')
