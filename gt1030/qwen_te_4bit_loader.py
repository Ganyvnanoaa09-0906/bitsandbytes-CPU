"""qwen_te_4bit_loader.py — load the Qwen3VL 8B text encoder from NF4 and encode a prompt.

Only the language-model part of Qwen3VLForConditionalGeneration is needed for
text-to-image; the vision tower (117 quantized linears) and lm_head are left on meta.
The 4bit embed_tokens table is dequantized to bf16 (1.24 GB) because the current CPU
Embedding path is faster that way and the total still fits.

Output: prompt_embeds (1, T, 4096), prompt_embeds_mask (1, T), image_pad_mask (1, T)
saved to a .pt file for the generation probe.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, Params4bit

TE_DIR = r'D:\work\qwen_env\Qwen-Image-2.1-4bit\text_encoder'
FULL = r'D:\work\qwen_env\Qwen-Image-2.1'
SYS = 'Comprehend and analyze the provided prompt.'


def read_q(name, shard_map):
    shard = shard_map[name]
    qs = {}
    with safe_open(TE_DIR + '\\' + shard, framework='pt', device='cpu') as f:
        keys = set(f.keys())
        packed = f.get_tensor(name).clone()
        for suffix in ('absmax', 'nested_absmax', 'nested_quant_map', 'quant_map',
                       'quant_state.bitsandbytes__nf4'):
            k = name + '.' + suffix
            if k in keys:
                qs[suffix] = f.get_tensor(k).clone()
    return packed, qs


def set_submodule(model, name, new):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def load_text_encoder(verbose=True):
    torch.set_num_threads(6)
    cfg = AutoConfig.from_pretrained(TE_DIR, local_files_only=True)
    with init_empty_weights():
        te = Qwen3VLForConditionalGeneration(cfg)
    qmeta = json.load(open(TE_DIR + r'\quant_meta.json'))['layers']
    qmap = json.load(open(TE_DIR + r'\model.safetensors.index.json'))['weight_map']

    # language linears only
    n_done = 0
    for _i, (n, m) in enumerate(list(te.named_modules())):
        if not isinstance(m, torch.nn.Linear):
            continue
        if not n.startswith('model.language_model.'):
            continue
        key = n + '.weight'
        if key not in qmeta:
            continue
        packed, qs = read_q(key, qmap)
        lin = Linear4bit(m.in_features, m.out_features, bias=m.bias is not None,
                         compute_dtype=torch.float32, compress_statistics=True,
                         quant_type='nf4')
        lin.weight = Params4bit.from_prequantized(packed, qs, requires_grad=False,
                                                  device='cpu', module=lin)
        set_submodule(te, n, lin)
        n_done += 1
    print('language linears replaced:', n_done, flush=True)

    # embedding: dequantize 4bit table to bf16
    ekey = 'model.language_model.embed_tokens.weight'
    packed, qs = read_q(ekey, qmap)
    p = Params4bit.from_prequantized(packed, qs, requires_grad=False, device='cpu')
    w = bnb.functional.dequantize_4bit(p.data, p.quant_state).to(torch.bfloat16)
    vocab, hidden = qmeta[ekey]['shape']
    emb = torch.nn.Embedding(vocab, hidden, dtype=torch.bfloat16)
    emb.weight.data = w
    set_submodule(te, 'model.language_model.embed_tokens', emb)
    print('embedding dequantized:', tuple(w.shape), flush=True)

    # non-quant language tensors (te.state_dict keeps the `model.` prefix)
    qkeys = set(qmeta)
    wanted = []
    for k in te.state_dict():
        if not k.startswith('model.language_model.'):
            continue
        if k in qkeys or any(k.startswith(q + '.') for q in qkeys):
            continue
        wanted.append(k)
    by_shard = {}
    for k in wanted:
        if k in qmap:
            by_shard.setdefault(qmap[k], []).append(k)
    sd = {}
    for shard, keys in by_shard.items():
        with safe_open(TE_DIR + '\\' + shard, framework='pt', device='cpu') as f:
            for k in keys:
                sd[k] = f.get_tensor(k).float()
    missing, _ = te.load_state_dict(sd, strict=False, assign=True)
    # report parameter names that are still meta
    left = [n for n, p in te.named_parameters() if p.is_meta and 'visual' not in n
            and 'lm_head' not in n]
    if left:
        print('WARN meta left:', left[:5], flush=True)
    te.eval()
    if verbose:
        import psutil
        print('text encoder loaded | RSS %.2f GB' % (psutil.Process().memory_info().rss / 1e9),
              flush=True)
    return te


def encode_prompt(te, prompt: str):
    import os
    processor = AutoProcessor.from_pretrained(os.path.join(FULL, 'processor'),
                                             local_files_only=True)
    template = (f"<|im_start|>system\n{SYS}<|im_end|>\n"
                f"<|im_start|>user\n{{}}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    text = template.format(prompt if prompt.strip() else ' ')
    inputs = processor(text=[text], padding=True, padding_side='left', return_tensors='pt')
    sys_message = [{"role": "system", "content": [{"type": "text", "text": SYS}]}]
    drop_idx = len(processor.apply_chat_template(sys_message, tokenize=True,
                                                 return_dict=False)[0])
    img_token_id = processor.tokenizer.encode('<|image_pad|>')[0]
    text_model = te.model.language_model
    handle = text_model.norm.register_forward_hook(lambda m, args, out: args[0])
    try:
        with torch.no_grad():
            out = te.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                           output_hidden_states=True, return_dict=True)
    finally:
        handle.remove()
    hidden = out.hidden_states[-1]
    mask = inputs.attention_mask
    bool_mask = mask.bool()
    valid = bool_mask.sum(dim=1)
    selected = hidden[bool_mask]
    parts = torch.split(selected, valid.tolist(), dim=0)
    parts = [p[drop_idx:] for p in parts]
    img_pad = [(inputs.input_ids[i][mask[i].bool()] == img_token_id)
               for i in range(mask.shape[0])]
    img_pad = [e[drop_idx:] for e in img_pad]
    max_len = max(p.shape[0] for p in parts)
    prompt_embeds = torch.stack([torch.cat([p, p.new_zeros(max_len - p.shape[0], p.shape[1])])
                                 for p in parts])
    emb_mask = torch.ones(1, max_len, dtype=torch.long)
    image_pad_mask = torch.stack([torch.cat([p, p.new_zeros(max_len - p.shape[0])])
                                  for p in img_pad])
    print('prompt embeds', tuple(prompt_embeds.shape), '| img_pad tokens',
          int(image_pad_mask.sum()), flush=True)
    return prompt_embeds.float(), emb_mask, image_pad_mask


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='a cat sitting on a chair, photo')
    ap.add_argument('--out', default=r'D:\work\research\bench_results\qwen_prompt_embeds.pt')
    args = ap.parse_args()
    t0 = time.perf_counter()
    te = load_text_encoder()
    emb, m, im = encode_prompt(te, args.prompt)
    torch.save({'prompt_embeds': emb, 'prompt_embeds_mask': m, 'image_pad_mask': im,
                'prompt': args.prompt}, args.out)
    print('saved', args.out, '| %.1f s total' % (time.perf_counter() - t0))
