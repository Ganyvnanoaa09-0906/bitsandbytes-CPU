"""Wan2.1 UMT5-XXL encoder conversion + prompt encoding smoke."""
from __future__ import annotations

import os
import re
import sys

import torch
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

P = r'D:\work\textmodel\Wan2.1-T2V-1.3B\models_t5_umt5-xxl-enc-bf16.pth'
TOK = r'D:\work\textmodel\Wan2.1-T2V-1.3B\google\umt5-xxl'


def build_model():
    cfg = UMT5Config(vocab_size=256384, d_model=4096, d_kv=64, d_ff=10240,
                     num_layers=24, num_heads=64, relative_attention_num_buckets=32,
                     dropout_rate=0.0, layer_norm_epsilon=1e-6,
                     feed_forward_proj='gated-gelu')
    return UMT5EncoderModel._from_config(cfg, torch_dtype=torch.bfloat16)


def map_key(k):
    m = re.match(r'^blocks\.(\d+)\.(.*)$', k)
    if m:
        i, rest = m.group(1), m.group(2)
        rest = rest.replace('norm1.', 'layer.0.layer_norm.')
        rest = rest.replace('norm2.', 'layer.1.layer_norm.')
        rest = rest.replace('attn.q.', 'layer.0.SelfAttention.q.')
        rest = rest.replace('attn.k.', 'layer.0.SelfAttention.k.')
        rest = rest.replace('attn.v.', 'layer.0.SelfAttention.v.')
        rest = rest.replace('attn.o.', 'layer.0.SelfAttention.o.')
        rest = rest.replace('pos_embedding.embedding.', 'layer.0.SelfAttention.relative_attention_bias.')
        rest = rest.replace('ffn.gate.0.', 'layer.1.DenseReluDense.wi_0.')
        rest = rest.replace('ffn.fc1.', 'layer.1.DenseReluDense.wi_1.')
        rest = rest.replace('ffn.fc2.', 'layer.1.DenseReluDense.wo.')
        return f'encoder.block.{i}.{rest}'
    if k == 'token_embedding.weight':
        return 'shared.weight'
    if k == 'norm.weight':
        return 'encoder.final_layer_norm.weight'
    return None


def load_model(verbose=True):
    sd = torch.load(P, map_location='cpu', weights_only=True, mmap=True)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    model = build_model()
    ref = model.state_dict()
    new = {}
    bad = []
    for k, v in sd.items():
        nk = map_key(k)
        if nk is None:
            bad.append(k)
            continue
        new[nk] = v.to(torch.bfloat16) if v.dtype != torch.bfloat16 else v
    # tied embedding alias (transformers keeps both names)
    if 'shared.weight' in new:
        new['encoder.embed_tokens.weight'] = new['shared.weight']
    missing = [k for k in ref if k not in new]
    unexpected = [k for k in new if k not in ref]
    shape = [(k, tuple(new[k].shape), tuple(ref[k].shape)) for k in new if k in ref and tuple(new[k].shape) != tuple(ref[k].shape)]
    if verbose:
        print('T5 mapped %d/%d missing %d unexpected %d shape %d unmapped %d'
              % (len(new), len(sd), len(missing), len(unexpected), len(shape), len(bad)), flush=True)
        for k in missing[:5]:
            print(' missing', k)
        for k in unexpected[:5]:
            print(' unexpected', k)
        for k in bad[:5]:
            print(' unmapped', k)
    model.load_state_dict(new, strict=True)
    model.eval()
    del sd, new
    return model


def main():
    torch.set_num_threads(6)
    model = load_model()
    tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True)
    text = 'a cat walking on grass'
    enc = tok([text], padding='max_length', max_length=64, truncation=True,
              return_tensors='pt')
    with torch.no_grad():
        out = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
    h = out.last_hidden_state
    print('prompt encoding', tuple(h.shape), 'finite', bool(torch.isfinite(h).all()),
          'mean %.4f std %.4f' % (float(h.float().mean()), float(h.float().std())))


if __name__ == '__main__':
    main()
