"""grpo_cpu_deepseek.py — minimal CPU GRPO with a DeepSeek API judge.

Policy: Qwen3-1.7B + LoRA (base frozen), rollout locally on CPU.
Reward: one DeepSeek chat-completion call per completion, strict JSON score 0-10.
Reference/KL: disabled (beta=0) for this first smoke; GRPO group-relative advantage only.

The first version ran a full backward through 1.7B even when all G rewards were identical
(zero advantage), which wastes minutes. This version:
  - skips the update when advantages are all ~0;
  - does ONE forward to get logprobs (old = detached current), so ratio starts at 1;
  - prints generation / judge / forward / backward timings separately;
  - has --no-update for a pure rollout+judge pass.

Usage:
  python -u gt1030/grpo_cpu_deepseek.py --steps 1 --group 2 --max-new 12 --no-update
  python -u gt1030/grpo_cpu_deepseek.py --steps 2 --group 2 --max-new 12
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

CRED = os.path.expanduser(r'~\.dsh\.credentials.yaml')
MODEL = r'D:\work\llm_env\Qwen3-1.7B'
OUT_DIR = r'D:\work\research\bench_results\grpo_cpu'

PROMPTS = [
    ('What is 17*23? Answer with a single integer.', '391'),
    ('What is 48+37? Answer with a single integer.', '85'),
    ('What is 96/8? Answer with a single integer.', '12'),
    ('What is 14*14? Answer with a single integer.', '196'),
    ('What is 123-57? Answer with a single integer.', '66'),
    ('What is 9*9+19? Answer with a single integer.', '100'),
    ('What is 987*654? Answer with a single integer.', '645498'),
    ('What is the 20th prime number? Answer with a single integer.', '71'),
    ('If a train travels 60 km/h for 2.5 hours, how many km does it travel? Answer with a single integer.', '150'),
    ('What is 12345+67890? Answer with a single integer.', '80235'),
]


def get_key() -> str:
    v = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if v:
        return v
    import yaml
    d = yaml.safe_load(io.open(CRED, encoding='utf-8'))
    v = str(d.get('refs', {}).get('DEEPSEEK_API_KEY', '')).strip()
    if not v:
        raise RuntimeError('no DEEPSEEK_API_KEY')
    return v


JUDGE_SYSTEM = (
    'You are a strict, deterministic grader for a math RL environment. '
    'Return JSON only: {"score": <integer 0-10>, "reason": "<short>"}. '
    '10 = final answer exactly correct and format follows the instruction; '
    '7-9 = correct answer but extra text/format violation; '
    '3-6 = wrong final answer with some valid reasoning; 0-2 = empty or irrelevant. '
    'Judge the final answer, not whether the reasoning looks nice.'
)


def judge_one(session, base, model, question, reference, answer, key):
    user = (f'Question:\n{question}\n\nReference answer:\n{reference}\n\n'
            f'Model answer:\n{answer}\n\nGrade the model answer.')
    body = {
        'model': model,
        'messages': [{'role': 'system', 'content': JUDGE_SYSTEM},
                     {'role': 'user', 'content': user}],
        'temperature': 0.0, 'max_tokens': 80,
        'response_format': {'type': 'json_object'},
    }
    try:
        r = session.post(base + '/chat/completions', headers={
            'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
            json=body, timeout=60)
        if r.status_code != 200:
            return 0.0, f'HTTP {r.status_code}: {r.text[:120]}'
        content = r.json()['choices'][0]['message']['content']
        obj = json.loads(content)
        return max(0.0, min(10.0, float(obj.get('score', 0)))) / 10.0, str(obj.get('reason', ''))[:160]
    except Exception as e:
        return 0.0, f'{type(e).__name__}: {e}'


def truncated_logprobs(model, full, prompt_len, gen_len, k):
    """Logprobs with gradients only through the last k transformer layers + head.

    This is the CPU-feasible policy class: early layers run under no_grad and their
    hidden states are detached, so backward never traverses the full 1.7B graph.
    PEFT adapters in the last k layers (and lm_head) are the trainable policy.
    """
    peft_base = model.base_model.model
    core = peft_base.model
    B, T = full.shape
    position_ids = torch.arange(T, device=full.device).unsqueeze(0).expand(B, -1)
    # No padding: let Qwen3 use its default causal SDPA mask. Passing a raw 2D
    # attention_mask here bypasses the model's create_causal_mask() and breaks SDPA.
    attn = None
    with torch.no_grad():
        h = core.embed_tokens(full)
        pos = core.rotary_emb(h, position_ids)
        if os.environ.get('GRPO_DEBUG'):
            print('DEBUG h', tuple(h.shape), 'pos0', tuple(pos[0].shape),
                  'pos1', tuple(pos[1].shape), flush=True)
        for ei, layer in enumerate(core.layers[:-k]):
            if os.environ.get('GRPO_DEBUG') and (ei == 0 or ei == 24):
                print('DEBUG early', ei, 'in h', tuple(h.shape), flush=True)
            h = layer(h, attention_mask=attn, position_ids=position_ids,
                      position_embeddings=pos, use_cache=False)
            if os.environ.get('GRPO_DEBUG') and (ei == 0 or ei == 24):
                print('DEBUG early', ei, 'out h', tuple(h.shape), flush=True)
    h = h.detach().requires_grad_(True)
    for li, layer in enumerate(core.layers[-k:]):
        if os.environ.get('GRPO_DEBUG'):
            print('DEBUG last layer', li, 'in h', tuple(h.shape), flush=True)
        h = layer(h, attention_mask=attn, position_ids=position_ids,
                  position_embeddings=pos, use_cache=False)
        if os.environ.get('GRPO_DEBUG'):
            print('DEBUG last layer', li, 'out h', tuple(h.shape), flush=True)
    h = core.norm(h)
    logits = peft_base.lm_head(h).float()
    logits = logits[:, :-1, :]
    labels = full[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, prompt_len - 1:prompt_len - 1 + gen_len] = True
    return (tok_lp * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--group', type=int, default=2)
    ap.add_argument('--max-new', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--judge-model', default=os.environ.get('DEEPSEEK_JUDGE_MODEL', 'deepseek-chat'))
    ap.add_argument('--base-url', default=os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'))
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--no-update', action='store_true')
    ap.add_argument('--train-last-layers', type=int, default=2)
    ap.add_argument('--synthetic-adv', default='')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    key = get_key()
    base = args.base_url.rstrip('/')
    log_path = os.path.join(OUT_DIR, f'grpo_g{args.group}_t{args.max_new}.jsonl')
    logf = io.open(log_path, 'w', encoding='utf-8')

    print('loading tokenizer/model...', flush=True)
    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map='cpu',
        low_cpu_mem_usage=True, local_files_only=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    print('model ready in %.1fs; params=%d trainable=%d' % (
        time.perf_counter() - t_load,
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in params)), flush=True)

    session = requests.Session()
    for step in range(1, args.steps + 1):
        prompt_item = PROMPTS[(step - 1) % len(PROMPTS)]
        q, ref = prompt_item
        text = f'Question: {q}\nAnswer:'
        enc = tok(text, return_tensors='pt')
        prompt_ids = enc.input_ids.repeat(args.group, 1)
        prompt_len = prompt_ids.shape[1]
        t0 = time.perf_counter()
        model.config.use_cache = True
        with torch.no_grad():
            gen = model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=args.max_new,
                do_sample=True, temperature=1.0, top_p=1.0,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        model.config.use_cache = False
        gen_ids = gen[:, prompt_len:]
        gen_mask = torch.ones_like(gen_ids)
        answers = [tok.decode(g, skip_special_tokens=True).strip() for g in gen_ids]
        t_gen = time.perf_counter() - t0

        t1 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.group) as ex:
            judged = list(ex.map(lambda a: judge_one(session, base, args.judge_model, q, ref, a, key), answers))
        t_judge = time.perf_counter() - t1
        rewards = torch.tensor([j[0] for j in judged], dtype=torch.float32)
        if args.synthetic_adv:
            adv = torch.tensor([float(x) for x in args.synthetic_adv.split(',')], dtype=torch.float32)
            adv = adv[:len(rewards)]
        elif rewards.std() > 1e-8:
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            adv = rewards - rewards.mean()
        rec = {
            'step': step, 'question': q, 'reference': ref, 'answers': answers,
            'rewards': rewards.tolist(), 'advantages': adv.tolist(),
            'gen_s': round(t_gen, 2), 'judge_s': round(t_judge, 2),
            'reasons': [j[1] for j in judged],
        }
        print('step %d gen %.1fs judge %.1fs rewards=%s' % (
            step, t_gen, t_judge, rewards.tolist()), flush=True)
        if args.no_update:
            rec['skipped'] = 'no_update'
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            logf.write(json.dumps(rec, ensure_ascii=False) + '\n'); logf.flush()
            continue
        if adv.abs().sum() < 1e-8:
            rec['skipped'] = 'zero_advantage'
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            logf.write(json.dumps(rec, ensure_ascii=False) + '\n'); logf.flush()
            continue

        t2 = time.perf_counter()
        full = torch.cat([prompt_ids, gen_ids], dim=1)
        if args.train_last_layers > 0:
            new = truncated_logprobs(model, full, prompt_len, gen_ids.shape[1],
                                     args.train_last_layers)
        else:
            out = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False)
            logits = out.logits[:, :-1, :].float()
            labels = full[:, 1:]
            logp = F.log_softmax(logits, dim=-1)
            tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[:, prompt_len - 1:prompt_len - 1 + gen_ids.shape[1]] = gen_mask.bool()
            new = (tok_lp * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        t_fwd = time.perf_counter() - t2
        old = new.detach()
        ratio = torch.exp(new - old)
        surr = torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
        loss = -surr.mean()
        t3 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        t_bwd = time.perf_counter() - t3
        rec.update({'loss': float(loss.detach()), 'fwd_s': round(t_fwd, 2),
                    'bwd_s': round(t_bwd, 2), 'total_s': round(time.perf_counter() - t0, 2)})
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        logf.write(json.dumps(rec, ensure_ascii=False) + '\n'); logf.flush()
    logf.close()
    print('done ->', log_path, flush=True)


if __name__ == '__main__':
    main()
