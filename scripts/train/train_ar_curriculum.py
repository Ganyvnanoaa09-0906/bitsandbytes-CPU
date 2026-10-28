# -*- coding: utf-8 -*-
"""train_ar_curriculum.py - 渐进序列长度训练：64 -> 128 -> 256 token。

不引入 MoE，只改训练课程；利用短序列降低 GEMM 计算量。
阶段格式：--stages "64:1000,128:1000,256:3000"

支持 --masks：屏蔽马赛克区域的 token（见 mosaic_mask.py）。
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, json, os, sys, time, threading
import numpy as np
import torch, torch.nn.functional as F, psutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitsandbytes"))
import bitsandbytes as bnb

def rss(): return psutil.Process().memory_info().rss/1e9
class Peak:
    def __init__(self): self.p=rss(); self.stop=threading.Event(); self.t=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set(): self.p=max(self.p,rss()); time.sleep(0.02)
    def start(self): self.t.start()
    def done(self): self.stop.set(); self.t.join(0.2); return self.p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stages', default='64:1000,128:1000,256:3000')
    ap.add_argument('--batch', default='1',
                    help='batch；单个整数则所有阶段共用，或分阶段指定如 "16:64,64:16,256:8"。'
                         '实测吞吐：T=16 b64=763 tok/s(b1 只有 56)；T=64 b16=607(b1 192)；'
                         'T=256 b8=624(b1 481)。短序列必须加大 batch 否则 CPU 喂不饱。')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--d_model', type=int, default=1280)
    ap.add_argument('--n_layer', type=int, default=16)
    ap.add_argument('--n_head', type=int, default=16)
    ap.add_argument('--ffn_dim', type=int, default=7168)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--ckpt_every', type=int, default=500)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--log_every', type=int, default=10, help='每 N 步写一条 train.jsonl（loss 曲线用）')
    ap.add_argument('--no_ckpt', action='store_true',
                    help='完全不写 checkpoint（404M 模型单个 ckpt ~1.5GB，磁盘紧张时用）')
    ap.add_argument('--masks', default='',
                    help='mosaic_masks.npz（mosaic_mask.py 产出）。提供后码区 token 不参与 loss，'
                         '马赛克被彻底中和而不是稀释。')
    a=ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads); torch.backends.mkldnn.enabled=True
    os.makedirs(a.out, exist_ok=True)
    tokens=torch.load(a.tokens, map_location='cpu')
    if tokens.ndim!=2: tokens=tokens.reshape(tokens.shape[0],-1)
    N=tokens.shape[0]
    masks=None
    if a.masks:
        z=np.load(a.masks, allow_pickle=False)
        m=z['masks']                                     # (M,16,16) bool
        M=m.shape[0]
        if M!=N:
            raise SystemExit(f'[masks] 张数不匹配: masks={M} tokens={N}（两边必须来自同一目录、同一顺序）')
        masks=torch.from_numpy(m.reshape(M,-1).astype(np.bool_))   # (N,256)
        print(f'[curriculum] masks loaded: {tuple(m.shape)}  屏蔽率 {masks.float().mean()*100:.2f}%',flush=True)
    cfg=SmallImageConfig(d_model=a.d_model,n_layer=a.n_layer,n_head=a.n_head,ffn_dim=a.ffn_dim,total_tokens=256)
    model=SmallARImageModel(cfg)
    print(f'[curriculum] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M tokens={tuple(tokens.shape)}',flush=True)
    opt=bnb.optim.AdamW8bit(model.parameters(), lr=a.lr)
    # 解析 batch 规格：单值 -> 所有阶段共用；"16:64,64:16,256:8" -> 分阶段
    bspec=[x for x in str(a.batch).split(',') if x.strip()]
    bmap={}
    for x in bspec:
        if ':' in x:
            k,v=x.split(':'); bmap[int(k)]=int(v)
        else:
            bmap[None]=int(x)
    def batch_for(T):
        return bmap.get(T, bmap.get(None, 1))
    peak=Peak(); peak.start()
    stage_stats=[]
    global_step=0
    log=open(os.path.join(a.out,'train.jsonl'),'w',encoding='utf-8')
    for stage in a.stages.split(','):
        T,steps=stage.split(':'); T=int(T); steps=int(steps)
        B=batch_for(T)
        text=torch.zeros(B,cfg.text_dim)
        print(f'[curriculum] stage T={T} steps={steps} batch={B}',flush=True)
        ts=[]; losses=[]; t0=time.time()
        for i in range(steps):
            idx=torch.randint(0,N,(B,)); x=tokens[idx,:T].long()
            t1=time.perf_counter()
            th,ge=model(x,text)
            # 自回归必须把目标**移位一位**：用 x[0..i] 预测 x[i+1]。
            # 旧写法 cross_entropy(ge, x) 的目标是输入本身，位置 i 能看到 x[i]，
            # 于是模型只学「输出=输入」的恒等映射 —— loss 趋 0 但完全无法生成
            # （实测 top-1 等于输入 100%，等于下一个 token 仅 6.9%）。
            tt=min(cfg.think_tokens,T)
            if masks is None:
                gen=F.cross_entropy(ge[:,:-1].reshape(-1,cfg.vocab_size),x[:,1:].reshape(-1))
                think=(F.cross_entropy(th[:,:tt-1].reshape(-1,cfg.vocab_size),x[:,1:tt].reshape(-1))
                       if tt>1 else torch.zeros((),device=x.device))
            else:
                # mosaic_mask.py 里 True = 检测到马赛克 = **要屏蔽**，故参与 loss 的是 keep=~m。
                # 目标移位后，位置 i 预测的是 x[i+1]，所以 mask 也要跟着切到 [1:].
                keep=~masks[idx][:,:T]
                gen_all=F.cross_entropy(ge[:,:-1].reshape(-1,cfg.vocab_size),x[:,1:].reshape(-1),
                                        reduction='none').view(x.shape[0],T-1)
                kg=keep[:,1:]
                gen=(gen_all*kg).sum()/kg.sum().clamp(min=1.0)
                if tt>1:
                    th_all=F.cross_entropy(th[:,:tt-1].reshape(-1,cfg.vocab_size),x[:,1:tt].reshape(-1),
                                           reduction='none').view(x.shape[0],tt-1)
                    kt=keep[:,1:tt]
                    think=(th_all*kt).sum()/kt.sum().clamp(min=1.0)
                else:
                    think=torch.zeros((),device=x.device)
            loss=gen+0.1*think
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ts.append(time.perf_counter()-t1); losses.append(float(loss.detach())); global_step+=1
            if i % a.log_every == 0 or i == steps-1:
                log.write(json.dumps(dict(step=global_step, T=T, stage_step=i+1,
                                          loss=losses[-1], gen=float(gen.detach()), think=float(think.detach()),
                                          step_s=ts[-1], elapsed_s=time.time()-t0, rss_gb=rss()),
                                     ensure_ascii=False)+'\n'); log.flush()
            if global_step % a.ckpt_every == 0 and not a.no_ckpt:
                torch.save(dict(model=model.state_dict(),step=global_step,cfg=cfg.__dict__,T=T),os.path.join(a.out,f'ckpt_{global_step}.pt'))
            if i % max(1,steps//5) == 0 or i==steps-1:
                print(f'[curriculum T={T}] {i+1}/{steps} loss={losses[-1]:.4f} {ts[-1]:.3f}s',flush=True)
        rec=dict(T=T,steps=steps,batch=B,mean_step_s=sum(ts)/len(ts),first_loss=losses[0],last_loss=losses[-1],
                 elapsed_s=time.time()-t0,
                 tokens_per_s=B*T/max(1e-9,sum(ts)/len(ts)))
        stage_stats.append(rec); print(json.dumps(rec,ensure_ascii=False),flush=True)
    log.close()
    if not a.no_ckpt:
        torch.save(dict(model=model.state_dict(),step=global_step,cfg=cfg.__dict__),os.path.join(a.out,'ckpt_last.pt'))
    summary=dict(kind='ar_curriculum',params=sum(p.numel() for p in model.parameters()),
                 stages=stage_stats,peak_rss_gb=float(peak.done()))
    with open(os.path.join(a.out,'summary.json'),'w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()