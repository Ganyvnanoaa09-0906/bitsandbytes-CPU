# -*- coding: utf-8 -*-
"""small_vqvae.py - R5 可训练轻量 VQ-VAE（256x256 -> 16x16 token）。

用途：为小 AR 模型提供冻结的离散 tokenizer。先在安全本地图上冒烟/训练。
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
import argparse, json, os, sys, time, threading
import torch
import torch.nn
import torch.nn.functional
import psutil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitsandbytes"))
import bitsandbytes

class ImageFolderLite(Dataset):
    def __init__(self, root, size=256, max_items=0):
        self.paths=[]
        exts=('.jpg','.jpeg','.png','.webp','.bmp')
        for dp,_,fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(exts): self.paths.append(os.path.join(dp,fn))
        self.paths=sorted(self.paths)
        if max_items: self.paths=self.paths[:max_items]
        self.tf=transforms.Compose([transforms.Resize((size,size)), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        # 必须用 with：Image.open 是惰性的，不关会持续累积文件句柄与缓冲区
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert('RGB'))
class ResBlock(nn.Module):
    """残差块。

    原版第二个卷积是 1x1，表达力弱于标准 ResBlock 的 3x3。
    two_3x3=True 时用两个 3x3（默认 False 以保持旧 checkpoint 兼容）。
    """

    def __init__(self, ch, two_3x3=False):
        super().__init__()
        if two_3x3:
            self.net = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, 1, 1),
                                     nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, 1, 1))
        else:
            self.net = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, 1, 1),
                                     nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 1))
    def forward(self,x): return x+self.net(x)


class BottleneckAttn(nn.Module):
    """瓶颈处的自注意力（VQGAN 的 attn_resolutions 做法）。

    为什么需要：卷积的感受野有限，16x16 瓶颈处的全局一致性（比如
    "整张图是一个人" vs "上下两半各自是什么"）只能靠注意力建立。
    VQGAN 在同样 f16 设置下就有这一层，我们没有 —— 这是结构缺口。

    位置数 16x16=256，自注意力开销很小。
    """

    def __init__(self, ch, n_head=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.n_head = n_head

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.n_head, C // self.n_head, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]          # (B, nh, hd, HW)
        q = q.permute(0, 1, 3, 2)                          # (B, nh, HW, hd)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        o = F.scaled_dot_product_attention(q, k, v)        # 双向（非因果）
        o = o.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(o)

class Encoder(nn.Module):
    def __init__(self, ch=(64,128,256,256), embed_dim=64, use_attn=False, two_3x3=False):
        super().__init__()
        layers=[nn.Conv2d(3,ch[0],4,2,1)]
        for i in range(len(ch)):
            layers += [ResBlock(ch[i], two_3x3),
                       nn.Conv2d(ch[i], ch[min(i+1,len(ch)-1)], 4,2,1) if i < len(ch)-1 else nn.Identity()]
        # 瓶颈处（最低分辨率）加自注意力：VQGAN 的 attn_resolutions 做法
        if use_attn:
            layers += [BottleneckAttn(ch[-1])]
        self.net=nn.Sequential(*layers)
        self.proj=nn.Conv2d(ch[-1], embed_dim, 1)
    def forward(self,x): return self.proj(self.net(x))

class Decoder(nn.Module):
    def __init__(self, ch=(256,256,128,64), embed_dim=64, use_attn=False, two_3x3=False):
        super().__init__()
        layers=[nn.Conv2d(embed_dim, ch[0],1)]
        # 对称：解码器入口（瓶颈）也加一层注意力
        if use_attn:
            layers += [BottleneckAttn(ch[0])]
        for i in range(len(ch)):
            layers += [ResBlock(ch[i], two_3x3),
                       nn.ConvTranspose2d(ch[i], ch[min(i+1,len(ch)-1)], 4,2,1) if i < len(ch)-1 else nn.ConvTranspose2d(ch[i],3,4,2,1)]
        self.net=nn.Sequential(*layers)
    def forward(self,z): return torch.tanh(self.net(z))

class VQ(nn.Module):
    def __init__(self, n_codes=1024, dim=64, beta=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        self.codebook.weight.requires_grad_(False)
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon
        self.register_buffer("ema_count", torch.zeros(n_codes))
        self.register_buffer("ema_weight", self.codebook.weight.data.clone())

    def quantize_only(self, z):
        """只做量化，返回量化后的潜变量（无 idx、无 loss）。

        给 --perc_stopgrad 用：感知损失走"detach 过的量化潜变量 -> 解码器"，
        这样感知梯度只更新解码器，不会经 VQ 直通回传去推歪编码器和码本。
        """
        B, C, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, C)
        zn = F.normalize(zf, dim=1)
        wn = F.normalize(self.codebook.weight, dim=1)
        idx = (2.0 - 2.0 * (zn @ wn.t())).argmin(1)
        return self.codebook(idx).view(B, H, W, C).permute(0, 3, 1, 2)

    def forward(self, z):
        B, C, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, C)
        zn = F.normalize(zf, dim=1)
        wn = F.normalize(self.codebook.weight, dim=1)
        d = 2.0 - 2.0 * (zn @ wn.t())
        idx = d.argmin(1)
        quant = self.codebook(idx).view(B, H, W, C).permute(0, 3, 1, 2)
        commit = F.mse_loss(z, quant.detach())
        if self.training:
            one_hot = F.one_hot(idx, self.codebook.num_embeddings).float()
            count = one_hot.sum(0)
            weight = one_hot.t() @ zf
            self.ema_count.mul_(self.decay).add_(count, alpha=1.0 - self.decay)
            self.ema_weight.mul_(self.decay).add_(weight, alpha=1.0 - self.decay)
            total = self.ema_count.sum()
            denom = self.ema_count + self.epsilon
            self.codebook.weight.data.copy_(self.ema_weight / denom.unsqueeze(1))
            dead = self.ema_count < 0.5
            if dead.any():
                n_dead = int(dead.sum().item())
                rand = torch.randint(0, zf.shape[0], (n_dead,), device=zf.device)
                self.codebook.weight.data[dead] = zf[rand].clone()
                self.ema_weight[dead] = zf[rand].clone()
                self.ema_count[dead] = 1.0
            quant = self.codebook(idx).view(B, H, W, C).permute(0, 3, 1, 2)
        quant_st = z + (quant - z).detach()
        return quant_st, idx.view(B, H, W), self.beta * commit

class VQVAE(nn.Module):
    def __init__(self, embed_dim=64, n_codes=1024, ch=(64,128,256,256),
                 use_attn=False, two_3x3=False):
        """ch 的长度决定下采样倍率 = 2**len(ch)。
             len=4 -> 16x -> 16x16 网格 = 256 token（原默认，向后兼容）
             len=5 -> 32x ->  8x8 网格 =  64 token
             len=3 ->  8x -> 32x32 网格 = 1024 token
        Decoder 用反转的 ch，保持对称。

        use_attn: 瓶颈处加自注意力（VQGAN 的做法，默认关以兼容旧 ckpt）
        two_3x3:  ResBlock 用两个 3x3 而不是 3x3+1x1
        """
        super().__init__()
        self.use_attn = use_attn
        self.ch = tuple(ch)   # 存档要写它；vqvae_from_ckpt 按它重建
        self.two_3x3 = two_3x3
        self.enc=Encoder(ch=ch, embed_dim=embed_dim, use_attn=use_attn, two_3x3=two_3x3)
        self.vq=VQ(n_codes=n_codes,dim=embed_dim)
        self.dec=Decoder(ch=tuple(reversed(ch)), embed_dim=embed_dim,
                         use_attn=use_attn, two_3x3=two_3x3)
    def forward(self,x):
        z=self.enc(x); q,idx,vq_loss=self.vq(z); xr=self.dec(q)
        return xr, idx, vq_loss


def vqvae_from_ckpt(ckpt, map_location='cpu'):
    """从 checkpoint 读【全部架构字段】重建 VQVAE 并载入权重，返回 (model, ckpt_dict)。

    为什么需要这个函数：项目里有 7 处 `VQVAE(embed_dim=64, n_codes=1024, ch=...)` 的
    硬编码，每次加一个架构参数（ch / embed_dim / use_attn / two_3x3）就要改 7 个地方，
    而且漏改会造成 state_dict 对不上（之前 8x8 的实验就因为 ch 硬编码加载不了）。
    统一走这里，以后加参数只改这一个函数。

    兼容没有新字段的旧 checkpoint（缺字段时取默认值）。
    """
    vk = torch.load(ckpt, map_location=map_location, weights_only=False) if isinstance(ckpt, str) else ckpt
    ch = tuple(vk['ch']) if vk.get('ch') else (64, 128, 256, 256)
    model = VQVAE(embed_dim=int(vk.get('embed_dim', 64)),
                  n_codes=int(vk.get('n_codes', 1024)),
                  ch=ch,
                  use_attn=bool(vk.get('use_attn', False)),
                  two_3x3=bool(vk.get('two_3x3', False)))
    model.load_state_dict(vk['model'])
    model.eval()
    return model, vk


_PROC = psutil.Process()          # 复用同一个对象：原来每次调用都 new，是内存泄漏源


def rss_gb(): return _PROC.memory_info().rss/1e9


class Peak:
    def __init__(self): self.p=rss_gb(); self.stop=threading.Event(); self.t=threading.Thread(target=self.run,daemon=True)
    def run(self):
        # 0.02s -> 0.5s：原来每秒采样 50 次，配合 rss_gb 的 new Process 是泄漏源
        while not self.stop.is_set(): self.p=max(self.p,rss_gb()); time.sleep(0.5)
    def start(self): self.t.start()
    def done(self): self.stop.set(); self.t.join(0.2); return self.p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--embed_dim', type=int, default=64)
    ap.add_argument('--n_codes', type=int, default=1024)
    ap.add_argument('--levels', type=int, default=4,
                    help='ch 的长度 = log2(下采样倍率): 4 -> 16x -> 16x16 网格, '
                         '3 -> 8x -> 32x32 网格（同 size 下 token 数 4 倍，'
                         'tokenizer 成本约 4 倍降为 1 倍）')
    ap.add_argument('--max_items', type=int, default=0)
    ap.add_argument('--ckpt_every', type=int, default=500)
    ap.add_argument('--time_limit_s', type=float, default=0.0)
    ap.add_argument('--threads', type=int, default=6)
    a=ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    ds=ImageFolderLite(a.data, a.size, a.max_items)
    if len(ds)==0: raise SystemExit('no images')
    dl=DataLoader(ds,batch_size=a.batch,shuffle=True,drop_last=True,num_workers=0)
    CH_BY_LEVELS = {4: (64, 128, 256, 256), 3: (64, 128, 256), 5: (64, 128, 256, 256, 256)}
    _ch = CH_BY_LEVELS.get(a.levels)
    if _ch is None:
        raise SystemExit("--levels must be 3, 4 or 5; got %r" % a.levels)
    model=VQVAE(embed_dim=a.embed_dim,n_codes=a.n_codes,ch=_ch)
    print("[vqvae] ch=%s => downsample %dx => grid %dx%d for size %d"
        % (_ch, 2 ** len(_ch), a.size // (2 ** len(_ch)),
        a.size // (2 ** len(_ch)), a.size), flush=True)
    nparam=sum(p.numel() for p in model.parameters())
    print(f'[vqvae] images={len(ds)} params={nparam/1e6:.2f}M batch={a.batch} size={a.size}',flush=True)
    opt=bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    log=open(os.path.join(a.out,'train.jsonl'),'a',encoding='utf-8')
    peak=Peak(); peak.start()
    step=0; t0=time.time()
    it=iter(dl)
    while step < a.steps:
        if a.time_limit_s and time.time()-t0 >= a.time_limit_s:
            print(f"[vqvae] time limit {a.time_limit_s}s reached at step {step}", flush=True)
            break
        try: x=next(it)
        except StopIteration: it=iter(dl); x=next(it)
        ts=time.perf_counter()
        xr,idx,vq_loss=model(x)
        rec=F.mse_loss(xr,x)+0.5*F.l1_loss(xr,x)
        loss=rec+vq_loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step+=1
        if step%20==0:
            rec_val=float(rec.detach()); vq_val=float(vq_loss.detach())
            recd=dict(step=step,loss=float(loss.detach()),rec=rec_val,vq=vq_val,
                      step_s=time.perf_counter()-ts,elapsed_s=time.time()-t0,rss_gb=rss_gb())
            log.write(json.dumps(recd,ensure_ascii=False)+'\n'); log.flush()
            print(f'[vqvae step {step}] loss={recd["loss"]:.4f} rec={rec_val:.4f} vq={vq_val:.4f} {recd["step_s"]:.2f}s rss={recd["rss_gb"]:.2f}GB',flush=True)
        if step%a.ckpt_every==0:
            torch.save(dict(model=model.state_dict(),step=step,ch=tuple(model.ch),embed_dim=a.embed_dim,n_codes=a.n_codes,use_attn=getattr(model,'use_attn',False),two_3x3=getattr(model,'two_3x3',False)),os.path.join(a.out,f'ckpt_{step}.pt'))
            torch.save(dict(model=model.state_dict(),step=step,ch=tuple(model.ch),embed_dim=a.embed_dim,n_codes=a.n_codes,use_attn=getattr(model,'use_attn',False),two_3x3=getattr(model,'two_3x3',False)),os.path.join(a.out,'ckpt_last.pt'))
    torch.save(dict(model=model.state_dict(),step=step,ch=tuple(model.ch),embed_dim=a.embed_dim,n_codes=a.n_codes,use_attn=getattr(model,'use_attn',False),two_3x3=getattr(model,'two_3x3',False)),os.path.join(a.out,'ckpt_last.pt'))
    print(f'[vqvae] done steps={step} peak_rss={peak.done():.2f}GB',flush=True)

if __name__=='__main__': main()
