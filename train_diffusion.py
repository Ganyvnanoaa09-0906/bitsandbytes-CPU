# -*- coding: utf-8 -*-
"""train_diffusion.py — 生图 / 视频统一训练入口（配合 diffusion_backends.py）。

覆盖 D 阶段扩散训练方法（--method）：
    lora / ti / dreambooth / full / controlnet / ip_adapter
    wan_lora / cogvideo_lora / svd_lora / animatediff_lora / video_full

本入口整合 train_sd_lora.py 已验证的工程化经验：
  * fp32 铁律（AVX2 CPU 上 bf16 GEMM 卡死）
  * 线程自动检测（R5-4500U=6 / i5-10400 生图=12）
  * 潜变量落盘缓存（按 VAE 哈希 + 路径 + 分辨率寻址）
  * 文本嵌入磁盘缓存（按 text_encoder 哈希 + prompt 哈希）
  * 冻结层 8bit/NF4 量化、内存/swap 监视、定时采样 + 检查点

当前完成度：
  - sd_denoise（lora/ti/dreambooth/full）        —— 完整训练循环，可实测
  - controlnet / ip_adapter / video               —— 注入就绪，训练循环待 D 阶段后续接入
"""
import argparse
import hashlib
import json
import math          # ← set_lr() 的余弦退火要用；原来没导入（ast.parse 查不出来）
import os
import random
import sys
import time
from collections import OrderedDict

# ============== 环境（必须在 import torch 之前） ==============
def _pick_threads(lm_mode: bool) -> int:
    try:
        import subprocess
        out = subprocess.run(
            ["wmic", "cpu", "get", "NumberOfCores", "NumberOfLogicalProcessors", "/value"],
            capture_output=True, text=True, timeout=15).stdout
        physical = logical = None
        for line in out.splitlines():
            if "NumberOfCores=" in line:
                physical = int(line.split("=")[1])
            if "NumberOfLogicalProcessors=" in line:
                logical = int(line.split("=")[1])
        if physical is not None and logical is not None:
            if not lm_mode:
                # 生图（conv 密集）：用逻辑核，超线程被 conv 的 im2col/分块吃满
                # （实测 i5-10400 12 线程 618 vs 8 线程 489 GFLOP/s，conv 受益超线程）
                return logical
            # LLM（GEMM 密集）：超线程帮不到 GEMM，用逻辑核×2/3（i5-10400 12→8）
            return max(1, int(logical * 2 // 3))
    except Exception:
        pass
    return os.cpu_count() or 6


P = argparse.ArgumentParser(description="diffusion trainer (CPU-only, fp32)")
P.add_argument("--method", required=True,
               choices=["lora", "ti", "dreambooth", "full", "controlnet", "ip_adapter",
                        "wan_lora", "cogvideo_lora", "svd_lora", "animatediff_lora", "video_full"])
P.add_argument("--model", default="bk-sdm-tiny",
               help="tiny-sd | bk-sdm-tiny | bk-sdm-small | sd-v1-5 | 模型绝对路径")
P.add_argument("--data", default="",
               help="数据集目录（需 images/ + prompts.json，或只有 images/；缺省需显式指定）")
P.add_argument("--prompts_file", default="prompts.json",
               help="caption 文件名（在 --data 目录内）。条件式逐图 caption 见 "
                    "build_cond_captions.py -> prompts_cond.json")
P.add_argument("--paired_vqvae", default="",
               help="★配对模式：VQVAE ckpt 路径。给了就把【该图的 VQ 重建】当作 img2img 的"
                    "输入、把【真图】当作回归目标。动机：普通 LoRA 训练是把噪声加在【真图】"
                    "latent 上，推理时却是从【退化图】latent 出发 —— 两者状态分布不同。")
P.add_argument("--paired_tlo", type=int, default=60,
               help="配对模式的时间步下限。低于它 1/sqrt(1-ac) 会把目标放大到病态（t=10 时 28x）")
P.add_argument("--out", default="", help="输出目录")
P.add_argument("--res", type=int, default=256)
P.add_argument("--steps", type=int, default=500)
P.add_argument("--batch", type=int, default=1)
P.add_argument("--grad_accum", type=int, default=4)
P.add_argument("--lr", type=float, default=1e-4)
# LoRA / 通用
P.add_argument("--rank", type=int, default=4)
P.add_argument("--alpha", type=int, default=8)
P.add_argument("--targets", default="to_q,to_v,to_k,to_out.0")
P.add_argument("--lora_dropout", type=float, default=0.1)
P.add_argument("--opt", default="adamw8bit", choices=["adamw8bit", "adafactor", "adamw"])
P.add_argument("--quant", default="none", choices=["none", "8bit", "nf4"])
P.add_argument("--quant_cache", action="store_true")
P.add_argument("--threads", type=int, default=0)
P.add_argument("--max_samples", type=int, default=200)
P.add_argument("--concept", default="nsfw_art")
P.add_argument("--seed", type=int, default=42)
# ★ 2026-09-26 08:0x：**零算力**的数据游标探针（§10.133）。
#   `--order_only` 只创建 DataLoader 并记录前 N 个 batch 的样本身份，然后退出 ——
#   不做任何前向/反向。用来判断「续跑之后拿到的数据顺序是否与一口气跑相同」，
#   而不用跑 12 分钟的真训练。`--order_log` 是落盘路径（JSONL）。
P.add_argument("--order_only", action="store_true",
               help="只 dump 数据顺序然后退出（零算力 oracle，不训练）")
P.add_argument("--order_log", default="", help="数据顺序落盘 JSONL（配合 --order_only）")
# ★ 更啰嗦的诊断（§10.133 抓 `iter(DataLoader)` 消耗全局 RNG 时用的就是它）：
#   额外打印"恢复后 / 取批后 / 画 noise 前 / 画完 noise 后"的全局 RNG 指纹。
#   默认关闭 —— 默认的 `--order_only` 只回答"数据顺序是否一致"这一件事。
P.add_argument("--order_full", action="store_true",
               help="--order_only 时额外打印全局 RNG 三点指纹（诊断用）")
# ★★ 2026-09-26 09:1x：训练侧**带宽探针**（§10.134）。
#   动机（用户："跑一轮训练吧，加探针，看看内存带宽什么情况，有没有优化空间"）：
#   现有剖面 `membw\TRAIN_PROFILE.md` 测的是**分词器/AR 的 30M 模型**（T=256/B=8），
#   不是生成这条线的 SD1.5 UNet + LoRA —— 两个负载不能互相搬结论。
#   本探针用 torch.profiler 按算子取 self 时间，再按"算数 / 搬数据"归类，
#   并与本机**实测可达上限**（GEMM 233~351 GFLOP/s、DRAM 25.4 GB/s、交叉点 9 FLOP/byte）比。
P.add_argument("--bw_probe", type=int, default=0,
               help="跑 N 个训练步做带宽/算子探针，然后退出（0=关闭，不影响正常训练）")
P.add_argument("--bw_probe_json", default="", help="探针结果落盘 JSON")
P.add_argument("--ckpt_every", type=int, default=200)
P.add_argument("--sample_every", type=int, default=0)
P.add_argument("--sample_prompt", default="")
P.add_argument("--sample_steps", type=int, default=20)
P.add_argument("--cache_dir", default="", help="潜变量缓存目录（缺省=脚本目录/cache_latents）")
P.add_argument("--no_mem_monitor", action="store_true")
# ★ 系统级可用内存下限（GB）。低于它就存盘并退出（exit 3），避免把整机饿死。
#   2026-09-25 加：先前只报告不阻止，导致系统 RAM 顶到 15.29/15.37 GB 时
#   explorer.exe 假死、训练自己 0xC0000005。设 0 关闭。
P.add_argument("--min_free_gb", type=float, default=1.5)
# ★ 续训：指向 checkpoint-*（或 checkpoint-memguard-*）目录。
#   会载入 LoRA 权重 + trainer_state.pt（optimizer/scheduler/step）。
#   和 --min_free_gb 是一对：守卫负责体面退出，resume 负责退出后不白跑。
P.add_argument("--resume", default="")
# ★★ 2026-09-26 08:1x：把一个**已经存在的** checkpoint 的状态当成"续跑起点"
#   （`--resume` 指向它），但把权重**也**从 `--resume_dir` 载入。
#   存在的理由：验证"修好之后续跑能不能逐位重现"时，如果每次都从 step 0 重跑前 30 步，
#   一次验证要 12 分钟；而前 30 步是同一段代码、结果已经逐位相同（权重 sha 一致）。
#   用 `--resume_dir` 就能**直接接着那个 ckpt 往下跑 30 步**，把验证缩到 4 分钟，
#   而且它验证的正是"从那个 ckpt 续跑"这件事本身 —— 不改变被测对象。
P.add_argument("--resume_dir", default="",
               help="与 --resume 配合：适配器权重改从这个目录载入（默认同 --resume）")
# ★ 数据游标覆盖（§10.133 的验证用）。只影响"从排列的第几个样本接着吃"。
#   用途：`--resume` 的那些 ckpt 是**加游标之前**存的（没有 cursor 键），
#   而重建它们要跑 2.8 分钟。把"新代码本该写进去的那个值"直接注进去，
#   就能只跑续跑那一段来验证 —— 不改变被测对象（被测的正是"从那个 ckpt 续跑"）。
P.add_argument("--cursor_override", type=int, default=-1,
               help="覆盖续跑时的数据游标（-1=用 ckpt 里的 cursor，回退到 step）")
# ★ 学习率调度（2026-09-25 加）。默认 --warmup 0 = 【完全保持原行为】，A/B 才干净。
#   动机：分词器那条线（train_vqvae_gan.py:328-336）用了线性 warmup + 余弦退火，
#   报告 §5.3 记「收敛快约 20 倍」（S1 在 3620 步就到 C_mse 20000 步的水平），
#   而扩散这条线一直是 get_scheduler("constant", num_warmup_steps=0) —— 从没接上。
P.add_argument("--warmup", type=int, default=0, help="线性 warmup 步数；0=关闭（原行为）")
P.add_argument("--lr_min", type=float, default=0.0, help="余弦退火的下限学习率")
# ★ P1-2 的 oracle（2026-09-25 加）：逐步 loss 落盘成 JSONL。
#   起因（实测）：`trainer_state.pt` 的顶层键只有 ['step','opt_steps','opt','sched']，
#   **没有 log_history** ⇒ 训练历史根本没存；而 wait_then_run 捕获的 tqdm 输出用 \r 分隔，
#   结束时只剩十几行 ⇒ **loss 曲线从来没有被持久化过**。
#   P1-2 的判据是 loss 曲线、四路 A/B 也需要它 ⇒ 必须落盘。默认关（不开就是原行为）。
P.add_argument("--loss_log", default="", help="逐步 loss 落盘成 JSONL；空=不写（原行为）")
# ★ P1-4：预热全部文本嵌入，然后释放 CLIP text encoder（省约 0.49 GB）。
#   需要 --sample_every 0 才允许释放（采样要用 pipe.text_encoder）。默认全关 = 原行为。
P.add_argument("--prewarm_text_emb", action="store_true",
               help="开训前把数据集里所有不同 caption 的文本嵌入灌进缓存")
P.add_argument("--free_text_encoder", action="store_true",
               help="预热后释放 text_encoder（要求 --sample_every 0）")
# disk_balancer 硬盘均衡负载（--flash 系列，SSD 保护）：生图训练同样适用
try:
    from disk_balancer import add_flash_args
    add_flash_args(P)
except Exception:
    pass
# TI
P.add_argument("--placeholder_token", default="<concept>")
P.add_argument("--initializer_token", default="photo")
# DreamBooth
P.add_argument("--train_text_encoder", action="store_true")
P.add_argument("--db_lora", action="store_true", help="DreamBooth 用 LoRA 而非 UNet 全参")
P.add_argument("--class_data_dir", default="", help="DreamBooth 先验保留类图像目录")
P.add_argument("--class_prompt", default="", help="先验保留类 prompt")
P.add_argument("--prior_loss_weight", type=float, default=1.0)
# ControlNet / iP-Adapter（控制图预处理无 cv2 时用 torch 边缘检测兜底）
P.add_argument("--controlnet", default="", help="已有 ControlNetModel 路径（空则 from_unet 新建）")
P.add_argument("--conditioning_data", default="", help="控制图目录（canny/depth/pose，默认=原图）")
P.add_argument("--condition_mode", default="canny", choices=["canny", "rgb"],
               help="无控制图目录时如何生成控制图：canny=torch 边缘检测 | rgb=原图灰度")
P.add_argument("--image_encoder", default="", help="iP-Adapter 图像编码器路径")
P.add_argument("--ip_scale", type=float, default=0.7, help="iP-Adapter 注意力 scale")
P.add_argument("--num_tokens", type=int, default=4, help="iP-Adapter 图像投影 token 数")
# Video
P.add_argument("--video_model", default="", help="视频 transformer/unet 路径")
P.add_argument("--video_selftest", action="store_true", help="AnimateDiff 5D 训练最小验证（随机 latent，5-10 步）")
P.add_argument("--video_frames", type=int, default=2, help="视频帧数 T（selftest）")
P.add_argument("--video_size", type=int, default=32, help="视频像素尺寸（selftest；latent = size//8）")
P.add_argument("--lora_scope", default="all", choices=["all", "temporal", "spatial"],
               help="animatediff_lora only: temporal = train temporal layers only "
                    "(keeps the SD1.5 spatial prior); all = both (default). See report 10.157")
P.add_argument("--video_real_smoke", action="store_true", help="真实 (image,pose) 对 -> VAE latent -> 10 步 LoRA")
P.add_argument("--vae_model", default="", help="VAE 路径（real_smoke 用；空则随机 latent）")
args = P.parse_args()

# ★★ 2026-09-26 07:2x：跨参数校验放在这里（解析之后、加载任何东西之前）——**秒级失败**。
#   起因：`--free_text_encoder` 与"文本编码不可缓存"的组合会在训练循环里抛
#   AttributeError（'NoneType' object is not callable），而那条报错**指不出真正的原因**；
#   更糟的是，我第一次把检查写在 prewarm 之后 ⇒ 要等模型加载完（约 90 秒）才报。
#   ⇒ 校验要放在最前面：能立刻拒的，就不要等到加载完再拒。
#   两种不可缓存的组合（L628 的 `cacheable` 会因此为 False，每步都要真的跑编码器）：
#     · --method ti（textual inversion）      · --train_text_encoder
if args.free_text_encoder and ((args.method == "ti") or args.train_text_encoder):
    sys.exit("[args] !! --free_text_encoder 与【不可缓存的文本编码】冲突："
             "method=%s  train_text_encoder=%s。这两条线的每一步都必须真的跑 text_encoder "
             "⇒ 释放之后必然崩。请去掉 --free_text_encoder（或去掉 --train_text_encoder）。"
             % (args.method, args.train_text_encoder))
if args.free_text_encoder and args.sample_every:
    sys.exit("[args] !! --free_text_encoder 与 --sample_every>0 冲突："
             "generate_samples() 需要 pipe.text_encoder。请设 --sample_every 0。")

_ARGS = args
# 路径全部相对本脚本目录推导，整目录拷贝到其他机器/盘符均可直接运行
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # 本目录的上级（如 D:\work 或拷到 i5 后的新位置）
_MODEL_ALIASES = {"tiny-sd": os.path.join(_ROOT, "tiny-sd"),
                  "bk-sdm-tiny": os.path.join(_ROOT, "bk-sdm-tiny"),
                  "bk-sdm-small": os.path.join(_ROOT, "bk-sdm-small"),
                  "sd-v1-5": os.path.join(_ROOT, "sd-v1-5")}
MODEL_DIR = _MODEL_ALIASES.get(args.model, args.model)

TH = args.threads or _pick_threads(lm_mode=False)
os.environ["OMP_NUM_THREADS"] = str(TH)
os.environ["MKL_NUM_THREADS"] = str(TH)
os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))
sys.path.insert(1, os.path.join(_HERE, "pytorch"))

import torch
torch.set_num_threads(TH)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
random.seed(args.seed)
torch.manual_seed(args.seed)
print(f"[env] method={args.method} threads={TH} torch={torch.__version__} seed={args.seed}")

import bitsandbytes as bnb
import psutil
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler, ControlNetModel
from diffusers.optimization import get_scheduler

from diffusion_backends import apply_method, METHODS  # noqa: E402

# 早期分支：AnimateDiff 5D 训练最小验证（不加载 SD pipeline）
if args.method in ("wan_lora", "cogvideo_lora", "svd_lora", "animatediff_lora", "video_full"):
    if args.video_real_smoke:
        from animatediff_train import run_animatediff_real_smoke
        res = run_animatediff_real_smoke(args)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        sys.exit(0)
    if args.video_selftest:
        from animatediff_train import run_animatediff_selftest
        res = run_animatediff_selftest(args)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        sys.exit(0)

OUT_DIR = args.out or os.path.join(_ROOT, "output", f"diffusion_{args.method}_{os.path.basename(MODEL_DIR)}")
os.makedirs(OUT_DIR, exist_ok=True)
CACHE_DIR = args.cache_dir or os.path.join(_HERE, "cache_latents")
os.makedirs(CACHE_DIR, exist_ok=True)


# ============== 数据集 / 缓存（复用 train_sd_lora.py 已验证实现） ==============
def _weight_digest(sub: str) -> str:
    d = os.path.join(MODEL_DIR, sub)
    for name in ("diffusion_pytorch_model.safetensors", "model.safetensors",
                 "pytorch_model.safetensors", "diffusion_pytorch_model.bin",
                 "pytorch_model.bin", "model.bin"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            h = hashlib.md5()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()[:12]
    return "unknown"


class SDDataset(Dataset):
    def __init__(self, data_dir, res, max_samples=0, concept="nsfw_art",
                 prompts_file="prompts.json"):
        self.res = res
        meta = os.path.join(data_dir, prompts_file)
        self.samples = []
        if os.path.exists(meta):
            with open(meta, "r", encoding="utf-8") as f:
                data = json.load(f)
            for fname, info in data.items():
                self.samples.append((os.path.join(data_dir, "images", fname),
                                     info.get("prompt", "") if isinstance(info, dict) else str(info)))
        else:
            img_dir = os.path.join(data_dir, "images")
            if os.path.isdir(img_dir):
                for fn in sorted(os.listdir(img_dir)):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        self.samples.append((os.path.join(img_dir, fn), f"a photo of {concept}"))
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        self.samples = [(p, pr) for p, pr in self.samples if os.path.exists(p)]
        if not self.samples:
            sys.exit(f"[ERROR] 数据集为空: {data_dir}")
        self.transform = transforms.Compose([
            transforms.Resize((res, res), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
        ])
        print(f"[data] {len(self.samples)} 张图 @ {res}x{res}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, prompt = self.samples[i]
        return {"pixel_values": self.transform(Image.open(path).convert("RGB")), "prompt": prompt,
                "path": path}


class LatentDiskCache:
    def __init__(self, cache_dir, vae_id, res):
        self.cache_dir = os.path.join(cache_dir, f"latents_{vae_id}_{res}")
        self.vae_id, self.res = vae_id, res
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, img_path):
        return hashlib.sha256(f"{self.vae_id}|{self.res}|{os.path.abspath(img_path)}".encode()).hexdigest()

    def get_or_encode(self, img_path, vae, transform):
        key = self._key(img_path)
        path = os.path.join(self.cache_dir, key + ".pt")
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)
        pixel = transform(Image.open(img_path).convert("RGB"))
        with torch.no_grad():
            latents = vae.encode(pixel.unsqueeze(0)).latent_dist.sample()
            latents = (latents * vae.config.scaling_factor).squeeze(0)
        torch.save(latents, path)
        return latents


class CachedLatentDataset(Dataset):
    def __init__(self, base, latent_cache, vae):
        self.base, self.lc, self.vae = base, latent_cache, vae
        self.degrade = None      # 配对模式挂上去；None = 普通模式，行为完全不变

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        lat = self.lc.get_or_encode(self.base.samples[i][0], self.vae, self.base.transform)
        out = {"latents": lat, "prompt": item["prompt"]}
        if self.degrade is not None:
            out["lat_deg"] = self.degrade(self.base.samples[i][0])
        return out


# ============== 控制图生成（canny 无 cv2，用 torch 边缘检测兜底） ==============
def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] (0..1) -> [B,1,H,W]"""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def make_control_image(pixel: torch.Tensor, mode: str = "canny") -> torch.Tensor:
    """把 [3,H,W] 归一化图像转成控制图 [3,H,W]（canny / rgb 灰度）。

    用 Sobel 边缘 + 非极大值抑制的简化边缘检测（无 cv2 依赖，够训练冒烟，
    正式 canny 可换 --conditioning_data 预生成目录）。
    """
    if mode == "rgb":
        g = _rgb_to_gray(pixel.unsqueeze(0))  # [1,1,H,W]
        return g.squeeze(0).repeat(3, 1, 1)
    g = _rgb_to_gray(pixel.unsqueeze(0))  # [1,1,H,W]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(g, kx, padding=1)
    gy = torch.nn.functional.conv2d(g, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)          # 边缘强度
    # 简单阈值归一化到 0..1（边缘图）：保留强边缘，弱边缘置暗
    m = mag / (mag.max() + 1e-6)
    m = torch.clamp(m * 3.0, 0, 1)
    out = m.squeeze(0).expand(3, -1, -1)
    return out


def _load_control_image(img_path: str, transform, mode: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    pixel = transform(img)
    return make_control_image(pixel, mode)


# ============== 加载 SD 组件 ==============
print(f"[load] {MODEL_DIR} (fp32, safety_checker 关闭)")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_DIR, dtype=torch.float32, safety_checker=None,
    requires_safety_checker=False, local_files_only=True)
vae, text_encoder, unet, tokenizer = pipe.vae, pipe.text_encoder, pipe.unet, pipe.tokenizer
VAE_ID = _weight_digest("vae")
TEXT_ID = _weight_digest("text_encoder")
noise_sched = DDPMScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
print(f"[ids] vae={VAE_ID} text_encoder={TEXT_ID}")

# 文本嵌入缓存（TI / train_text_encoder 时不缓存，需保留梯度）
EMB_DIR = os.path.join(CACHE_DIR, f"emb_{TEXT_ID}")
os.makedirs(EMB_DIR, exist_ok=True)
emb_lru: "OrderedDict[str, torch.Tensor]" = OrderedDict()
EMB_LRU_MAX = 512


def _encode_text_emb(prompt, text_encoder, tokenizer, cacheable=True):
    if cacheable:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key in emb_lru:
            emb_lru.move_to_end(key)
            return emb_lru[key]
        path = os.path.join(EMB_DIR, key[:32] + ".pt")
        if os.path.exists(path):
            emb = torch.load(path, map_location="cpu", weights_only=True)
            emb_lru[key] = emb
            if len(emb_lru) > EMB_LRU_MAX:
                emb_lru.popitem(last=False)
            return emb
    ti = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                   truncation=True, return_tensors="pt")
    out = text_encoder(ti.input_ids)[0].cpu()
    if cacheable:
        torch.save(out, path := os.path.join(EMB_DIR, hashlib.sha256(prompt.encode()).hexdigest()[:32] + ".pt"))
    return out


# ============== 全局冻结（注入函数按需解冻） ==============
vae.requires_grad_(False)
vae.eval()
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# ============== 应用方法（注入） ==============
setup = None
controlnet = None
if args.method == "controlnet":
    controlnet = ControlNetModel.from_pretrained(args.controlnet, local_files_only=True) if args.controlnet else None
    setup = apply_method("controlnet", unet=unet, controlnet=controlnet)
elif args.method == "ip_adapter":
    from transformers import CLIPVisionModelWithProjection
    img_enc = None
    if args.image_encoder:
        img_enc = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder, local_files_only=True)
    setup = apply_method("ip_adapter", unet=unet, image_encoder=img_enc,
                         num_tokens=args.num_tokens, ip_scale=args.ip_scale)
elif args.method in ("wan_lora", "cogvideo_lora", "svd_lora", "animatediff_lora", "video_full"):
    # 视频族。注：原注释写"本地暂无权重"，实测 AnimateDiff 权重在本地且 5D 训练可跑
    # （report §10.156）。lora_scope 只在 animatediff_lora 上生效：temporal=只训时序层、
    # 保留 SD1.5 空间先验（report §10.157）。
    setup = apply_method(args.method, video_transformer=None
                         if not args.video_model else _load_video_transformer(args.video_model),
                         lora=(args.method != "video_full"),
                         rank=args.rank, alpha=args.alpha, dropout=args.lora_dropout,
                         target_modules=[x.strip() for x in args.targets.split(",") if x.strip()],
                         lora_scope=getattr(args, "lora_scope", "all"))
elif args.method == "ti":
    setup = apply_method("ti", text_encoder=text_encoder, tokenizer=tokenizer,
                         placeholder_token=args.placeholder_token,
                         initializer_token=args.initializer_token)
elif args.method == "dreambooth":
    setup = apply_method("dreambooth", unet=unet, text_encoder=text_encoder,
                         train_text_encoder=args.train_text_encoder,
                         lora=args.db_lora, rank=args.rank, alpha=args.alpha,
                         dropout=args.lora_dropout,
                         target_modules=args.targets.split(","))
elif args.method == "full":
    setup = apply_method("full", unet=unet, text_encoder=text_encoder,
                         train_text_encoder=args.train_text_encoder)
else:  # lora
    setup = apply_method("lora", unet=unet, rank=args.rank, alpha=args.alpha,
                         dropout=args.lora_dropout, target_modules=args.targets.split(","))


def _load_video_transformer(path):
    # 视频 transformer 加载占位：根据 family 选择 pipeline，尚未实现权重加载
    raise NotImplementedError("视频模型权重加载待 D 阶段接入；本地暂无 Wan/CogVideoX/SVD/AnimateDiff 权重")


print(setup.summary())

# 取回可能被 apply_method 包装过的模块（LoRA 会返回新的 PeftModel）
if "unet" in setup.extra:
    unet = setup.extra["unet"]

# 视频：本地无权重，仅打印结构提示后退出（保持交接文档约定）
if setup.kind == "video":
    print(f"[train_diffusion] method={args.method} 注入层已就绪，但本地无视频权重，"
          f"仅验证注入结构，不进入训练循环。")
    sys.exit(0)

# ============== 优化器构建（按 kind 收集可训练参数） ==============
def _collect_trainable() -> list:
    """收集 setup 中标记的可训练参数（含 controlnet / ip_adapter 的适配层）。"""
    params = []
    if setup.kind == "controlnet":
        cn = setup.extra["controlnet"]
        params += [p for p in cn.parameters() if p.requires_grad]
    elif setup.kind == "ip_adapter":
        # processor.to_k_ip/to_v_ip（挂在 unet 各 cross-attn 名下）+ image_proj
        from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
        for name, m in unet.named_modules():
            proc = getattr(m, "processor", None)
            if isinstance(proc, IPAdapterAttnProcessor2_0):
                for p in proc.parameters():
                    if p.requires_grad:
                        params.append(p)
        ip = setup.extra.get("image_proj")
        if ip is not None:
            params += [p for p in ip.parameters() if p.requires_grad]
    else:
        params += [p for p in (list(unet.parameters()) + list(text_encoder.parameters()))
                   if p.requires_grad]
    # 去重
    seen, dedup = set(), []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            dedup.append(p)
    return dedup


trainable = _collect_trainable()
n_train = sum(p.numel() for p in trainable)
print(f"[train] trainable={n_train / 1e6:.2f}M params")
if not trainable:
    sys.exit("[ERROR] 无可训练参数，检查注入/冻结逻辑")
if args.opt == "adamw8bit":
    opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
elif args.opt == "adafactor":
    from transformers import Adafactor
    opt = Adafactor(trainable, lr=args.lr, relative_step=False)
else:
    opt = torch.optim.AdamW(trainable, lr=args.lr)
sched = get_scheduler("constant", optimizer=opt, num_warmup_steps=0,
                      num_training_steps=max(1, args.steps // args.grad_accum + 1))

# ★ 逐步 loss 落盘（P1-2 的 oracle）。见 --loss_log 的说明。
#   ★★ 2026-09-26 08:1x：续跑时改成**追加**（§10.133⑦）。
#   缺陷原状：一直是 `open(..., "w")` ⇒ **续跑时若指向同一个路径，就会把续跑之前的
#   那些行清空** —— 而生产里这非常容易发生（`--out` 不变 ⇒ `--loss_log` 常常还是同一个
#   路径）⇒ "续跑一次就把续跑前的曲线抹掉"。§10.127⑩ 那次"证据被覆盖"我当时以为是
#   自己拷贝太慢，其实是**代码本身就会覆盖**。
#   `--resume` 的语义就是"接着这次 run 继续" ⇒ 它的日志当然也该接着写。
_LOSS_F = None
if args.loss_log:
    _lp = args.loss_log
    if not os.path.isabs(_lp):
        _lp = os.path.join(OUT_DIR, _lp)
    os.makedirs(os.path.dirname(_lp) or ".", exist_ok=True)
    _lmode = "a" if args.resume else "w"
    _LOSS_F = open(_lp, _lmode, encoding="utf-8")
    _lex = ""
    if _lmode == "a":
        try:
            _ln = sum(1 for _ in open(_lp, "r", encoding="utf-8", errors="replace"))
            _lex = f"（已有 {_ln} 行，追加而不是清空）"
        except Exception:
            pass
    print(f"[loss-log] -> {_lp}  mode={_lmode}{_lex}", flush=True)

# ============== 数据 ==============
if not args.data:
    sys.exit("[ERROR] 未指定数据集：请用 --data <目录>（目录内需 images/ 与 prompts.json，"
             "或只有 images/；也可参考测试脚本用合成图目录）")
base_ds = SDDataset(args.data, args.res, args.max_samples, args.concept,
                    prompts_file=args.prompts_file)

# ★ P1-4（2026-09-25）：预热文本嵌入后释放 CLIP text encoder，省约 0.49 GB。
#   依据：`_encode_text_emb` 的缓存命中路径（LRU / 磁盘）**完全不碰 text_encoder**，
#   而 EMB_LRU_MAX=512 > 我们的 307 条不同 caption ⇒ 预热后连磁盘读都不需要。
#   保险：--sample_every > 0 时【不许】释放 —— generate_samples() 要用 pipe.text_encoder。
if args.prewarm_text_emb:
    _ps = {p for _, p in base_ds.samples}
    if args.sample_prompt:
        _ps.add(args.sample_prompt)
    print(f"[prewarm] 预热 {len(_ps)} 条不同 caption 的文本嵌入 ...", flush=True)
    _t0 = time.perf_counter()
    for _i, _p in enumerate(sorted(_ps)):
        _encode_text_emb(_p, text_encoder, tokenizer, cacheable=True)
        if (_i + 1) % 100 == 0:
            print(f"[prewarm]   {_i + 1}/{len(_ps)}", flush=True)
    _hit = sum(1 for _p in _ps
               if hashlib.sha256(_p.encode("utf-8")).hexdigest() in emb_lru
               or os.path.exists(os.path.join(
                   EMB_DIR, hashlib.sha256(_p.encode("utf-8")).hexdigest()[:32] + ".pt")))
    print(f"[prewarm] 完成 {len(_ps)} 条，缓存覆盖 {_hit}/{len(_ps)}"
          f"（LRU 上限 {EMB_LRU_MAX}），耗时 {time.perf_counter() - _t0:.1f}s", flush=True)
    if _hit < len(_ps):
        sys.exit(f"[prewarm] !! 覆盖率不足 {_hit}/{len(_ps)}，拒绝释放 text_encoder（会在 cache miss 时炸）")
    if args.free_text_encoder:
        if args.sample_every:
            sys.exit("[prewarm] !! --free_text_encoder 与 --sample_every>0 冲突："
                     "generate_samples() 需要 pipe.text_encoder。请设 --sample_every 0。")
        # ★★ 2026-09-26 07:2x 补（同一类遗漏的第三处）：还有两种组合会**在释放后必然崩**，
        #   因为它们的文本编码路径【不可缓存】：
        #     · `--method ti`（textual inversion）      ⇒ L628 的 cacheable = False
        #     · `--train_text_encoder`                  ⇒ 同上
        #   ⇒ 这两条线上 `_encode_text_emb` **每次都真的调用编码器**（缓存命中那条捷径不走），
        #     而编码器已被置 None ⇒ 训练第一步就在 L376 抛 AttributeError，
        #     报错信息（'NoneType' object is not callable）**指向不了真正的原因**。
        #   ⇒ 在解析阶段就拒掉，比让它跑到训练循环里炸清楚得多（与上面那条冲突检查同一形状）。
        _uncacheable = (args.method == "ti") or args.train_text_encoder
        if _uncacheable:
            sys.exit("[prewarm] !! --free_text_encoder 与【不可缓存的文本编码】冲突："
                     "method=%s train_text_encoder=%s。这两条线的每步都要真的跑编码器 "
                     "⇒ 释放后必然崩。请去掉 --free_text_encoder（或去掉 --train_text_encoder）。"
                     % (args.method, args.train_text_encoder))
        _rss0 = psutil.Process().memory_info().rss / 1e9
        pipe.text_encoder = None
        text_encoder = None
        import gc
        gc.collect()
        print(f"[prewarm] 已释放 text_encoder，RSS {_rss0:.2f} → "
              f"{psutil.Process().memory_info().rss / 1e9:.2f} GB", flush=True)

lat_cache = LatentDiskCache(CACHE_DIR, VAE_ID, args.res)
ds = CachedLatentDataset(base_ds, lat_cache, vae)

# ---- ★配对模式：退化输入 = 该图的 VQ 重建（= 报告里"域匹配的超分"的输入分布）----
# 为什么需要：普通训练是 noise_sched.add_noise(真图 latent) ⇒ 模型见过的状态都是
# 「真图 + 噪声」。而推理时 img2img 的起点是「【VQ 重建图】latent + 噪声」——
# 两个分布不同（§10.28 曾把这个叫 exposure bias，§10.34 又推翻了那个解释；
# 不管叫什么，输入分布不一致是事实）。配对模式就是把训练状态改成推理时真正会遇到的那个。
if args.paired_vqvae:
    from small_vqvae import vqvae_from_ckpt
    _vq, _vk = vqvae_from_ckpt(args.paired_vqvae, map_location='cpu')
    _vq.eval()
    _vq.requires_grad_(False)
    print("[paired] VQVAE 已加载: %s  (step=%s)" % (args.paired_vqvae, _vk))
    _ptf = base_ds.transform

    def _degrade(path):
        with torch.no_grad():
            px = _ptf(Image.open(path).convert("RGB")).unsqueeze(0)
            rec = _vq(px)[0]                       # VQVAE.forward -> (xr, idx, vq_loss)
            rec = rec.clamp(-1, 1)
            lat = vae.encode(rec).latent_dist.sample()
            lat = (lat * vae.config.scaling_factor).squeeze(0)
        return lat

    ds.degrade = _degrade
    print("[paired] 已开启：状态从 VQ 重建出发，回归目标指向真图；t ∈ [%d, 1000)"
          % args.paired_tlo)

# ★★ 2026-09-26 08:1x：**数据顺序必须与全局 RNG 解耦**（§10.133 的真因）。
#   真因取证（零算力 `--order_only` 探针，两次运行）：
#       无 --resume 的一口气跑：step1 样本 = 98ae4ceb069b / ef06ecf767c0 / ef9c79798b76 ...
#       有 --resume 的续跑    ：step1 样本 = c6ccd941da33 / ee766c53c9e3 / 61655fb920e1 ...
#       ★ 两条序列的样本**交集 = 0/120** —— 不是"错位"，是**完全不同的排列**。
#   机制：`shuffle=True` 且不给 generator 时，torch 2.13 的 `RandomSampler.__iter__` 会
#       `seed = torch.empty((),dtype=int64).random_().item()` —— **从全局 RNG 取种子**。
#       而一口气跑在建 UNet 时会**随机初始化 LoRA 权重**（消耗全局 RNG），续跑则是
#       `load_state_dict` 载入权重（不消耗）⇒ 两边在 `iter(loader)` 那一刻的全局 RNG
#       **本来就不同**，与 restore 与否无关（我先前"DataLoader 构造消耗 RNG"的猜测也被
#       探针证伪：`构造消耗全局 RNG = False`，所以那个 `rng_before_loader` 补偿是无效的）。
#   修法：给 DataLoader 一个**专用 generator**（只用 `args.seed`，不碰全局 RNG），
#       于是排列只由 `--seed` 决定，与"是首跑还是续跑""LoRA 是随机初始化还是载入"全无关。
#       `loader` 保留给别的分支用；训练循环改用下面按"排列 + 游标"取批的 `_loader_subset()`。
_RNG_BEFORE_LOADER = torch.get_rng_state().clone()
_SHUF_GEN = torch.Generator()
_SHUF_GEN.manual_seed(int(args.seed))
loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False,
                    generator=_SHUF_GEN)
_rng_after_loader = torch.get_rng_state()
_RNG_LOADER_CONSUMED = not bool((_RNG_BEFORE_LOADER == _rng_after_loader).all())
print(f"[order] DataLoader 构造消耗全局 RNG = {_RNG_LOADER_CONSUMED} "
      f"(实测 False；数据顺序已改用专用 generator(seed={args.seed})，与全局 RNG 无关)",
      flush=True)


def _rng_fp():
    """全局 RNG 位置的指纹（12 位）。★ 只在 `--order_only` 诊断里打印，不进训练路径。"""
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:12]


if args.order_only:
    print(f"[rngfp] A: 建完 loader（建模型+Lora 之后）      = {_rng_fp()}", flush=True)


def _epoch_perm(epoch):
    """第 `epoch` 个 epoch 的样本排列：只由 `--seed` 与 epoch 决定（与全局 RNG 无关）。

    ★ 这是"续跑能否逐位重现"的地基：同一 (seed, epoch) 无论被谁、在哪一次运行里算出，
      都得到同一条排列。续跑侧只需要再恢复**游标**（`global_cursor`）即可接上。
    ★ 带缓存：训练循环每个微步都要取一次，`randperm(8000)` 虽小但没必要重复算。
    ★ 已知边界（诚实记下，实测范围内不影响）：`_loader_subset` 先按"行块"切 `Subset`，
      所以当**游标跨过 epoch 边界**（cursor 超过 `len(ds)`）时，一个 batch 可能由
      第 e 与第 e+1 个排列的样本拼成，而 `trainer_state` 只记了游标、没记 epoch。
      在"一个 epoch > 本次要跑的步数"时（本机实测：8000 样本 / 60 步、1500 步都远小于 8000）
      永远不会跨边界。真要做十万步级的长 run，这里应当改成记 `(epoch, offset)`。
    """
    _c = getattr(_epoch_perm, "_cache", None)
    if _c is None:
        _c = {}
        _epoch_perm._cache = _c
    if epoch not in _c:
        _g = torch.Generator()
        _g.manual_seed(int(args.seed) * 1000003 + int(epoch))
        _c[epoch] = torch.randperm(len(ds), generator=_g).tolist()
    return _c[epoch]


def _loader_subset(step0, n):
    """取"从第 `step0` 个微步起、共 `n` 个微步"的样本子集（按排列 + 游标切）。

    为什么要它：原来的 `for batch in loader` 每次 `iter()` 都**从头**重新洗牌 ——
    续跑时 `global_step` 已经是 30，但取到的是排列的**第 1 个**样本（而不是第 31 个）。
    这就是"续跑曲线与一口气曲线不同"的第二个成分（第一个成分是上面的排列种子）。

    ★★ 2026-09-26 08:4x：**`iter(DataLoader)` 一定会推进全局 RNG —— 除非给了显式 generator**。
    实测（本机 torch 2.13.0+cpu，每个用例都从 `manual_seed(0)` 起）：
        shuffle=False（默认 sampler）        构造变了=False  **iter变了=True**
        shuffle=False + SequentialSampler    构造变了=False  **iter变了=True**
        shuffle=True + **显式 generator**    构造变了=False  **iter变了=False**  ← 只有这个不动
    而本函数是在**训练循环里面**每一步调用的 ⇒ 它会把每一步的 `noise` 抽样整体推后
    （实测 2a9fe4ca15ae → bda1b6109cee）⇒ 续跑永远不可能与一口气跑逐位相同。
    修法：给每个 DataLoader 一个**自己的** generator（`manual_seed(0)` 固定值）。
    它不改变取批顺序（顺序完全由 `_epoch_perm` + 游标决定，`shuffle=False`），
    只是把"iter 时那次全局 RNG 消耗"引到一条一次性的专用流上。
    """
    _b = max(1, int(args.batch))
    _perm = _epoch_perm(0)
    _lo = (int(step0) * _b) % len(_perm)
    _sel = []
    while len(_sel) < max(_b, int(n) * _b):
        _need = max(_b, int(n) * _b) - len(_sel)
        _sel += _perm[_lo:_lo + _need]
        _lo = 0
        if len(_sel) >= len(_perm):
            break
    if not _sel:
        _sel = _perm[:max(_b, int(n) * _b)]
    _sub = torch.utils.data.Subset(ds, _sel)
    _g = torch.Generator()
    _g.manual_seed(0)
    return DataLoader(_sub, batch_size=_b, shuffle=False, generator=_g,
                      drop_last=False)

# ControlNet 控制图潜变量缓存（按 控制图哈希 + vae + res 寻址）
control_lat_cache = None
if setup.kind == "controlnet":
    control_lat_cache = LatentDiskCache(CACHE_DIR, f"{VAE_ID}_ctl", args.res)

# DreamBooth 先验保留
prior_loader = None
prior_iter = None
if args.class_data_dir and args.class_prompt:
    prior_base = SDDataset(args.class_data_dir, args.res, args.max_samples, concept=args.class_prompt)
    # 类图像统一用 class_prompt（先验保留要求）
    prior_base.samples = [(p, args.class_prompt) for p, _ in prior_base.samples]
    prior_ds = CachedLatentDataset(prior_base, lat_cache, vae)
    prior_loader = DataLoader(prior_ds, batch_size=1, shuffle=True, drop_last=False)
    print(f"[dreambooth] 先验保留启用: {args.class_data_dir} prompt={args.class_prompt!r} w={args.prior_loss_weight}")

# ============== 内存监视 ==============
mon_ev = None
if not args.no_mem_monitor:
    try:
        from torch_cpu_kit import start_mem_monitor, suspend_mem_monitor
        mon_ev = start_mem_monitor(interval=10)
        print("[mem] 监视器已启动")
    except Exception as e:
        print(f"[mem] 监视器不可用: {e}")

# ============== disk_balancer 硬盘均衡负载（--flash，SSD 保护）==============
balancer = None
try:
    if hasattr(args, "flash") and args.flash is not None:
        from disk_balancer import parse_flash_args, DiskLoadBalancer
        cfg = parse_flash_args(args)
        if cfg is not None:
            balancer = DiskLoadBalancer(cfg)
            balancer.attach_model(unet if setup.kind in ("lora", "full", "controlnet", "ip_adapter")
                                  else model)
            balancer.start()
            print(f"[flash] disk_balancer 已启动: 模式={cfg.mode}")
except Exception as e:
    print(f"[flash] disk_balancer 初始化失败（忽略）: {e}")
    balancer = None


# ============== 采样 ==============
def generate_samples(step):
    prompt = args.sample_prompt or (base_ds.samples[0][1] or f"a photo of {args.concept}")
    try:
        pipe.unet = unet
        pipe.text_encoder = text_encoder
        img = pipe(prompt=prompt, width=args.res, height=args.res,
                   num_inference_steps=args.sample_steps, guidance_scale=7.5).images[0]
        p = os.path.join(OUT_DIR, f"sample_step{step}.png")
        img.save(p)
        print(f"[sample] {p}")
    except Exception as e:
        print(f"[sample] 失败: {e}")


# ============== 训练循环 ==============
# 文本是否缓存 / 是否训练 text_encoder：
cacheable = (args.method not in ("ti",)) and (not args.train_text_encoder)
train_text = (args.method == "ti") or args.train_text_encoder
# ★★ 2026-09-26 07:0x 修复：`--free_text_encoder` 会把 text_encoder 置为 None（L530-531），
#   而这一行原来是**无条件**调用 ⇒ 释放之后立刻 AttributeError:
#       'NoneType' object has no attribute 'eval'
#   实测（p14_textenc 的 unload 臂，01:45:44）：prewarm 成功（271/271）、释放成功
#   （RSS 0.94 → 0.60 GB），**紧接着这一行就崩** ⇒ 这个特性此前从未被真正驱动过（§10.94），
#   第一次驱动就暴露了它。
#
#   为什么只修这一处就够 —— 我把所有 `text_encoder` 的用法都过了一遍（§10.127⑨）：
#     · L616 `pipe.text_encoder = text_encoder`  在 generate_samples 的 try 里（异常被吞，§10.106）
#     · L816 / L871 `_encode_text_emb(..., text_encoder, ...)`
#         ⇒ 该函数在【缓存命中】时提前返回、**完全不碰 text_encoder**（L364-373）；
#           而 --prewarm_text_emb 保证覆盖率（L524 覆盖率不足会拒绝释放）⇒ 安全
#     · L959 `text_encoder.get_input_embeddings()` 在 `if args.method == "ti"` 分支里
#         ⇒ TI 那条线本来就不可缓存（L628），与释放互斥 ⇒ 不在本修复范围内
#   ⇒ 只有本行是"无条件、释放后必达"的。
if text_encoder is not None:
    text_encoder.train() if train_text else text_encoder.eval()
else:
    # 显式记录"少做了一件事"，而不是靠吞异常（§10.106④ 的同一条原则）
    print("[textenc] text_encoder 已释放（--free_text_encoder）⇒ 跳过 train/eval 切换",
          flush=True)
unet.train()

# ControlNet / iP-Adapter 的可训练模块进入 train 模式
if setup.kind == "controlnet":
    setup.extra["controlnet"].train()
elif setup.kind == "ip_adapter":
    # 每个 Attention 的 processor（IPAdapterAttnProcessor2_0 是 nn.Module）置 train
    for m in unet.modules():
        proc = getattr(m, "processor", None)
        if isinstance(proc, torch.nn.Module):
            proc.train()

print(f"[train] kind={setup.kind} steps={args.steps} grad_accum={args.grad_accum} lr={args.lr} "
      f"cacheable_text={cacheable} train_text={train_text}")

# iP-Adapter 图像编码器 + 图像特征缓存（含 image_proj 前置）
image_encoder = None
image_proj = None
ip_cache = {}
if setup.kind == "ip_adapter":
    image_encoder = setup.extra.get("image_encoder")
    image_proj = setup.extra.get("image_proj")

    def _encode_image_feature(img_path):
        if img_path in ip_cache:
            return ip_cache[img_path]
        key = hashlib.sha256(f"{img_path}|{TEXT_ID}|{args.num_tokens}".encode()).hexdigest()
        pc = os.path.join(os.path.join(CACHE_DIR, f"ipfeat_{TEXT_ID}"), key + ".pt")
        os.makedirs(os.path.dirname(pc), exist_ok=True)
        with torch.no_grad():
            if image_encoder is None:
                # 无图像编码器：用原图直接构造零特征（冒烟用，可训练适配层仍能学）
                feat = torch.zeros(args.num_tokens,
                                   getattr(getattr(unet, "config", None), "cross_attention_dim", 768) or 768)
            else:
                img = Image.open(img_path).convert("RGB")
                img = base_ds.transform(img)  # [3,H,W] 0..1
                arr = (img.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
                from transformers import CLIPImageProcessor
                from PIL import Image as _PIL
                try:
                    proc = CLIPImageProcessor.from_pretrained(
                        MODEL_DIR, subfolder="feature_extractor", local_files_only=True)
                except Exception:
                    # tiny-sd 等无 feature_extractor 目录：用默认配置
                    proc = CLIPImageProcessor()
                vimg = proc(images=_PIL.fromarray(arr.round().astype("uint8")), return_tensors="pt")
                feats = image_encoder(**vimg)
                feat = feats.image_embeds.squeeze(0)  # [proj_dim]
                if image_proj is not None:
                    feat = image_proj(feat.unsqueeze(0)).squeeze(0)  # [num_tokens, dim]
        torch.save(feat, pc)
        ip_cache[img_path] = feat
        return feat

    if image_encoder is not None:
        print(f"[ip_adapter] image_encoder 已加载，图像特征将落盘缓存 {CACHE_DIR}/ipfeat_{TEXT_ID}")

# ControlNet 控制图处理
control_loader = None
if setup.kind == "controlnet":
    if args.conditioning_data and os.path.isdir(args.conditioning_data):
        print(f"[controlnet] 控制图目录: {args.conditioning_data}（按文件名与原图对齐）")

    def _control_latents(img_path):
        if args.conditioning_data and os.path.isdir(args.conditioning_data):
            cpath = os.path.join(args.conditioning_data, os.path.basename(img_path))
            if not os.path.exists(cpath):
                # 回退：用原图生成控制图
                cpath = img_path
        else:
            cpath = img_path
        key = os.path.abspath(cpath)
        cdir = os.path.join(control_lat_cache.cache_dir, "c")
        os.makedirs(cdir, exist_ok=True)
        # v3：ControlNet 条件是原始分辨率像素控制图（v2 误存 32×32, 作废）
        cfile = os.path.join(cdir, hashlib.sha256(f"v3|{key}|{args.condition_mode}|{args.res}".encode()).hexdigest() + ".pt")
        if os.path.exists(cfile):
            return torch.load(cfile, map_location="cpu", weights_only=True)
        # ControlNet 的 controlnet_cond 是 **原始像素** 控制图（3 通道, 原始分辨率），
        # cond_embedding 内部会下采样到 latent 尺寸 —— 不应预先缩到 32×32
        ctrl_pixel = _load_control_image(cpath, base_ds.transform, args.condition_mode)  # [3,H,W] 原始
        torch.save(ctrl_pixel, cfile)
        return ctrl_pixel

# ============== 训练循环主体 ==============
def set_lr(gstep: int) -> float:
    """线性 warmup + 余弦退火到 lr_min。逐字照抄 train_vqvae_gan.py:328-336 的验证过的公式。

    默认 --warmup 0 时【直接返回 1.0 且不碰 opt】，即完全保持原行为（原先是
    get_scheduler("constant")，LR 恒定）。
    """
    if args.warmup <= 0:
        return 1.0
    if gstep < args.warmup:
        f = float(gstep + 1) / float(args.warmup)
    else:
        span = max(1, args.steps - args.warmup)
        prog = min(1.0, max(0.0, (gstep - args.warmup) / float(span)))
        ratio = args.lr_min / args.lr if args.lr else 0.0
        f = ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
    for pg in opt.param_groups:
        pg["lr"] = args.lr * f
    return f


global_step = 0
opt_steps = 0
# ★ 2026-09-26 08:1x：**数据游标** = 已经消费了多少个样本（§10.133 真因 #2）。
#   它必须与 step 分开记：step 是"训练了多少微步"，游标是"排列吃到第几个"。
#   续跑时两者一起恢复，取批函数才能从排列的正确位置接着吃。
global_cursor = 0


def _trainer_state():
    """`--resume` 要保存的全部状态。★ 两个保存点【共用】这一个函数。

    为什么必须共用：守卫那条路径与正常 checkpoint 那条路径原来**各自写了一遍**同样的 dict，
    而它们**已经漂移过一次** —— 守卫那边少了 `trainer_state.pt`（§10.117②/§10.129③）。
    共用之后就不可能再漂移。

    ★★ 2026-09-26 07:4x 加 RNG 状态（这是 §10.131 的直接结论）：
      原来只有 step/opt_steps/opt/sched，**没有 RNG** ⇒ 续跑之后
      latents 抽样 / noise(`torch.randn`) / timestep(`torch.randint`) 会从另一个
      RNG 状态继续 ⇒ 实测「分段 vs 一口气」从续跑的第一个微步起就分岔（30/30 步全不同）。
      加上这两个之后，续跑才可能**逐位重现**一条未分段的训练。
    """
    import random as _rnd
    _st = {"step": global_step, "opt_steps": opt_steps,
           "cursor": global_cursor,          # ★ 数据游标（§10.133 真因 #2）
           "opt": opt.state_dict(), "sched": sched.state_dict(),
           "rng_torch": torch.get_rng_state(),
           "rng_py": _rnd.getstate()}
    # 注：`rng_before_loader` 保留只是为了旧 ckpt 的兼容读取；实测"DataLoader 构造消耗
    #     全局 RNG = False"，且数据顺序已改用专用 generator ⇒ 这个键**不再是必需的**。
    #     真正的两个原因是"排列种子取自全局 RNG"与"游标没恢复"（§10.133）。
    try:
        _st["rng_before_loader"] = _RNG_BEFORE_LOADER
    except NameError:
        pass
    try:
        import numpy as _np
        _st["rng_np"] = _np.random.get_state()
    except Exception:
        pass
    return _st


# ★ 2026-09-25 加：--resume 续训。
#   动机：今天因为内存事故重跑了三次、每次从零开始，白扔约 2 小时。
#   内存守卫（--min_free_gb）负责"体面地退出"，resume 负责"退出了不白跑"，
#   两个是一对，缺一个都不成立。
if args.resume:
    _rd = args.resume
    if not os.path.isdir(_rd):
        sys.exit(f"[ERROR] --resume 指向的目录不存在: {_rd}")
    if args.resume_dir:
        # 只把"适配器权重"的来源换掉；trainer_state.pt 仍从 --resume 的那个目录读。
        # 这样两边的**状态**是同一个 ckpt 的，权重也是同一个 ckpt 的 ⇒ 起点完全相同。
        _rdw = args.resume_dir
        if not os.path.isdir(_rdw):
            sys.exit(f"[ERROR] --resume_dir 指向的目录不存在: {_rdw}")
        print(f"[resume] 权重改从 --resume_dir 载入: {_rdw}")
    else:
        _rdw = _rd
    _ck = None
    for _f in ("adapter_model.safetensors", "pytorch_lora_weights.safetensors"):
        if os.path.exists(os.path.join(_rdw, _f)):
            _ck = os.path.join(_rdw, _f)
            break
    if _ck is None:
        sys.exit(f"[ERROR] {_rd} 里找不到 adapter_model.safetensors / "
                 f"pytorch_lora_weights.safetensors。"
                 f"（--resume 目前只支持 LoRA；full/controlnet 请传目录自己接）")
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    _sd = load_file(_ck)
    _miss, _unexp = set_peft_model_state_dict(unet, _sd)
    print(f"[resume] LoRA 权重已载入 {_ck}  missing={len(_miss)} unexpected={len(_unexp)}")
    # ★★ 2026-09-26 07:2x 修：原来这里判 `len(_miss) or len(_unexp)` ⇒
    #   而 `missing` **永远不等于 0**：适配器文件里只有 LoRA 键（实测 256 个），
    #   基模的参数会被算成 missing。实测这个数**正好等于 UNet 的张量数**：
    #       missing=686，而 unet/diffusion_pytorch_model.safetensors 的张量数 = 686（逐个数过）
    #   ⇒ 于是这条警告**每一次续训都会响**，而它说的是"先核对再信这次续训" ——
    #     一个每次都响的警告等于没有警告（甚至更糟：它会让人不信一次本来正确的续训）。
    #   正确的判据是 **unexpected == 0**（适配器里的每个键都被模型接受了）；
    #   missing 只要落在那 686 个基模张量上就是正常的。
    _n_base = 0
    try:
        _n_base = len(unet.state_dict()) - len(_sd)      # 基模张量数（不该被适配器提供）
    except Exception:
        _n_base = -1
    if len(_unexp):
        print(f"[resume] !! 有 {len(_unexp)} 个 unexpected 键 ⇒ 适配器与模型不匹配，"
              f"先核对再信这次续训")
    elif _n_base >= 0 and len(_miss) == _n_base:
        print(f"[resume] ✓ 适配器键全部匹配（missing={len(_miss)} 全是基模张量，属正常）")
    elif len(_miss):
        print(f"[resume] !! missing={len(_miss)} 既不等于基模张量数 {_n_base}、"
              f"又不是 0 ⇒ 先核对再信这次续训")
    else:
        print(f"[resume] ✓ 逐键完全匹配（missing=0 unexpected=0）")
    _stp = os.path.join(_rd, "trainer_state.pt")
    if os.path.exists(_stp):
        _st = torch.load(_stp, map_location="cpu", weights_only=False)
        for _k, _obj, _nm in (("opt", opt, "optimizer"), ("sched", sched, "scheduler")):
            if _k in _st:
                try:
                    _obj.load_state_dict(_st[_k])
                    print(f"[resume] {_nm} 状态已载入")
                except Exception as _e:
                    print(f"[resume] {_nm} 状态载入失败（继续跑，但 LR 动量会重置）: {str(_e)[:160]}")
        global_step = int(_st.get("step", 0))
        opt_steps = int(_st.get("opt_steps", 0))
        # ★★ 2026-09-26 07:4x：恢复 RNG 状态 —— 这是"续跑能否逐位重现"的关键（§10.131）。
        #   原来 trainer_state.pt 里没有 RNG ⇒ 续跑后的 lenats/noise/timestep 采样序列
        #   与一口气跑不同 ⇒ 实测从续跑第一个微步起就分岔（30/30 步全不同，最大差 0.63）。
        _rng_restored = []
        _rng_missing = []
        # ★★ 2026-09-26 08:2x：**顺序很重要** —— 先恢复"loader 构造前快照"，最后才恢复
        #   `rng_torch`。上一版把顺序写反了（rng_torch 先、快照后），结果是
        #   **正确的微步边界状态被一个 setup 期的状态覆盖掉** ⇒ 实测 step 31 起
        #   `rng`/`noise` 两列仍然不同（而 `ids`/`cur` 已经相同，证明数据顺序已修好）。
        #   现在 `rng_torch` 是**最后**写入的，所以它赢。
        #   注：`rng_before_loader` 这个键本身已经没有用了（实测 DataLoader 构造
        #   不消耗全局 RNG，且数据顺序已改用专用 generator），留着只为兼容旧 ckpt。
        if "rng_before_loader" in _st:
            try:
                torch.set_rng_state(_st["rng_before_loader"])
                _rng_restored.append("loader 构造前快照")
            except Exception as _e:
                print(f"[resume] loader 快照恢复失败（不影响数据顺序）: {str(_e)[:120]}")
        for _k, _setter, _nm in (("rng_torch", torch.set_rng_state, "torch RNG"),):
            if _k in _st:
                try:
                    _setter(_st[_k])
                    _rng_restored.append(_nm)
                except Exception as _e:
                    print(f"[resume] {_nm} 恢复失败: {str(_e)[:120]}")
            else:
                _rng_missing.append(_k)
        if "rng_py" in _st:
            try:
                import random as _rnd
                _rnd.setstate(tuple(_st["rng_py"]))
                _rng_restored.append("python random")
            except Exception as _e:
                print(f"[resume] python random 恢复失败: {str(_e)[:120]}")
        else:
            _rng_missing.append("rng_py")
        if "rng_np" in _st:
            try:
                import numpy as _np
                _np.random.set_state(_st["rng_np"])
                _rng_restored.append("numpy")
            except Exception:
                pass
        if _rng_restored:
            print(f"[resume] RNG 状态已恢复（{'/'.join(_rng_restored)}）⇒ 采样序列与一口气跑一致")
        if _rng_missing:
            print(f"[resume] !! trainer_state.pt 里没有 {','.join(_rng_missing)} ⇒ "
                  f"**续跑不会逐位重现**（采样序列会不同；见报告 §10.131）。"
                  f"若这个 ckpt 是旧版存的，属正常，重存一次即可。")
        # ★★ 2026-09-26 08:1x：恢复**数据游标**（§10.133 的真因 #2）。
        #   第一个真因是"排列种子取自全局 RNG"（已用专用 generator 修掉）；
        #   第二个真因是游标：续跑时 global_step 已经是 30，但原来的
        #   `for batch in loader` 会**从头**重新洗牌 ⇒ 吃到的是排列的第 1 个样本。
        #   这里把"已经消费了多少个样本"读回来，训练循环用它去切排列。
        global_cursor = int(_st.get("cursor", global_step))
        _cur_src = "ckpt.cursor" if "cursor" in _st else f"回退到 step={global_step}"
        if args.cursor_override >= 0:
            global_cursor = int(args.cursor_override)
            _cur_src = f"--cursor_override={args.cursor_override}"
        print(f"[resume] 数据游标 = {global_cursor} 个样本（来源: {_cur_src}）"
              f" ⇒ 从排列的第 {global_cursor + 1} 个样本接着吃")
        if args.order_only:
            print(f"[rngfp] B: 恢复完（A→B 若不同 ⇒ 恢复生效）    = {_rng_fp()}", flush=True)
        print(f"[resume] 从 step {global_step} 继续，目标 {args.steps}")
    else:
        print("[resume] 没有 trainer_state.pt ⇒ 只载权重，步数从 0 重算（会重复训一段）")
    if global_step >= args.steps:
        print(f"[resume] step {global_step} 已达目标 {args.steps}，无需再跑")
        sys.exit(0)

bar = tqdm(total=args.steps, desc="train")
t_start = time.perf_counter()
times = []
prior_iter = iter(prior_loader) if prior_loader is not None else None
controlnet_model = setup.extra.get("controlnet") if setup.kind == "controlnet" else None

if args.order_only:
    print(f"[rngfp] C: 训练循环开工前（B→C 若不同 ⇒ 中间有人消耗） = {_rng_fp()}", flush=True)

# ★★ 2026-09-26 08:0x：零算力数据游标探针（§10.133）。
#   只迭代 loader、对每个样本算 sha256 指纹（**不做任何前向/反向**），然后退出。
#   它的作用是让"续跑的数据顺序是否等于一口气跑"变成一个**秒级**可判定的问题，
#   而不是靠 12 分钟的训练 + loss 曲线反推。
if args.order_only:
    _ol = args.order_log
    if _ol and not os.path.isabs(_ol):
        _ol = os.path.join(OUT_DIR, _ol)
    if _ol:
        os.makedirs(os.path.dirname(_ol) or ".", exist_ok=True)
    _of = open(_ol, "w", encoding="utf-8") if _ol else None

    def _ids_of(_batch):
        return [hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()[:16]
                for x in _batch["latents"]]

    # ★★ 在**续跑的第一个 batch** 上，按训练循环的真实顺序复现那两次抽样：
    #     noise = torch.randn_like(latents)  然后 timesteps = torch.randint(0, NTT, (B,))
    #   目的：把"E 的 step31 noise 为什么与 A 不同"钉死在**这一步**上。
    def _probe_first_draw(_batch, _before_iter, _after_iter):
        if not args.resume:
            return
        _lat = _batch["latents"]
        print(f"[rngfp]   进循环前 = {_before_iter}", flush=True)
        print(f"[rngfp]   迭代器产出第一个 batch 之后 = {_after_iter}", flush=True)
        print(f"[rngfp]   ids_of 之后 / 画 noise 之前 = {_rng_fp()}", flush=True)
        print(f"[rngfp]   latents shape = {tuple(_lat.shape)} dtype={_lat.dtype} "
              f"contig={_lat.is_contiguous()}", flush=True)
        _n = torch.randn_like(_lat)
        print(f"[rngfp]   画出的 noise = "
              f"{hashlib.sha256(_n.contiguous().numpy().tobytes()).hexdigest()[:12]}"
              f"   （A.step31=bb66473290cd, E.step31=1b11003faf70）", flush=True)
        _t = torch.randint(0, noise_sched.config.num_train_timesteps,
                           (_lat.shape[0],)).long()
        print(f"[rngfp]   画完 timestep 后 = {_rng_fp()}", flush=True)

    _t0 = time.perf_counter()
    # ① "一口气跑"侧：用与训练循环**同一个**取批函数（step0=0）。
    _rows = []
    for _b in _loader_subset(0, args.steps):
        _rows.append({"step": len(_rows) + 1, "ids": _ids_of(_b),
                      "prompts": [str(p) for p in _b["prompt"]]})
    print(f"[order] ① 一口气侧：{len(_rows)} 个 batch，step1 ids = {_rows[0]['ids']}")
    # ② "续跑"侧：从 `--resume` 那个 ckpt 的游标位置接着取（这正是训练循环续跑时做的事）。
    _rows2 = None
    _r2 = None
    if args.resume:
        _p2 = os.path.join(args.resume, "trainer_state.pt")
        if not os.path.exists(_p2):
            print(f"[order] !! {_p2} 不存在 ⇒ 无法模拟续跑侧")
        else:
            _s2 = torch.load(_p2, map_location="cpu", weights_only=False)
            # ★ 用**已经解析好的** global_cursor（它已经把 --cursor_override 与
            #   "回退到 step" 都算进去了），不要在这里重新读一遍 —— 否则打印出来的
            #   游标会和训练循环实际用的那个不是同一个数（这个显示 bug 我已经踩过一次）。
            _cur = int(global_cursor)
            _r2 = f"游标={_cur}（start_step={_s2.get('step', '?')}）"
            _rows2 = []
            _before_iter = _rng_fp()
            _dl2 = _loader_subset(_cur, args.steps - _cur)
            if args.order_full:
                print(f"[rngfp]   _loader_subset 返回后 = {_rng_fp()}", flush=True)
            _it = iter(_dl2)
            _after_iter = _rng_fp()
            if args.order_full:
                print(f"[rngfp]   next(iter) 拿到第一个 batch 后 = {_rng_fp()}", flush=True)
            _b1 = next(_it)
            for _b in [_b1]:
                _rows2.append({"step": _cur + len(_rows2) + 1, "ids": _ids_of(_b),
                               "prompts": [str(p) for p in _b["prompt"]]})
                if args.order_full:
                    _probe_first_draw(_b, _before_iter, _after_iter)
            for _b in _it:
                _rows2.append({"step": _cur + len(_rows2) + 1, "ids": _ids_of(_b),
                               "prompts": [str(p) for p in _b["prompt"]]})
            print(f"[order] ② 续跑侧（{_r2}）：{len(_rows2)} 个 batch，"
                  f"step{_cur + 1} ids = {_rows2[0]['ids']}")
    # ★★ 对照必须按【绝对步号】对齐，不能按"列表下标"对齐（§10.133，我自己踩过的显示 bug）：
    #   ① 是 step 1..N，② 是 step (cur+1)..N ⇒ 直接 zip 会把 ② 的 step31 和 ① 的 step1 比，
    #   于是永远报"全部不同"。这里按 step 号建字典再比共同步号。
    _map1 = {r["step"]: r["ids"] for r in _rows}
    _map2 = None if _rows2 is None else {r["step"]: r["ids"] for r in _rows2}
    _eq = None
    _ndiff = None
    if _map2 is not None:
        _common = sorted(set(_map1) & set(_map2))
        _ndiff = sum(1 for st in _common if _map1[st] != _map2[st])
        _eq = (_ndiff == 0) and len(_common) > 0
        print(f"[order] ★ 两侧数据顺序一致 = {_eq}   比了 {len(_common)} 个共同步号，"
              f"不同的 = {_ndiff}")
        if _common and not _eq:
            for st in _common:
                if _map1[st] != _map2[st]:
                    print(f"[order]   第一个不同的 step = {st}")
                    print(f"[order]     一口气 ids = {_map1[st]}")
                    print(f"[order]     续跑   ids = {_map2[st]}")
                    break
    for _tag, _rr in (("①", _rows), ("②", _rows2)):
        if _rr and _of:
            for _row in _rr:
                _row["arm"] = _tag
                _of.write(json.dumps(_row, ensure_ascii=False) + "\n")
    if _of:
        _of.close()
    print(f"[order] {len(_rows) + (len(_rows2) if _rows2 else 0)} 行已落盘"
          f"（{time.perf_counter() - _t0:.2f}s，零前向）-> {_ol or '(未落盘)'}")
    if args.resume:
        print(f"[rngfp] D: 迭代完 loader 之后              = {_rng_fp()}", flush=True)
    sys.exit(0)


def _sd_step(noisy, timesteps, emb=None, extra_emb=None):
    """sd_denoise 通用一前向；extra_emb 供 ip_adapter 传 (text, ip) 元组。"""
    if extra_emb is not None:
        return unet(noisy, timesteps, extra_emb).sample
    return unet(noisy, timesteps, emb).sample


def _controlnet_step(noisy, timesteps, emb, control_lat):
    """ControlNet 训练一前向：controlnet 输出残差喂给 unet。"""
    out = controlnet_model(noisy, timesteps, encoder_hidden_states=emb,
                           controlnet_cond=control_lat, conditioning_scale=1.0)
    if hasattr(out, "to_tuple"):
        down_res, mid_res = out.to_tuple()
    else:
        down_res, mid_res = out.down_block_res_samples, out.mid_block_res_sample
    return unet(noisy, timesteps, emb,
                down_block_additional_residuals=list(down_res),
                mid_block_additional_residual=mid_res).sample


# ★★ 2026-09-26 09:1x：训练侧带宽探针（§10.134）。
#   放在**两个 _step 函数定义之后、训练循环之前**，并跑完就退出 ——
#   ★ 第一次写时插在了 `_sd_step` 定义**之前**，于是 `NameError: _sd_step is not defined`。
#     教训：往一个长脚本里插代码块，要先确认它依赖的名字**已经定义过**（不是"文件里存在"）。
#   跑完必须退出：探针步骤会做真的 `opt.step()`，会改模型状态 ⇒ 继续训练会污染曲线。
if args.bw_probe > 0:
    sys.path.insert(0, r"D:\work\membw")
    import bw_probe as _bwp

    _bp_it = max(1, int(args.bw_probe))

    def _one_step(_i):
        """一个**真实**微步（与训练循环同一条路径，含 AdamW8bit 的 opt.step()）。"""
        _b = None
        for _bb in _loader_subset(global_cursor + _i, 1):
            _b = _bb
        _lat = _b["latents"]
        _pr = list(_b["prompt"])
        _emb = torch.stack([_encode_text_emb(p, text_encoder, tokenizer, cacheable)
                            for p in _pr])
        if _emb.dim() == 4:
            _emb = _emb.squeeze(1)
        _noise = torch.randn_like(_lat)
        _ts = torch.randint(0, noise_sched.config.num_train_timesteps,
                            (_lat.shape[0],)).long()
        _noisy = noise_sched.add_noise(_lat, _noise, _ts)
        _pred = _sd_step(_noisy, _ts, _emb)
        _loss = torch.nn.functional.mse_loss(_pred, _noise)
        del _pred
        _loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(_loss.item())

    print(f"\n[§10.134] 带宽探针启动：{_bp_it} 个真实微步"
          f"（threads={TH}, batch={args.batch}, res={args.res}, rank={args.rank}）",
          flush=True)

    # ① 不带 profiler 的净墙钟（profiler 有 10~40% 开销，用它算利用率会高估）
    for _i in range(1):
        _one_step(-1 - _i)
    _t0 = time.perf_counter()
    for _i in range(_bp_it):
        _one_step(_i)
    _wall_total = time.perf_counter() - _t0
    _wall = _wall_total / _bp_it
    print(f"[§10.134] 净墙钟（无 profiler）：{_wall * 1000:.1f} ms/步"
          f"（{_bp_it} 步共 {_wall_total:.1f} s）", flush=True)

    # ② 带 profiler 的算子级（profiler 会拖慢，只用来看**占比**）
    _res = _bwp.probe_operators(lambda: _one_step(1000), warmup=0, iters=_bp_it,
                                threads=TH)
    _res["wall_s_per_iter_clean"] = _wall

    _n_par = sum(p.numel() for p in unet.parameters())
    _n_tr = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    _extra = {
        "UNet 参数量": f"{_n_par / 1e6:.1f} M（可训练 {_n_tr / 1e6:.3f} M = LoRA）",
        "净墙钟 ms/步": f"{_wall * 1000:.1f}",
        "本机上限（实测）": (f"DRAM {_bwp.CEILING['dram_gbs_nt']} GB/s(NT) / "
                         f"{_bwp.CEILING['dram_gbs_plain']} GB/s(普通)  "
                         f"GEMM {_bwp.CEILING['gemm_gflops_4096']}~"
                         f"{_bwp.CEILING['gemm_gflops_proj']} GFLOP/s  "
                         f"交叉点 {_bwp.CEILING['crossover_flop_per_byte']} FLOP/byte"),
    }
    _txt = _bwp.report(_res, title=f"SD1.5 UNet+LoRA 训练（{_bp_it} 步）", extra=_extra)

    if args.bw_probe_json:
        _jp = args.bw_probe_json
        if not os.path.isabs(_jp):
            _jp = os.path.join(OUT_DIR, _jp)
        os.makedirs(os.path.dirname(_jp) or ".", exist_ok=True)
        with open(_jp, "w", encoding="utf-8") as _jf:
            json.dump({"wall_s_per_iter_clean": _wall, "res": _res,
                       "ceiling": _bwp.CEILING, "report": _txt,
                       "params": _n_par, "trainable": _n_tr}, _jf,
                      ensure_ascii=False, indent=1)
        print(f"[§10.134] 探针结果 -> {_jp}", flush=True)
    print("[§10.134] 探针模式结束（未做正式训练）", flush=True)
    sys.exit(0)


while global_step < args.steps:
    # ★★ 2026-09-26 08:1x：改用「排列 + 游标」取批（§10.133 真因 #1/#2）。
    #   原来这里是 `for batch in loader:` —— 每次 `iter()` 都从头重新洗牌，
    #   于是续跑（global_step=30）吃到的是排列的**第 1 个**样本，而不是第 31 个；
    #   再加上排列种子取自全局 RNG（首跑随机初始化 LoRA 会消耗它），
    #   两条曲线必然从续跑的第一个微步起就走在**完全不同的数据**上（实测样本交集 0/120）。
    for batch in _loader_subset(global_cursor, args.steps - global_step):
        if balancer is not None:
            balancer.update_step()   # 每步检查内存，超过阈值卸载冷参数
        global_cursor += args.batch
        latents = batch["latents"]
        prompts = list(batch["prompt"])
        # 数据身份指纹（§10.133 的取证手段）：latents 可能是 [C,H,W]（batch=1）或 [B,C,H,W]。
        _ids_all = [hashlib.sha256(_x.contiguous().numpy().tobytes()).hexdigest()[:12]
                    for _x in (latents if latents.dim() == 4 else latents.unsqueeze(0))]
        emb = torch.stack([_encode_text_emb(p, text_encoder, tokenizer, cacheable) for p in prompts])
        if emb.dim() == 4:
            emb = emb.squeeze(1)

        noise = torch.randn_like(latents)
        NTT = noise_sched.config.num_train_timesteps
        if "lat_deg" in batch:
            # ★配对模式。普通做法是 noisy=f(真图)，回归目标是所加的 noise；
            #   但那套状态分布是「真图+噪声」，而推理时 img2img 的起点是
            #   「VQ 重建图 + 噪声」。这里把状态改成推理时真正会遇到的：
            #
            #       x_t = sqrt(ac_t) * z_deg + sqrt(1-ac_t) * eps
            #
            #   然后要求模型的 x0 预测等于【真图】z_tgt。由
            #       x0_hat = (x_t - sqrt(1-ac) * eps_pred) / sqrt(ac)
            #   解 x0_hat = z_tgt 得回归目标
            #       eps_target = (x_t - sqrt(ac) * z_tgt) / sqrt(1-ac)
            #   z_deg == z_tgt 时它恒等于 eps ⇒ 与普通目标自洽，是它的推广。
            #   t 下限的意义：t=10 时 1/sqrt(1-ac)≈28 会把目标放大到病态。
            lat_deg = batch["lat_deg"].to(latents.dtype)
            timesteps = torch.randint(args.paired_tlo, NTT, (latents.shape[0],)).long()
            ac = noise_sched.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            noisy = ac.sqrt() * lat_deg + (1.0 - ac).sqrt() * noise
            eps_target = (noisy - ac.sqrt() * latents) / (1.0 - ac).sqrt()
        else:
            timesteps = torch.randint(0, NTT, (latents.shape[0],)).long()
            noisy = noise_sched.add_noise(latents, noise, timesteps)
            eps_target = noise

        t0 = time.perf_counter()
        if setup.kind == "controlnet":
            # 控制图（像素级, 3通道, 缩放到 latent 尺寸） + controlnet forward
            img_paths = [base_ds.samples[i % len(base_ds.samples)][0] for i in range(len(prompts))]
            ctrl = torch.stack([_control_latents(p) for p in img_paths])  # [B,3,H/8,W/8]
            if ctrl.dim() == 3:
                ctrl = ctrl.unsqueeze(0)
            ctrl = ctrl.to(latents.dtype)
            noise_pred = _controlnet_step(noisy, timesteps, emb, ctrl)
        elif setup.kind == "ip_adapter":
            # 图像特征 → image_proj → (text, ip) 元组传给 unet
            img_paths = [base_ds.samples[i % len(base_ds.samples)][0] for i in range(len(prompts))]
            ip_feats = torch.stack([_encode_image_feature(p) for p in img_paths])  # [B,num_tokens,dim]
            # 传给 UNet 的 cross-attn：text_emb [B,77,dim] + ip [B,num_tokens,dim]
            noise_pred = _sd_step(noisy, timesteps, extra_emb=(emb, ip_feats))
        else:
            noise_pred = _sd_step(noisy, timesteps, emb)
        loss = torch.nn.functional.mse_loss(noise_pred, eps_target)
        del noise_pred

        # DreamBooth 先验保留
        if prior_iter is not None:
            pb = next(prior_iter, None)
            if pb is None:
                prior_iter = iter(prior_loader)
                pb = next(prior_iter)
            p_emb = _encode_text_emb(args.class_prompt, text_encoder, tokenizer, cacheable)
            if p_emb.dim() == 4:
                p_emb = p_emb.squeeze(0)
            p_noise = torch.randn_like(pb["latents"])
            p_ts = torch.randint(0, noise_sched.config.num_train_timesteps, (1,)).long()
            p_noisy = noise_sched.add_noise(pb["latents"], p_noise, p_ts)
            p_pred = unet(p_noisy, p_ts, p_emb.unsqueeze(0)).sample
            loss = loss + args.prior_loss_weight * torch.nn.functional.mse_loss(p_pred, p_noise)

        loss.backward()
        times.append(time.perf_counter() - t0)

        if (global_step + 1) % args.grad_accum == 0:
            if args.warmup > 0:
                # 在 opt.step() 之前设 LR（sched 是 constant，设完再 step 也只会被它覆盖回常数，
                # 所以 --warmup>0 时下面那句 sched.step() 要跳过）
                _f = set_lr(global_step)
                if global_step % 50 == 0:
                    print(f"\n[lr] step {global_step}  factor={_f:.4f}  lr={args.lr * _f:.3e}")
            opt.step()
            if args.warmup <= 0:
                sched.step()
            opt.zero_grad()
            opt_steps += 1

        global_step += 1
        bar.update(1)
        if _LOSS_F:
            _LOSS_F.write(json.dumps({
                "step": global_step, "loss": loss.item(),
                "lr": opt.param_groups[0]["lr"], "t": time.time(),
                "ids": _ids_all,
                "cur": global_cursor,
                "rng": hashlib.sha256(
                    torch.get_rng_state().numpy().tobytes()).hexdigest()[:12],
                "noise": hashlib.sha256(
                    noise.contiguous().numpy().tobytes()).hexdigest()[:12],
            }) + "\n")
            _LOSS_F.flush()
        if times:
            bar.set_description(f"loss {loss.item():.4f} | {times[-1]:.2f}s/步")
        if global_step % 10 == 0 and args.min_free_gb > 0:
            # ★★ 系统级内存硬上限（2026-09-25 加）★★
            # 起因：并发跑别的任务时系统 RAM 被顶到 15.29/15.37 GB + swap 2.21 GB，
            # 把用户的 explorer.exe 饿到假死，训练自己也死在 step 695（0xC0000005）。
            # 先前 [tck.mem] 只【报告】不【阻止】，balancer 也没兜住。
            # 这里改成真上限：宁可自己退出，也不能饿死系统。
            # 每 10 步查一次（psutil 查询是微秒级，开销可忽略；50 步的窗口太长）。
            _free = psutil.virtual_memory().available / 1e9
            if _free < args.min_free_gb:
                _ck = os.path.join(OUT_DIR, "checkpoint-memguard-%d" % global_step)
                print("\n[mem-guard] 可用内存 %.2f GB < %.2f GB —— 主动退出，避免饿死系统。"
                      % (_free, args.min_free_gb), flush=True)
                try:
                    if setup.kind == "controlnet" and controlnet_model is not None:
                        controlnet_model.save_pretrained(_ck)
                    elif hasattr(unet, "save_pretrained"):
                        unet.save_pretrained(_ck)
                    # ★★ 2026-09-26 07:1x 修（§10.117②）：守卫这条路径原来**只存权重**，
                    #   不存 trainer_state.pt —— 而正常 checkpoint 是存的。
                    #   ⇒ 后果：从守卫留下的 checkpoint 续训时会走"没有 trainer_state.pt"那条分支：
                    #       [resume] 没有 trainer_state.pt ⇒ 只载权重，步数从 0 重算（会重复训一段）
                    #     也就是**补救恰好不能从"守卫实际产出的那个 checkpoint"出发**。
                    #   取证：三个独立实例都缺它（checkpoint-memguard-30 / -1450 / -150），
                    #        而同目录的正常 checkpoint-30 有。
                    #   ⇒ 这里与正常 ckpt 用**同一个格式**（step/opt_steps/opt/sched）。
                    try:
                        # ★ 用共用的 _trainer_state()（含 RNG）—— 不在这里再写一遍 dict，
                        #   那正是两条保存路径会漂移的原因。
                        torch.save(_trainer_state(), os.path.join(_ck, "trainer_state.pt"))
                        print("[mem-guard] 训练器状态已一并保存（trainer_state.pt）⇒ 可直接 --resume",
                              flush=True)
                    except Exception as _e2:
                        print("[mem-guard] !! trainer_state.pt 存盘失败（续训会退化为重跑）: %s"
                              % str(_e2)[:160], flush=True)
                    print("[mem-guard] 已存 %s" % _ck, flush=True)
                except Exception as _e:
                    print("[mem-guard] 存盘失败: %s" % str(_e)[:200], flush=True)
                sys.exit(3)
        if global_step % 50 == 0:
            avg = sum(times[-50:]) / max(1, len(times[-50:]))
            mem = psutil.Process().memory_info().rss / 1048576
            print(f"\n[st] step {global_step}/{args.steps} avg={avg:.2f}s/step RSS={mem:.0f}MB "
                  f"elapsed={(time.perf_counter() - t_start) / 60:.1f}m "
                  f"sysfree={psutil.virtual_memory().available / 1e9:.2f}GB")
        if args.sample_every and global_step % args.sample_every == 0:
            generate_samples(global_step)
        if args.ckpt_every and global_step % args.ckpt_every == 0:
            ck = os.path.join(OUT_DIR, f"checkpoint-{global_step}")
            if setup.kind == "controlnet" and controlnet_model is not None:
                controlnet_model.save_pretrained(ck)
            elif hasattr(unet, "save_pretrained"):
                unet.save_pretrained(ck)
            # ★ 与权重同目录存训练器状态，供 --resume 用
            try:
                # ★ 与守卫那条路径共用 _trainer_state()（含 RNG）—— 避免两条路径漂移
                torch.save(_trainer_state(), os.path.join(ck, "trainer_state.pt"))
            except Exception as _e:
                print(f"\n[ckpt] !! trainer_state.pt 存盘失败: {str(_e)[:160]}")
            print(f"\n[ckpt] {ck}")
        if global_step >= args.steps:
            break

bar.close()

# ============== 收尾 ==============
final = os.path.join(OUT_DIR, "final")
os.makedirs(final, exist_ok=True)
if args.method == "ti":
    # 导出学到的占位 token embedding
    emb = text_encoder.get_input_embeddings().weight.data
    pid = setup.extra["placeholder_token_id"]
    torch.save({"token": args.placeholder_token, "token_id": pid,
                "embedding": emb[pid].clone()}, os.path.join(final, "learned_embeds.bin"))
    print(f"[save] TI embedding -> {final}/learned_embeds.bin")
elif setup.kind == "controlnet" and controlnet_model is not None:
    controlnet_model.save_pretrained(final)
    print(f"[save] ControlNet -> {final}")
elif setup.kind == "ip_adapter":
    # 保存 image_proj + adapter 状态
    torch.save({
        "image_proj": image_proj.state_dict() if image_proj is not None else None,
        "token": args.placeholder_token, "num_tokens": args.num_tokens, "scale": args.ip_scale,
    }, os.path.join(final, "ip_adapter_state.bin"))
    print(f"[save] ip_adapter 状态 -> {final}/ip_adapter_state.bin")
elif hasattr(unet, "save_pretrained"):
    unet.save_pretrained(final)
    print(f"[save] -> {final}")
generate_samples("final")
if times:
    avg = sum(times) / len(times)
    print(f"[done] {args.steps} 步完成, 平均 {avg:.2f}s/步, 总耗时 {(time.perf_counter() - t_start) / 60:.1f} 分钟")
if mon_ev is not None:
    try:
        suspend_mem_monitor(mon_ev)
    except Exception:
        pass
if balancer is not None:
    balancer.cleanup()
    print("[flash] disk_balancer 已清理")
