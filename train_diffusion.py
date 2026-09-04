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
P.add_argument("--ckpt_every", type=int, default=200)
P.add_argument("--sample_every", type=int, default=0)
P.add_argument("--sample_prompt", default="")
P.add_argument("--sample_steps", type=int, default=20)
P.add_argument("--cache_dir", default="", help="潜变量缓存目录（缺省=脚本目录/cache_latents）")
P.add_argument("--no_mem_monitor", action="store_true")
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
args = P.parse_args()

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
    def __init__(self, data_dir, res, max_samples=0, concept="nsfw_art"):
        self.res = res
        meta = os.path.join(data_dir, "prompts.json")
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

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        lat = self.lc.get_or_encode(self.base.samples[i][0], self.vae, self.base.transform)
        return {"latents": lat, "prompt": item["prompt"]}


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
    # 视频：本地暂无权重，仅打印结构提示后退出
    setup = apply_method(args.method, video_transformer=None
                         if not args.video_model else _load_video_transformer(args.video_model))
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

# ============== 数据 ==============
if not args.data:
    sys.exit("[ERROR] 未指定数据集：请用 --data <目录>（目录内需 images/ 与 prompts.json，"
             "或只有 images/；也可参考测试脚本用合成图目录）")
base_ds = SDDataset(args.data, args.res, args.max_samples, args.concept)
lat_cache = LatentDiskCache(CACHE_DIR, VAE_ID, args.res)
ds = CachedLatentDataset(base_ds, lat_cache, vae)
loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)

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
text_encoder.train() if train_text else text_encoder.eval()
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
global_step = 0
opt_steps = 0
bar = tqdm(total=args.steps, desc="train")
t_start = time.perf_counter()
times = []
prior_iter = iter(prior_loader) if prior_loader is not None else None
controlnet_model = setup.extra.get("controlnet") if setup.kind == "controlnet" else None


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


while global_step < args.steps:
    for batch in loader:
        if balancer is not None:
            balancer.update_step()   # 每步检查内存，超过阈值卸载冷参数
        latents = batch["latents"]
        prompts = list(batch["prompt"])
        emb = torch.stack([_encode_text_emb(p, text_encoder, tokenizer, cacheable) for p in prompts])
        if emb.dim() == 4:
            emb = emb.squeeze(1)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_sched.config.num_train_timesteps,
                                  (latents.shape[0],)).long()
        noisy = noise_sched.add_noise(latents, noise, timesteps)

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
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
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
            opt.step()
            sched.step()
            opt.zero_grad()
            opt_steps += 1

        global_step += 1
        bar.update(1)
        if times:
            bar.set_description(f"loss {loss.item():.4f} | {times[-1]:.2f}s/步")
        if global_step % 50 == 0:
            avg = sum(times[-50:]) / max(1, len(times[-50:]))
            mem = psutil.Process().memory_info().rss / 1048576
            print(f"\n[st] step {global_step}/{args.steps} avg={avg:.2f}s/step RSS={mem:.0f}MB "
                  f"elapsed={(time.perf_counter() - t_start) / 60:.1f}m")
        if args.sample_every and global_step % args.sample_every == 0:
            generate_samples(global_step)
        if args.ckpt_every and global_step % args.ckpt_every == 0:
            ck = os.path.join(OUT_DIR, f"checkpoint-{global_step}")
            if setup.kind == "controlnet" and controlnet_model is not None:
                controlnet_model.save_pretrained(ck)
            elif hasattr(unet, "save_pretrained"):
                unet.save_pretrained(ck)
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