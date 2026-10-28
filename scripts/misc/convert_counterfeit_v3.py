# -*- coding: utf-8 -*-
"""Convert Counterfeit-V3.0 (A1111/LDM single-file checkpoint) to diffusers layout.

Uses diffusers' own proven converters so key mapping stays canonical:
  - convert_ldm_unet_checkpoint  (model.diffusion_model.* -> down_blocks/up_blocks/mid_block)
  - convert_ldm_vae_checkpoint   (first_stage_model.*     -> encoder/decoder/quant_conv)
  - convert_ldm_clip_checkpoint  (cond_stage_model.transformer.* -> text_model.*)

Notes:
  * Source dtypes are mixed: unet/clip are F64, vae is F16 -> we normalize everything to fp32
    so CPU inference does not pay float64 cost.
  * Scheduler config is taken from the existing SD1.5 diffusers dir (same 1000-step linear beta).

Output:
  D:\work\textmodel\counterfeit-v3-diffusers\{unet,vae,text_encoder,tokenizer,scheduler,model_index.json}
"""
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
import os
import sys
import json
import time
import shutil

import torch
from safetensors.torch import load_file as sf_load, save_file as sf_save

from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_ldm_clip_checkpoint,
)
from transformers import CLIPTextModel, CLIPTokenizer

SRC = r"D:\work\textmodel\counterfeit-v3\Counterfeit-V3.0_fp16.safetensors"
SD15_FULL = r"D:\work\textmodel\sd15_full"
OUT = r"D:\work\textmodel\counterfeit-v3-diffusers"

UNET_CFG = dict(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                      "CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
    block_out_channels=(320, 640, 1280, 1280),
    layers_per_block=2,
    cross_attention_dim=768,
    attention_head_dim=8,
    norm_num_groups=32,
)
VAE_CFG = dict(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D",) * 4,
    up_block_types=("UpDecoderBlock2D",) * 4,
    block_out_channels=(128, 256, 512, 512),
    layers_per_block=2,
    latent_channels=4,
    norm_num_groups=32,
    sample_size=512,
)


def log(*a):
    print(*a, flush=True)


def main():
    t0 = time.time()
    log(f"[load] reading {SRC}")
    sd = sf_load(SRC)
    log(f"[load] {len(sd)} tensors in {time.time()-t0:.1f}s")

    unet_sd = {k[len("model.diffusion_model."):]: v for k, v in sd.items()
               if k.startswith("model.diffusion_model.")}
    vae_sd = {k[len("first_stage_model."):]: v for k, v in sd.items()
              if k.startswith("first_stage_model.")}
    clip_sd = {k[len("cond_stage_model.transformer."):]: v for k, v in sd.items()
               if k.startswith("cond_stage_model.transformer.")}
    log(f"[split] unet={len(unet_sd)} vae={len(vae_sd)} clip={len(clip_sd)}")

    os.makedirs(OUT, exist_ok=True)

    # ---------------- UNet ----------------
    log("[unet] building empty UNet2DConditionModel + converting keys")
    unet = UNet2DConditionModel(**UNET_CFG)
    # diffusers' convert_ldm_unet_checkpoint reads a lot of optional fields from the
    # config dict (class_embed_type, addition_embed_type, ...), so build the config from
    # the instantiated model instead of passing our minimal dict.
    cfg_full = dict(unet.config)
    cfg_full.update(UNET_CFG)
    log(f"[unet] config keys available: class_embed_type={cfg_full.get('class_embed_type')!r} "
        f"addition_embed_type={cfg_full.get('addition_embed_type')!r}")
    # convert_ldm_unet_checkpoint internally re-extracts keys starting with
    # "model.diffusion_model." -- but we already stripped that prefix, so its extraction
    # would yield an empty dict. skip_extract_state_dict=True makes it use our dict as-is
    # (it expects the stripped "input_blocks.*"/"out.*" naming, which is what unet_sd holds).
    new_unet = convert_ldm_unet_checkpoint(
        dict(unet_sd), cfg_full, path=SRC, extract_ema=False, skip_extract_state_dict=True
    )
    missing, unexpected = unet.load_state_dict(new_unet, strict=False)
    log(f"[unet] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        log("        e.g. missing:", missing[:5])
    if unexpected[:5]:
        log("        e.g. unexpected:", unexpected[:5])
    unet = unet.to(torch.float32)
    updir = os.path.join(OUT, "unet")
    unet.save_pretrained(updir, safe_serialization=True)
    log(f"[unet] saved -> {updir}")

    # ---------------- VAE ----------------
    log("[vae] building empty AutoencoderKL + converting keys")
    vae = AutoencoderKL(**VAE_CFG)
    vae_cfg_full = dict(vae.config)
    vae_cfg_full.update(VAE_CFG)
    new_vae = convert_ldm_vae_checkpoint(dict(vae_sd), vae_cfg_full)
    missing, unexpected = vae.load_state_dict(new_vae, strict=False)
    log(f"[vae] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        log("       e.g. missing:", missing[:5])
    if unexpected[:5]:
        log("       e.g. unexpected:", unexpected[:5])
    vae = vae.to(torch.float32)
    vdir = os.path.join(OUT, "vae")
    vae.save_pretrained(vdir, safe_serialization=True)
    log(f"[vae] saved -> {vdir}")

    # ---------------- Text encoder ----------------
    log("[clip] converting key names (cond_stage_model.transformer.* -> text_model.*)")
    # remap manually: HF CLIPTextModel wants 'text_model.' prefix, source has
    # 'cond_stage_model.transformer.text_model.' -> strip 'cond_stage_model.transformer.'
    # remap: source keys are already stripped to 'text_model.*' (prefix removed above).
    remapped = {}
    for k, v in clip_sd.items():
        if k == "text_model.embeddings.position_ids":
            continue  # regenerated by the model, non-persistent buffer
        remapped[k] = v.to(torch.float32)
    te = CLIPTextModel.from_pretrained(os.path.join(SD15_FULL, "text_encoder"))
    missing, unexpected = te.load_state_dict(remapped, strict=False)
    log(f"[clip] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        log("       e.g. missing:", missing[:5])
    if unexpected[:5]:
        log("       e.g. unexpected:", unexpected[:5])
    te = te.to(torch.float32)
    tdir = os.path.join(OUT, "text_encoder")
    te.save_pretrained(tdir, safe_serialization=True)
    log(f"[clip] saved -> {tdir}")

    # ---------------- tokenizer + scheduler ----------------
    # NOTE: sd15_full has no top-level tokenizer/ dir; the CLIP tokenizer files
    # (vocab.json/merges.txt/tokenizer_config.json) live next to the text encoder.
    # Matching them with the same text encoder avoids the vocab-size mismatch we hit before.
    tok_src = os.path.join(SD15_FULL, "text_encoder")
    tok_dst = os.path.join(OUT, "tokenizer")
    if not os.path.isdir(tok_dst):
        os.makedirs(tok_dst, exist_ok=True)
        copied = []
        for fn in ("vocab.json", "merges.txt", "tokenizer_config.json",
                   "special_tokens_map.json"):
            s = os.path.join(tok_src, fn)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(tok_dst, fn))
                copied.append(fn)
        log(f"[copy] tokenizer <- {tok_src}: {copied}")
        # sanity: vocab size must match SD1.5 CLIP (49408)
        try:
            with open(os.path.join(tok_dst, "vocab.json"), encoding="utf-8") as f:
                vs = len(json.load(f))
            log(f"[copy] tokenizer vocab size = {vs} (expect 49408)")
        except Exception as e:
            log(f"[copy] vocab check failed: {e}")

    sch_src = os.path.join(SD15_FULL, "scheduler")
    sch_dst = os.path.join(OUT, "scheduler")
    if os.path.isdir(sch_src) and not os.path.isdir(sch_dst):
        shutil.copytree(sch_src, sch_dst)
        log(f"[copy] scheduler -> {sch_dst}")

    # ---------------- model_index.json ----------------
    mi = {
        "_class_name": "StableDiffusionPipeline",
        "_diffusers_version": "0.27.0",
        "scheduler": ["diffusers", "DDIMScheduler"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "safety_checker": [None, None],
        "feature_extractor": [None, None],
    }
    with open(os.path.join(OUT, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump(mi, f, indent=2, ensure_ascii=False)
    log(f"[done] all written to {OUT} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
