@echo off
REM ============================================================
REM  Does the trained LoRA do anything? Same everything, only --lora differs.
REM
REM  Why this is necessary: the first generation produced a real anime portrait,
REM  which is a genuine result -- but the LoRA was trained on cosplay PHOTOS, so
REM  an anime-looking output is equally consistent with "the LoRA never loaded and
REM  AnimateDiff's anime prior produced this". My peft_config check printed 0 -> 0,
REM  which measured the wrong object (peft_config lives on the pipeline; the
REM  adapter is injected into the unet), so it settles nothing.
REM
REM  Frames halved to 8 to keep each arm under ~10 minutes; both arms use
REM  identical settings, which is what makes the comparison valid.
REM
REM  THIS FILE MUST STAY PURE ASCII.
REM ============================================================
setlocal
cd /d D:\work\bitsandbytes-CPU
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set ADAPTER=D:\work\bitsandbytes-CPU\models\animatediff-motion-adapter-v1-5-2
set LORA=D:\work\bitsandbytes-CPU\i5build\adiff_run1\diffusers_lora.safetensors

echo ==== arm A: WITHOUT the trained LoRA ====
python -u anime_video_e2e.py gen --frames 8 --res 256 --steps 20 --seed 1234 ^
  --adapter "%ADAPTER%" > i5build\ab_nolora.log 2>&1
if exist "D:\work\cloud_results\anime_e2e\lr_frames" (
    rmdir /s /q "D:\work\cloud_results\anime_e2e\lr_frames_nolora" 2>nul
    move "D:\work\cloud_results\anime_e2e\lr_frames" "D:\work\cloud_results\anime_e2e\lr_frames_nolora" >nul
)

echo ==== arm B: WITH the trained LoRA ====
python -u anime_video_e2e.py gen --frames 8 --res 256 --steps 20 --seed 1234 ^
  --adapter "%ADAPTER%" --lora "%LORA%" > i5build\ab_lora.log 2>&1
if exist "D:\work\cloud_results\anime_e2e\lr_frames" (
    rmdir /s /q "D:\work\cloud_results\anime_e2e\lr_frames_lora" 2>nul
    move "D:\work\cloud_results\anime_e2e\lr_frames" "D:\work\cloud_results\anime_e2e\lr_frames_lora" >nul
)

echo ==== both arms done ====
