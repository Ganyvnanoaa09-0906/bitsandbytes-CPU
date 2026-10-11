@echo off
REM ============================================================
REM  Gate between step 1 and step 2 of the 24x24 path.
REM
REM  When the 24x24 tokenizer finishes training, produce the SAME kind of
REM  reconstruction comparison that started this whole thread (vq_compare.png:
REM  originals on top, VQ reconstruction below). Then stop and let the eye decide.
REM
REM  Why stop rather than chain straight into the AR:
REM    the 16x16 tokenizer gives 1024 pixels per token and the reconstruction is
REM    visibly mushy. 24x24 gives 456. Whether that is ENOUGH to be worth 20 hours
REM    of AR training is a judgement about images, not a number, and it is a
REM    judgement that costs nothing to make here. Chaining blindly would risk
REM    spending a day to rediscover that the ceiling is still too low.
REM
REM  Order: wait for the training process -> build the comparison -> exit.
REM
REM  THIS FILE MUST STAY PURE ASCII.
REM ============================================================
setlocal
cd /d D:\work\bitsandbytes-CPU
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com

set VQPID=15800
echo [%DATE% %TIME%] gate waiting for vq32 (pid %VQPID%) >> i5build\vq32_gate.log

:waitloop
tasklist /FI "PID eq %VQPID%" 2>nul | find /I "%VQPID%" >nul
if not errorlevel 1 (
    timeout /t 120 /nobreak >nul
    goto waitloop
)
echo [%DATE% %TIME%] vq32 finished >> i5build\vq32_gate.log

REM make the reconstruction comparison with whatever checkpoints exist
python -u vq_recon_compare.py ^
  --ckpt "i5build\vq32" --data "D:\work\cloud_salvage\DATASET\data" ^
  --out "i5build\vq32_compare.png" --n 4 --size 384 ^
  >> i5build\vq32_gate.log 2>&1

if exist "i5build\vq32_compare.png" (
    echo [%DATE% %TIME%] wrote vq32_compare.png >> i5build\vq32_gate.log
) else (
    echo [%DATE% %TIME%] FAILED to write comparison >> i5build\vq32_gate.log
)
echo [%DATE% %TIME%] gate done >> i5build\vq32_gate.log
