@echo off
REM ============================================================
REM  VQGAN training driver: restart until the step budget is complete.
REM
REM  The trainer already knows about its own memory growth and how to survive it:
REM
REM    --trim_every N   gc.collect() every N steps. The author measured ~17 MB/step
REM                     without it. Note their companion fix, trim_heap(), calls
REM                     malloc_trim through libc.so.6 and therefore does nothing on
REM                     Windows -- gc.collect() is the lever available here, which
REM                     is why this machine hits the wall the author's Linux
REM                     container does not.
REM    --max_rss_gb X   save and exit VOLUNTARILY once RSS passes X, so a driver can
REM                     resume. This is the difference between a clean exit and the
REM                     crash: three earlier runs died at 560-660 steps with
REM                     0xC0000005 in c10.dll and no traceback, because nothing
REM                     stopped them before the native allocator gave out.
REM    --resume         restores model, discriminator, optimiser state and step.
REM
REM  Step budget: train_vqvae_gan counts --steps from the resumed step, so the
REM  driver keeps relaunching with the same --steps value until the trainer
REM  reports it has reached it.
REM
REM  THIS FILE MUST STAY PURE ASCII.
REM ============================================================
setlocal enabledelayedexpansion
cd /d D:\work\bitsandbytes-CPU
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com

set OUT=i5build\vqgan32
set LOG=i5build\vqgan32_drive.log
set BUDGET=8000
set ROUND=0

echo [%DATE% %TIME%] driver start, budget %BUDGET% steps >> "%LOG%"

:round
set /a ROUND+=1
echo [%DATE% %TIME%] round !ROUND! starting >> "%LOG%"

set RES=
if exist "%OUT%\ckpt_last.pt" set RES=--resume "%OUT%\ckpt_last.pt"

python -u train_vqvae_gan.py ^
  --data "D:\work\cloud_salvage\DATASET\data" ^
  --out "%OUT%" ^
  --size 256 --ch 64,128,256 --use_attn ^
  --batch 4 --steps %BUDGET% ^
  --lr 2e-4 --lr_d 2e-4 ^
  --embed_dim 64 --n_codes 1024 ^
  --ckpt_every 200 --trim_every 50 --max_rss_gb 8.0 ^
  --threads 5 !RES! ^
  >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo [%DATE% %TIME%] round !ROUND! ended rc=%RC% >> "%LOG%"

REM stop when the trainer says the budget is done
findstr /C:"[gan] done steps=" "%LOG%" | findstr /C:"steps=%BUDGET%" >nul
if not errorlevel 1 (
    echo [%DATE% %TIME%] budget reached, driver stopping >> "%LOG%"
    goto end
)
if !ROUND! GEQ 40 (
    echo [%DATE% %TIME%] round cap hit, driver stopping >> "%LOG%"
    goto end
)
timeout /t 15 /nobreak >nul
goto round

:end
echo [%DATE% %TIME%] driver done after !ROUND! rounds >> "%LOG%"
