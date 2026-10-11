@echo off
REM ============================================================
REM  Two jobs, in order:
REM
REM  A. The clean comparison the last one failed to be. Yesterday's watchdog
REM     changed TWO variables at once -- with-condition ran at guidance 3.0 and
REM     without ran at 1.0 -- so the visible difference could be CFG rather than
REM     the condition. Same checkpoint, same seed, same guidance this time; the
REM     only difference is whether --cond-pt is passed.
REM
REM  B. The scheduled-sampling run that addresses the actual complaint ("there is
REM     a humanoid shape but it is not watchable"). Report 7.10 traced that to
REM     exposure bias: teacher forcing means the model never sees its own errors
REM     (measured high-frequency gap -9.25 dB on real prefixes against -16.08 dB
REM     on its own output). Scheduled sampling feeds it its own samples during
REM     training, which is the direct treatment.
REM
REM     Resumed from the conditioned run's final checkpoint so both gains stack.
REM     --lr-min equal to --lr keeps the schedule flat: the ramp formula
REM     g_step/(ramp*total_steps) saturates on resume and otherwise drives the
REM     learning rate to zero.
REM
REM  Runs detached; logs to sched1.log / sched1.jsonl.
REM  THIS FILE MUST STAY PURE ASCII.
REM ============================================================
setlocal
cd /d D:\work\bitsandbytes-CPU
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com

set VQVAE=D:\work\cloud_salvage\CrswCc\vqvae_16x16_cmse.pt
set COND=D:\work\cloud_salvage\CrswCc\cond_clip_29113.pt
set TOK=D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt
set CKPT=i5build\loop_cond1\ckpt_last.pt
if not exist "%CKPT%" for /f "delims=" %%f in ('dir /b /o-d "i5build\loop_cond1\ckpt*.pt"') do if not defined CKPT set CKPT=i5build\loop_cond1\%%f

echo ==== A: clean comparison (same guidance, only the condition differs) ====
mkdir i5build\cond_result2 2>nul
python -u sample_ar.py --ar "%CKPT%" --vqvae "%VQVAE%" ^
  --out "i5build\cond_result2\nocond_g1.png" --n 4 --threads 6 --seed 0 --guidance 1.0 > a1.log 2>&1
python -u sample_ar.py --ar "%CKPT%" --vqvae "%VQVAE%" ^
  --out "i5build\cond_result2\cond_g1.png" --n 4 --threads 6 --seed 0 --guidance 1.0 ^
  --cond-pt "%COND%" --cond-row 0 >> a1.log 2>&1
python -u -c "from PIL import Image; import os; d=r'i5build\cond_result2'; a=Image.open(os.path.join(d,'nocond_g1.png')).convert('RGB'); b=Image.open(os.path.join(d,'cond_g1.png')).convert('RGB'); h=max(a.height,b.height); a=a.resize((int(a.width*h/a.height),h)); b=b.resize((int(b.width*h/b.height),h)); o=Image.new('RGB',(a.width+b.width+16,h),(255,255,255)); o.paste(a,(0,0)); o.paste(b,(a.width+16,0)); o.save(os.path.join(d,'compare.png')); print('wrote', o.size)" >> a1.log 2>&1
echo ==== A done ====

echo ==== B: scheduled sampling, 3000 steps ====
python -u train_ar_v2.py ^
  --tokens "%TOK%" --vqvae "%VQVAE%" ^
  --out "i5build\loop_sched1" ^
  --stages "256:27500" ^
  --batch "256:8" ^
  --cond-mode adaln --cond-pt "%COND%" --cond-dropout 0.1 ^
  --resume "%CKPT%" ^
  --loop_random --loop_L_max 3 --use_bos ^
  --lr 5e-5 --lr-min 5e-5 ^
  --sched-sample 0.3 --sched-ramp 0.4 ^
  --ckpt_every 500 --val_every 250 --val_frac 0.02 ^
  > "i5build\loop_sched1.log" 2> "i5build\loop_sched1.err"
echo exit=%ERRORLEVEL%
echo ==== B done ====
