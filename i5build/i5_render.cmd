@echo off
REM Render Qwen-Image-2.1 4-bit on the i5-10400 (6C/12T, AVX2, 11.8 GB RAM).
REM
REM Identical work to the R5-4500U run -- same prompt, seed, resolution and step count
REM -- so the two machines differ only in the CPU:
REM     laptop : 1024x1024, 20 steps, seed 1234, cfg 1.0  ->  328 s/step
REM     i5     : the same, on 12 threads instead of 6
REM
REM The prompt embeddings come from qwen21_encode.py, so the text encoder (4.22 GB) is
REM never loaded here. Together with tiled VAE decoding that keeps the peak inside the
REM 7.4 GB this box has free.
set PY=D:\scoop\apps\python\current\python.exe
set RUN=D:\qwen21\run
set OMP_NUM_THREADS=12
cd /d %RUN%
echo ==================== i5 render ====================
%PY% -c "import os,platform;print('host',platform.node());print('cpus',os.cpu_count())"
%PY% -u qwen21_render.py ^
  --ckpt D:\qwen21\Qwen-Image-2.1-4bit ^
  --src D:\qwen21\Qwen-Image-2.1 ^
  --height 1024 --width 1024 --steps 20 --cfg 1.0 ^
  --threads 12 ^
  --embeds D:\qwen21\run\embeds.pt ^
  --out D:\qwen21\out ^
  --tag _i5 ^
  --prompt "a red panda astronaut floating in space, photorealistic, cinematic lighting, earth in the background, detailed fur"
echo render_exit=%ERRORLEVEL%
dir /b D:\qwen21\out
