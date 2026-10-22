# Clean-window measurement plan (run only after lead confirms python=0/ffmpeg=0/load<30).
# 1) upload_weights minimal control: normal 2 steps vs skip-norm-upload 2 steps
# 2) 20-step steady state (last 10 steps median; STEPTIME series)
# 3) one-layer per-op OpenCL event profiler

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location D:\work\bitsandbytes-CPU
$out = 'D:\work\research\bench_results'
$meta = Join-Path $out 'clean_window_meta.txt'

function Snap($tag) {
    $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB, 0)
    $py = (Get-Process python -ErrorAction SilentlyContinue).Count
    $ff = (Get-Process ffmpeg -ErrorAction SilentlyContinue).Count
    $z = (Get-Process zig -ErrorAction SilentlyContinue).Count
    "$tag $(Get-Date -Format o) load=$cpu% freeMB=$free python=$py ffmpeg=$ff zig=$z" | Add-Content $meta -Encoding UTF8
    if ($cpu -gt 30 -or $py -gt 0 -or $ff -gt 0 -or $z -gt 0) {
        Write-Output "WINDOW_NOT_CLEAN: $tag load=$cpu py=$py ff=$ff zig=$z"
    }
}

Remove-Item $meta -ErrorAction SilentlyContinue
Snap 'BEFORE'

# --- 1. upload control ---
$env:STEPS = '2'; $env:NO_NORM_UPLOAD = '0'; $env:NO_NORM_HOST = '0'
python -u gt1030\full_gpu_train_sgemm.py *> (Join-Path $out 'upload_normal.log')
$env:NO_NORM_UPLOAD = '1'
python -u gt1030\full_gpu_train_sgemm.py *> (Join-Path $out 'upload_skip.log')

# --- 2. steady 20 steps ---
$env:STEPS = '20'; $env:NO_NORM_UPLOAD = '0'; $env:NO_NORM_HOST = '0'
python -u gt1030\full_gpu_train_sgemm.py *> (Join-Path $out 'ar_sgemm_steady20_clean.log')

# --- 3. per-op event profiler ---
python -u gt1030\bench_layer_ops.py *> (Join-Path $out 'layer_ops_profile.log')

Snap 'AFTER'
Write-Output 'DONE. Check upload_normal.log / upload_skip.log / ar_sgemm_steady20_clean.log / layer_ops_profile.log'
