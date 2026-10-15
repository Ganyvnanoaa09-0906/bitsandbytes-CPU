# run_queue.ps1 -- run the CUDA-borrowing experiments in order, each waiting for a quiet machine.
#
# Serialised deliberately. The retry loop already running would collide with anything else I start,
# and two measurements sharing a machine is precisely how the numbers in 7.41 ended up 10-13% high.
# One chain, one thing at a time, each gated by the guard rather than by my own judgement of
# whether the machine looks busy -- that judgement has been wrong twice today already (testing for
# a process to *exist* when Windows Update services persist, and reading Get-Process .CPU which
# returns 0 for SYSTEM-owned processes).
#
# The queue, in the order the evidence supports:
#
#   1. register blocking     LDS reads per FMA from 0.5 to 0.25, against a 2x gap
#   2. swizzle / padding     two 2-way bank conflicts identified by hand from the bank mapping
#   3. CPU attention         is torch's CPU flash attention already good, or is there room?
#
# 1 and 2 attack the iGPU, which 7.41.4 showed is slower than the CPU even alone, so neither will
# change any training number. They are run because they test our own conclusion about LDS, and
# because the user asked. 3 is on the critical path: attention is 18.3% of a real training step.
#
# Each writes its own log. The chain does not commit -- results go into the report by hand, after
# being read.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\queue.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('queue start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

# stop the older retry loop so it cannot collide
Get-Process powershell -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $PID } |
    ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
Say 'stopped older retry loops'

function RunGated($label, $script, $outfile) {
    for ($i = 1; $i -le 30; $i++) {
        Say ("$label : attempt $i")
        python -u $script *> $outfile
        if ($LASTEXITCODE -eq 0) {
            Say "  $label : ran"
            Add-Content -Path $log -Value ("===== " + $outfile + " =====") -Encoding UTF8
            Get-Content $outfile -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8
            return $true
        }
        if (Select-String -Path $outfile -Pattern 'OUT_OF_HOST_MEMORY' -Quiet -ErrorAction SilentlyContinue) {
            Say "  $label : OpenCL driver failure, not a load problem"
            return $false
        }
        if (Select-String -Path $outfile -Pattern 'Traceback' -Quiet -ErrorAction SilentlyContinue) {
            Say "  $label : script error, skipping"
            Add-Content -Path $log -Value ("===== " + $outfile + " (error) =====") -Encoding UTF8
            Get-Content $outfile -Encoding UTF8 -Tail 25 | Add-Content -Path $log -Encoding UTF8
            return $false
        }
        Say '  guard refused; waiting 5 min'
        Start-Sleep -Seconds 300
    }
    Say "  $label : gave up after 30 attempts"
    return $false
}

RunGated 'register blocking' 'igpu_register_block.py' 'q1_regblock.txt' | Out-Null
RunGated 'swizzle/padding' 'igpu_swizzle.py' 'q2_swizzle.txt' | Out-Null
RunGated 'CPU attention' 'cpu_attention_compare.py' 'q3_attention.txt' | Out-Null

Say 're-running fixed v4 as the reference point'
python -u igpu_gemm_v4.py *> q4_v4.txt
Add-Content -Path $log -Value '===== q4_v4.txt =====' -Encoding UTF8
Get-Content q4_v4.txt -Encoding UTF8 -Tail 12 | Add-Content -Path $log -Encoding UTF8

New-Item -ItemType Directory -Force -Path D:\work\research\evidence\queue_2026-10-15 | Out-Null
Copy-Item q1_regblock.txt, q2_swizzle.txt, q3_attention.txt, q4_v4.txt D:\work\research\evidence\queue_2026-10-15\ -Force -ErrorAction SilentlyContinue
Say 'done'
