# retry_block.ps1 -- retry the register-blocking sweep until the guard lets it through.
#
# The first waiter was wrong: it tested whether wuaucltcore.exe existed, but Windows Update
# processes persist as idle services long after the work is done, so the loop would never exit.
# Checking process existence is the wrong test; checking whether the measurement is allowed is the
# right one, and the script already knows how to answer that itself.
#
# So: just run it, and if the guard refuses, wait and run it again. The guard is the authority on
# whether the machine is quiet, and reimplementing its judgement outside it was the mistake.
#
# The exemption question does not arise: the iGPU is a unified-memory part, so a CPU-saturating
# process competes for the same DRAM. Post-reboot Windows Update is exactly the load that would
# corrupt the numbers, which is why waiting is correct and allow=True would not be.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\regblock.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('register block retry ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

$ok = $false
for ($i = 1; $i -le 40; $i++) {
    Say ("attempt $i")
    python -u igpu_register_block.py *> regblock_out.txt
    $rc = $LASTEXITCODE
    if ($rc -eq 0) {
        Say '  guard allowed it and the sweep completed'
        $ok = $true
        break
    }
    if (Select-String -Path regblock_out.txt -Pattern 'OUT_OF_HOST_MEMORY' -Quiet -ErrorAction SilentlyContinue) {
        Say '  OpenCL driver still failing -- the reboot did not clear it'
        break
    }
    Say '  guard refused; waiting 5 min'
    Start-Sleep -Seconds 300
}

if ($ok) {
    Add-Content -Path $log -Value '===== sweep =====' -Encoding UTF8
    Get-Content regblock_out.txt -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8
    Say 'reference: re-running fixed v4'
    python -u igpu_gemm_v4.py *> regblock_v4.txt
    Get-Content regblock_v4.txt -Encoding UTF8 -Tail 12 | Add-Content -Path $log -Encoding UTF8
} else {
    Say 'gave up or driver broken; last output follows'
    Get-Content regblock_out.txt -Encoding UTF8 -Tail 20 | Add-Content -Path $log -Encoding UTF8
}

Copy-Item regblock_out.txt, regblock_v4.txt D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say 'done'
