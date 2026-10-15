# wait_and_block.ps1 -- run the register-blocking sweep once Windows Update settles.
#
# The reboot cleared the OpenCL driver failure, but wuaucltcore.exe is now at 116% doing
# post-reboot update work, and the guard refused the measurement. That refusal is correct here
# rather than over-cautious: the iGPU is a unified-memory part, so a CPU-saturating process is
# competing for the same DRAM the GPU reads from. An allow=True exemption is for cases where the
# load provably cannot touch the measured object, and this is not one.
#
# So wait, then run. The sweep itself is short -- five thread/tile combinations, three timed
# repetitions each, a few minutes.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\regblock.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('register block run ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

Say 'waiting for Windows Update to settle'
$waited = 0
while ($waited -lt 90) {
    $wu = @(Get-Process wuaucltcore, TiWorker, TrustedInstaller -ErrorAction SilentlyContinue)
    if ($wu.Count -eq 0) { break }
    Start-Sleep -Seconds 30
    $waited = $waited + 0.5
    if (($waited % 10) -eq 0) { Say ('  waited ' + $waited + ' min') }
}
Say ('update processes quiet after ' + $waited + ' min')

Say 'running the register-blocking sweep'
python -u igpu_register_block.py *> regblock_out.txt
Say ('  exit ' + $LASTEXITCODE)

if (Test-Path regblock_out.txt) {
    Add-Content -Path $log -Value '----- sweep output -----' -Encoding UTF8
    Get-Content regblock_out.txt -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8
}

Say 'also re-running the fixed v4 kernel as a reference point'
python -u igpu_gemm_v4.py *> regblock_v4.txt
Say ('  v4 exit ' + $LASTEXITCODE)
Get-Content regblock_v4.txt -Encoding UTF8 -Tail 12 | Add-Content -Path $log -Encoding UTF8

Copy-Item regblock_out.txt, regblock_v4.txt D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say 'done'
