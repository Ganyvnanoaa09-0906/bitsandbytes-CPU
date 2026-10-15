# run_gil_test.ps1 -- the process-based concurrency retest, chained behind the FMA analysis.
Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = "utf-8"
$log = "D:\work\bitsandbytes-CPU\giltest.log"
function Say($m) {
    $s = (Get-Date -Format "HH:mm:ss") + "  " + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}
Set-Content -Path $log -Value ("gil test start " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -Encoding UTF8
Say "waiting for the FMA analysis to finish"
$w = 0
while ($w -lt 300) {
    if ((Test-Path "D:\work\bitsandbytes-CPU\fma.log") -and
        (Select-String -Path "D:\work\bitsandbytes-CPU\fma.log" -Pattern "^done" -Quiet)) { break }
    Start-Sleep -Seconds 60
    $w = $w + 1
}
Say "FMA analysis finished after $w min"
Say "running the process-based concurrency test"
python -u igpu_concurrency_procs.py *> giltest_out.txt
Say ("  exit " + $LASTEXITCODE)
if (Test-Path giltest_out.txt) {
    Add-Content -Path $log -Value "----- output -----" -Encoding UTF8
    Get-Content giltest_out.txt -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8
}
Copy-Item giltest_out.txt D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say "done"
