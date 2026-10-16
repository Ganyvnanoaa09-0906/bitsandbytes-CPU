# run_fma.ps1 -- the FMA analysis, chained behind the L2 blocking test.
Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = "utf-8"
$log = "D:\work\bitsandbytes-CPU\fma.log"
function Say($m) {
    $s = (Get-Date -Format "HH:mm:ss") + "  " + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}
Set-Content -Path $log -Value ("fma analysis start " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -Encoding UTF8
Say "waiting for the L2 blocking test to finish"
$w = 0
while ($w -lt 300) {
    if ((Test-Path "D:\work\bitsandbytes-CPU\l2block.log") -and
        (Select-String -Path "D:\work\bitsandbytes-CPU\l2block.log" -Pattern "done" -Quiet)) { break }
    Start-Sleep -Seconds 60
    $w = $w + 1
}
Say "L2 test finished after $w min"
Say "running the FMA analysis"
python -u cpu_fma_analysis.py *> fma_out.txt
Say ("  exit " + $LASTEXITCODE)
if (Test-Path fma_out.txt) {
    Add-Content -Path $log -Value "----- output -----" -Encoding UTF8
    Get-Content fma_out.txt -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8
}
Copy-Item fma_out.txt D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say "done"
