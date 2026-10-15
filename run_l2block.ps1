# run_l2block.ps1 -- run the L2 blocking test after the task-split measurement.
#
# Chained rather than run now: the machine has training jobs queued through the night and every
# measurement in this project that shared the machine with something else had to be redone. This
# waits for tasksplit.log's completion marker.
Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = "utf-8"
$log = "D:\work\bitsandbytes-CPU\l2block.log"
function Say($m) {
    $s = (Get-Date -Format "HH:mm:ss") + "  " + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}
Set-Content -Path $log -Value ("l2block start " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -Encoding UTF8
Say "waiting for the task-split measurement to finish"
$w = 0
while ($w -lt 300) {
    if ((Test-Path "D:\work\bitsandbytes-CPU\tasksplit.log") -and
        (Select-String -Path "D:\work\bitsandbytes-CPU\tasksplit.log" -Pattern "^done" -Quiet)) { break }
    Start-Sleep -Seconds 60
    $w = $w + 1
}
Say "task-split finished after $w min"
Say "running the L2 blocking test"
python -u cpu_l2_blocking.py *> l2block_out.txt
Say ("  exit " + $LASTEXITCODE)
if (Test-Path l2block_out.txt) {
    Add-Content -Path $log -Value "----- output -----" -Encoding UTF8
    Get-Content l2block_out.txt -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8
}
Copy-Item l2block_out.txt D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say "done"
