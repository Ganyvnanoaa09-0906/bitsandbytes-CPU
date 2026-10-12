# mem_diag_wait.ps1 -- run the memory diagnostic once the trainer stops.
#
# The diagnostic's own measure_guard refuses while anything else is running, which is correct
# and is why this exists: rather than relaxing the guard, wait for the machine to be genuinely
# idle. A bandwidth measurement taken while six cores are busy would be worthless, and the
# guard was built precisely because three sets of numbers were invalidated that way earlier.
#
# Third time writing a wrapper like this, so it is a file rather than an inline -Command: the
# previous two attempts with nested quotes in -Command failed to start at all.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\mem_diag.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('mem diag start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8
Say 'waiting for the trainer to stop'

$waited = 0
while ($waited -lt 300) {
    $busy = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($busy.Count -eq 0) { break }
    Start-Sleep -Seconds 60
    $waited = $waited + 1
    if (($waited % 20) -eq 0) { Say ('  waited ' + $waited + ' min') }
}
Say ('idle after ' + $waited + ' min')

Say 'running the diagnostic'
python -u mem_diagnostic.py *>> $log
Say ('  exit ' + $LASTEXITCODE)

Say 'also comparing the trainer runs'
python -u trainer_compare.py *>> $log
Say ('  compare exit ' + $LASTEXITCODE)

Say 'done'
