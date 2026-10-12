# h3_finalize.ps1 -- wait for the NS=3 2000-step arm, then produce both analyses.
#
# Two consumption patterns are wrong here and this avoids both:
#   * polling in the foreground burns rounds for no information;
#   * running the decomposition measurement now would steal the same six cores as the
#     training it is meant to interpret -- the mistake recorded in 7.26.2, made once already.
#
# So: sleep in the background until the arm lands, then run the two analyses that are already
# written and verified. Both write to files, so the verdict can be read in one call.
#
# ns_h3_2000_verdict.py    three-arm curves, steps-to-target, optimiser cost per step
# ns_h3_fwd_bwd_split.py   the harness's forward/backward split, which is what converts the
#                          optimiser saving into a whole-step figure (its own guard refuses
#                          to run while training is live, as a second line of defence)
#
# Nothing here writes to the report. The verdict goes into 7.31.4 by hand, so a human or a
# later round reads the numbers first rather than having them spliced in unverified.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\h3_finalize.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('h3_finalize start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8
Say 'waiting for the NS=3 2000-step arm'

$waited = 0
while ($waited -lt 300) {
    $busy = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($busy.Count -eq 0) { break }
    Start-Sleep -Seconds 120
    $waited = $waited + 2
    if (($waited % 20) -eq 0) { Say ('  waited ' + $waited + ' min') }
}
Say ('training finished after ' + $waited + ' min of waiting')

Say 'step 1: verdict (three-arm comparison)'
python -u ns_h3_2000_verdict.py *> h3_verdict.txt
Say ('  exit ' + $LASTEXITCODE)

Say 'step 2: forward/backward split'
python -u ns_h3_fwd_bwd_split.py *> h3_split.txt
Say ('  exit ' + $LASTEXITCODE)

Say 'step 3: raw log copy'
Copy-Item ns_ab_muon3_2000.log D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say '  copied'

Say 'appending results to this log'
Add-Content -Path $log -Value '----- h3_verdict.txt -----' -Encoding UTF8
Get-Content h3_verdict.txt -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8
Add-Content -Path $log -Value '----- h3_split.txt -----' -Encoding UTF8
Get-Content h3_split.txt -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8

Say 'done'
