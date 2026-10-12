# h3_write_report.ps1 -- wait for the analyses, then write the verdict into the report.
#
# Second stage of the unattended chain:
#   h3_finalize.ps1       waits for training, runs the two analyses
#   h3_write_report.ps1   waits for that, writes 7.31.4.9, commits and pushes  ← this file
#
# Exists because the run outlasts the round budget: without it the analyses would finish and
# their numbers would sit in text files while the report stayed silent.
#
# The writing itself is done by h3_write_verdict.py, which parses the analysis output and
# refuses to write if a number is missing. That refusal matters -- a script that fills gaps
# with defaults is how a project ends up reporting results it never measured.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\h3_write.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('h3_write start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8
Say 'waiting for h3_finalize to finish'

$waited = 0
while ($waited -lt 220) {
    if (Test-Path h3_finalize.log) {
        if (Select-String -Path h3_finalize.log -Pattern '^done' -Quiet -ErrorAction SilentlyContinue) {
            break
        }
    }
    Start-Sleep -Seconds 60
    $waited = $waited + 1
    if (($waited % 20) -eq 0) { Say ('  waited ' + $waited + ' min') }
}
Say ('finalize done after ' + $waited + ' min of waiting')

# make sure the analyses actually produced output before touching the report
$ok = (Test-Path h3_verdict.txt) -and ((Get-Item h3_verdict.txt -ErrorAction SilentlyContinue).Length -gt 200)
Say ('h3_verdict.txt usable: ' + $ok)
if (-not $ok) {
    Say 'verdict output missing or too small -- NOT writing to the report'
    Say 'done'
    exit 0
}

Say 'writing 7.31.4.9'
python -u h3_write_verdict.py *>> $log
Say ('  exit ' + $LASTEXITCODE)

Say 'committing'
git add -A 2>&1 | Out-Null
git -c user.name=cpu-forge -c user.email=forge@local commit -m "H3 2000-step verdict written by the unattended chain

The run outlasted the round budget, so the final step is automated too. The conclusion is not
chosen by the script: the rule was fixed in 7.31.4.8 beforehand -- steps-to-target within 5%
at loss <= 5.5 counts as equivalent -- and h3_write_verdict.py applies it mechanically.

It parses h3_verdict.txt and h3_split.txt, refuses to write if any number is missing, labels
the 1.24x arithmetic ceiling as an arithmetic ceiling rather than a result, and appends a
dated subsection so the pre-registered predictions stay visible beside the outcome." *>> $log
Say ('  commit exit ' + $LASTEXITCODE)

Say 'pushing'
for ($i = 1; $i -le 3; $i++) {
    git push *>> $log 2>&1
    if ($LASTEXITCODE -eq 0) { Say '  pushed'; break }
    Start-Sleep -Seconds 20
}

Say 'done'
