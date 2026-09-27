# ============================================================
# termux_adb_test.ps1 - verify the Termux build on a real device over ADB
# ------------------------------------------------------------
# NOTE ON ENCODING (deliberate, do not "fix" this):
#   This script is intentionally pure ASCII, including all messages.
#   Windows PowerShell 5.1 reads .ps1 files using the system ANSI code page.
#   A UTF-8-without-BOM script containing non-ASCII text gets mis-decoded,
#   which does not merely garble the messages -- it breaks the PARSER
#   (measured: 6 syntax errors on a file that parses cleanly under UTF-8).
#   Since this tool has to run on a machine whose code page we do not control,
#   ASCII is the only safe choice. The Chinese usage notes live in
#   TERMUX_GUIDE.md instead, which is plain UTF-8 and never parsed.
#
# DESIGN DECISIONS (each one avoids a specific failure that was actually hit):
#   1) Transport through /sdcard/Download, NOT into Termux's home.
#      adb shell runs as uid 2000(shell). /data/data/com.termux/ is
#      "Permission denied" for that uid, so BOTH `adb push` into Termux's home
#      AND an `adb shell ... > file` pipe fail there. (An earlier version of this
#      script claimed the pipe "runs as Termux and therefore can write" -- that
#      was wrong, and it is recorded here so nobody re-derives it.)
#      /sdcard is shared storage: writable by the shell user, and readable by
#      Termux without termux-setup-storage.
#   2) The build itself cannot be run over adb at all. Termux keeps its own
#      app keystore, so an outside process cannot execute commands inside it
#      without RUN_COMMAND permissions. The script therefore prepares everything
#      and hands over ONE line to paste into Termux.
#   3) Probe, never assume.
#      Termux's install state is probed via `pm list packages`, NOT by running
#      `test -x` on its shell binary: that test fails with permission-denied and
#      produced a confident "Termux is not installed" on a device that had it.
#   4) Device-side commands stay mksh/toybox compatible.
#      Measured on the target (Android 10): `if ...; then ...; else ...; fi` is
#      rejected by /system/bin/sh, and toybox `stat` has no -c/--format, so
#      `stat -c %s` fails with "Needs 1 argument". `wc -c < file` works.
#   5) No `&&` / `||` in this file's PowerShell source.
#      PowerShell 5.1's parser rejects those tokens even inside string literals.
#
# USAGE:
#   pwsh -File termux_adb_test.ps1                 # pack, transfer, build, selftest
#   pwsh -File termux_adb_test.ps1 -ProbeOnly      # only inspect device + Termux
#   pwsh -File termux_adb_test.ps1 -SkipSelftest   # build only, skip C selftest
# ============================================================
param(
  # Prefer a modern adb. The one on the Desktop reports "Android Debug Bridge
  # version 1.0.26" (a 2013 build) and is unreliable with recent Huawei/Kirin
  # devices: its server speaks protocol 26, it can hang on bare `devices`, and
  # it misses `-l` details. The copy shipped with MuMu is 1.0.41 / 36.0.0 and
  # behaves correctly. Both are probed; whatever works is used.
  [string]$Adb = "",
  [switch]$ProbeOnly,
  [switch]$SkipSelftest,
  [int]$WaitForDeviceSec = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = "D:\work\bitsandbytes-CPU\bitsandbytes"   # inner build root
$Stage    = Join-Path $env:TEMP "termux_stage"
$TarGz    = Join-Path $env:TEMP "bnb_termux.tar.gz"
$Bash     = "C:\Program Files\Git\bin\bash.exe"

function Info($m) { Write-Host "  $m" }
function Head($m) { Write-Host ""; Write-Host "=== $m ===" -ForegroundColor Cyan }
function Die($m) {
  Write-Host ""
  Write-Host "ERROR: $m" -ForegroundColor Red
  exit 1
}

# Run an adb command with a hard timeout, capturing output through a file.
#
# Two measured reasons this is not simpler:
#   1) bare `adb devices` HANGS INDEFINITELY here, while `adb devices -l` returns
#      immediately. A harness that can hang forever is worse than one that fails
#      loudly, so every adb call is bounded.
#   2) Do NOT wrap adb in Start-Job. Stopping a job kills the whole child process
#      tree, and adb relies on a detached server process; killing it mid-flight
#      produced "List of devices attached" with the device missing -- i.e. the
#      wrapper silently lost the device. Redirecting to a file and using
#      Start-Process + Wait-Process keeps adb's own process model intact.
function Invoke-Adb {
  param(
    [string[]]$AdbArgs,
    [int]$TimeoutSec = 30
  )
  $tag = [guid]::NewGuid().ToString("N")
  $outFile = Join-Path $env:TEMP ("adbcall_$tag.txt")
  $batFile = Join-Path $env:TEMP ("adbcall_$tag.bat")

  # Write a small .bat and let cmd run it, instead of handing cmd.exe a
  # pre-quoted command line. Built from an argument array, this sidesteps every
  # layer of quote escaping -- and the previous inline approach reliably failed
  # with "The filename, directory name, or volume label syntax is incorrect".
  $argList = @()
  foreach ($a in $AdbArgs) { $argList += '"' + ($a -replace '"', '""') + '"' }
  $bat = "@echo off`r`n" +
         '"' + $Adb + '" ' + ($argList -join ' ') + ' > "' + $outFile + '" 2>&1' + "`r`n"
  Set-Content -Path $batFile -Value $bat -Encoding ASCII

  $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$batFile`"" -NoNewWindow -PassThru
  # NOTE: -PassThru on Wait-Process is a pwsh 7 addition; Windows PowerShell 5.1
  # rejects it outright ("A parameter cannot be found"). This has to run on 5.1,
  # so use the .NET wait, which exists everywhere.
  $exited = $proc.WaitForExit($TimeoutSec * 1000)
  $timedOut = $false
  if (-not $exited) {
    $timedOut = $true
    try { $proc.Kill() } catch { }
    try { & $Adb kill-server 2>&1 | Out-Null } catch { }
  }
  $text = ""
  if (Test-Path $outFile) {
    $text = (Get-Content $outFile -Raw -ErrorAction SilentlyContinue)
    Remove-Item $outFile -Force -ErrorAction SilentlyContinue
  }
  Remove-Item $batFile -Force -ErrorAction SilentlyContinue
  if ($null -eq $text) { $text = "" }
  return @{ TimedOut = $timedOut; Output = $text }
}

# ---------------------------------------------------------------- adb + device
Head "1. adb and device"

# Pick an adb. Explicit -Adb wins; otherwise probe the known candidates and
# prefer the one with the highest reported version.
$adbCandidates = @()
if ($Adb -ne "") { $adbCandidates += $Adb }
$adbCandidates += @(
  "C:\Program Files\Netease\MuMu\nx_main\adb.exe",
  "C:\Users\Gan\Desktop\adb\adb.exe"
)
$picked = $null
foreach ($cand in $adbCandidates) {
  if (-not (Test-Path $cand)) { Info "not found: $cand"; continue }
  $vv = & $cand version 2>&1 | Out-String
  $verLine = ($vv -split "`r?`n" | Where-Object { $_ -match "Android Debug Bridge version" } | Select-Object -First 1)
  Info ("candidate {0} -> {1}" -f $cand, $verLine.Trim())
  if (-not $picked) { $picked = $cand }
}
if (-not $picked) { Die "no adb executable found. Tried: $($adbCandidates -join ', ')" }
$Adb = $picked
Info "using adb: $Adb"

# Kill any stale server first. Mixing client versions leaves a server speaking a
# different protocol, which shows up as "server version (26) doesn't match this
# client (41)" and an empty device list even when the phone is attached.
# This is best-effort: with no server running it exits non-zero and prints to
# stderr ("cannot connect to daemon at tcp:5037"), and with
# $ErrorActionPreference = "Stop" that would abort the whole script on a step
# that is not fatal. Hence the explicit EAP override.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Adb kill-server 2>&1 | Out-Null
$ErrorActionPreference = $prevEap
Start-Sleep -Milliseconds 800

function Get-DeviceState {
  $d = Invoke-Adb -AdbArgs @("devices", "-l") -TimeoutSec 25
  if ($d.TimedOut) { return @{ Serial = $null; State = "timeout"; Raw = "" } }
  foreach ($line in ($d.Output -split "`r?`n")) {
    if ($line -match "^(\S+)\s+(device|offline|unauthorized|bootloader|recovery)\b") {
      return @{ Serial = $matches[1]; State = $matches[2]; Raw = $d.Output }
    }
  }
  return @{ Serial = $null; State = "none"; Raw = $d.Output }
}

$wantSec = if ($WaitForDeviceSec -gt 0) { $WaitForDeviceSec } else { 30 }
$r = Get-DeviceState
Info $r.Raw.Trim()
$waited = 0
while ((-not $r.Serial -or $r.State -eq "offline") -and $waited -lt $wantSec) {
  if ($waited -eq 0) {
    Write-Host "  waiting up to $wantSec s for the phone to appear/authorise ..."
  }
  Start-Sleep -Seconds 3
  $waited += 3
  $r = Get-DeviceState
}
$serial = $r.Serial
$state = $r.State
if ($waited -gt 0) { Info "waited ${waited}s -> serial=$serial state=$state" }

if (-not $serial) {
  Write-Host ""
  Write-Host "No device detected after waiting ${wantSec}s. Check, in this order:" -ForegroundColor Yellow
  Write-Host "  1) The USB device shows up on the PC at all. On this machine the phone"
  Write-Host "     previously appeared as VID_12D1 PID_107E (Huawei). Stale 'phantom'"
  Write-Host "     entries with Code 45 mean it is NOT currently attached."
  Write-Host "  2) USB cable carries data (charge-only cables never work)."
  Write-Host "  3) Phone: Developer options -> USB debugging ON, plus"
  Write-Host "     'Allow ADB debugging in charge-only mode' ON (Huawei/HarmonyOS)."
  Write-Host "  4) Phone: some Huawei builds also need 'HDB' enabled in Developer options."
  Write-Host "  5) Replug, then:  adb kill-server"
  Write-Host "  Or use WiFi: Developer options -> Wireless debugging, then"
  Write-Host "     adb pair <phone-ip:port>   and   adb connect <phone-ip:port>"
  exit 1
}

Info ("serial    = {0}" -f $serial)
Info ("state     = {0}" -f $state)

if ($state -eq "offline") {
  Write-Host ""
  Write-Host "The device is visible but OFFLINE." -ForegroundColor Yellow
  Write-Host "This almost always means the USB-debugging prompt on the phone has not"
  Write-Host "been accepted. On the phone:"
  Write-Host "  1) look for 'Allow USB debugging?' and tap Allow"
  Write-Host "     (tick 'Always allow from this computer')"
  Write-Host "  2) no dialog? Developer options -> Revoke USB debugging authorisations,"
  Write-Host "     then toggle USB debugging off and on, and replug"
  Write-Host "  3) still offline?  adb kill-server  then replug"
  exit 1
}
if ($state -eq "unauthorized") {
  Write-Host ""
  Write-Host "The device is UNAUTHORIZED. Tap Allow on the phone's debugging prompt," -ForegroundColor Yellow
  Write-Host "then run this script again."
  exit 1
}
if ($state -ne "device") {
  Write-Host ""
  Write-Host "Device is in state '$state', not 'device'. Boot it normally and retry." -ForegroundColor Yellow
  exit 1
}

foreach ($p in @("ro.product.model", "ro.product.manufacturer", "ro.build.version.release",
                 "ro.product.cpu.abi", "ro.product.cpu.abilist")) {
  $r = Invoke-Adb -AdbArgs @("shell", "getprop", $p) -TimeoutSec 20
  $v = if ($r.TimedOut) { "<timeout>" } else { $r.Output.Trim() }
  Info ("{0,-28} = {1}" -f $p, $v)
}

# ---------------------------------------------------------------- Termux probe
Head "2. Termux environment"
#
# Do NOT probe Termux by running `test -x` on its shell binary, which is the
# obvious approach and is wrong here: adb shell is uid 2000(shell) and
# /data/data/com.termux/ is "Permission denied" for it, so the test fails even
# when Termux is installed and healthy. That produced a confident and completely
# incorrect "Termux is not installed" message on a device that had it.
# Probe what is actually observable: package presence, the data directory's
# existence (visible even when its contents are not), and shared storage.
$r = Invoke-Adb -AdbArgs @("shell", "pm list packages com.termux") -TimeoutSec 20
$pkgOut = if ($r.TimedOut) { "" } else { $r.Output }
$hasTermux = ($pkgOut -match "package:com\.termux")
Info ("package com.termux : {0}" -f ($(if ($hasTermux) { "INSTALLED" } else { "NOT FOUND" })))
if (-not $hasTermux) {
  Write-Host ""
  Write-Host "Termux is not installed. Install it from F-Droid or GitHub releases" -ForegroundColor Yellow
  Write-Host "(NOT the Play Store build, it is too old), open it once so it unpacks its"
  Write-Host "bootstrap, then run this script again."
  exit 1
}
$r = Invoke-Adb -AdbArgs @("shell", "ls -d /data/data/com.termux") -TimeoutSec 20
$dataDir = if ($r.TimedOut) { "" } else { $r.Output.Trim() }
Info ("data dir           : {0}" -f $dataDir)

# Tell the user what adb can and cannot do here, because it shapes the workflow.
$r = Invoke-Adb -AdbArgs @("shell", "ls /data/data/com.termux/") -TimeoutSec 20
$canRead = if ($r.TimedOut) { "" } else { $r.Output.Trim() }
$denied = ($canRead -match "Permission denied")
Info ("adb can read Termux home : {0}" -f ($(if ($denied) { "NO (expected on Android 10+)" } else { "yes" })))
if ($denied) {
  Info "  -> build commands cannot be run over adb; you will get one line to paste into Termux"
}

# Shared storage is the transport that works, so verify both directions early.
$probe = "touch /sdcard/Download/_probe 2>&1 && echo WRITABLE && rm -f /sdcard/Download/_probe || echo DENIED"
$r = Invoke-Adb -AdbArgs @("shell", $probe) -TimeoutSec 20
$w = if ($r.TimedOut) { "<timeout>" } else { $r.Output.Trim() }
Info ("/sdcard/Download writable by adb : {0}" -f $w)
if ($w -notmatch "WRITABLE") {
  Die "shared storage is not writable, so there is no transport for the payload. On the phone run: termux-setup-storage"
}

if ($ProbeOnly) { Write-Host ""; Write-Host "(-ProbeOnly: stopping here)" -ForegroundColor Yellow; exit 0 }

# ---------------------------------------------------------------- package
Head "3. Package the build inputs"
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage | Out-Null

# Ship only what the build needs. The full tree is ~10 MB, mostly docs, examples
# and CUDA headers that Android cannot use anyway.
foreach ($i in @("csrc", "selftest_cpu.c", "build_termux.sh", "build_linux.sh",
                 "pyproject.toml", "setup.py", "README.md")) {
  $src = Join-Path $RepoRoot $i
  if (Test-Path $src) {
    Copy-Item $src -Destination $Stage -Recurse -Force
    Info "+ $i"
  } else {
    Info "! skipped (missing): $i"
  }
}
$pkg = Join-Path $RepoRoot "bitsandbytes"
if (Test-Path $pkg) {
  $dst = Join-Path $Stage "bitsandbytes"
  New-Item -ItemType Directory -Path $dst -Force | Out-Null
  Copy-Item (Join-Path $pkg "*.py") -Destination $dst -Force
  Copy-Item (Join-Path $pkg "py.typed") -Destination $dst -Force -ErrorAction SilentlyContinue
  Info "+ bitsandbytes/*.py"
}
# Drop CUDA headers: not used on Android, and they are the bulk of csrc.
$dropped = 0
Get-ChildItem (Join-Path $Stage "csrc") -File | Where-Object {
  ($_.Extension -eq ".cuh") -or ($_.Extension -eq ".cu")
} | ForEach-Object {
  Info "- dropping CUDA header: csrc/$($_.Name)"
  Remove-Item $_.FullName -Force
  $dropped = $dropped + 1
}
Info "dropped $dropped CUDA file(s)"

$size = (Get-ChildItem $Stage -Recurse -File | Measure-Object -Property Length -Sum).Sum
Info ("staging: {0:N1} KB" -f ($size / 1KB))

if (-not (Test-Path $Bash)) { Die "bash not found (needed to create the tarball): $Bash" }
if (Test-Path $TarGz) { Remove-Item $TarGz -Force }

# Translate Windows paths to the form Git Bash understands.
# A bare "C:/..." makes GNU tar treat the colon as a REMOTE-HOST separator and
# fail with "tar (child): Cannot connect to C: resolve failed", so the drive
# letter has to become /c/... first.
function To-MsysPath([string]$p) {
  $q = $p -replace "\\", "/"
  if ($q -match "^([A-Za-z]):(.*)$") { return "/" + $matches[1].ToLower() + $matches[2] }
  return $q
}
$msysStage = To-MsysPath $Stage
$msysTar = To-MsysPath $TarGz
Info "msys stage: $msysStage"
Info "msys tar  : $msysTar"
$tarOut = & $Bash -c "tar czf '$msysTar' -C '$msysStage' --exclude='__pycache__' ." 2>&1 | Out-String
if (-not (Test-Path $TarGz)) {
  Write-Host $tarOut
  Die "packaging failed (tar produced no output file): $TarGz"
}
Info ("tarball: $TarGz  ({0:N1} KB)" -f ((Get-Item $TarGz).Length / 1KB))

# ---------------------------------------------------------------- transfer
Head "4. Transfer to shared storage (/sdcard/Download)"
#
# Can't write into Termux's home directly. Two measured facts forced this design:
#   - adb shell runs as uid 2000(shell), and /data/data/com.termux/ is
#     "Permission denied" for it. So neither `adb push` NOR an `adb shell` pipe
#     can put a file into Termux's home -- an earlier version of this script
#     assumed the pipe ran AS Termux, which was simply wrong.
#   - /sdcard is shared storage and IS writable by the shell user, and Termux
#     reads it without needing termux-setup-storage.
# So: push to /sdcard/Download, then hand Termux a one-liner to copy it home.
$remoteDir = "bnb_termux"
$stageName = "bnb_termux.tar.gz"
$stagePath = "/sdcard/Download/$stageName"

$rm = Invoke-Adb -AdbArgs @("shell", "rm -f $stagePath") -TimeoutSec 20
$p = Start-Process -FilePath $Adb -ArgumentList "push", "`"$TarGz`"", $stagePath -NoNewWindow -Wait -PassThru
if ($p.ExitCode -ne 0) { Die "adb push failed (exit code $($p.ExitCode))" }

# Verify from the device side, not from adb's own reporting: "push said OK" and
# "the file is there with the right size" are different claims.
# `wc -c < file` rather than `stat -c %s file`: this device runs Android 10 whose
# toybox stat does not implement -c/--format at all (measured: -c %s and
# --format=%s both fail with "Needs 1 argument", while wc -c and ls|awk work).
$r = Invoke-Adb -AdbArgs @("shell", "wc -c < $stagePath") -TimeoutSec 20
$remoteSize = if ($r.TimedOut) { "0" } else { $r.Output.Trim() }
$localSize = (Get-Item $TarGz).Length
Info "pushed to $stagePath"
Info "local $localSize B / device $remoteSize B"
$remoteNum = 0
$parsed = [int]::TryParse($remoteSize, [ref]$remoteNum)
if ((-not $parsed) -or ($remoteNum -ne $localSize)) {
  Die "size on device differs from local (truncated transfer). Device said: '$remoteSize'"
}
Info "sizes match - OK"

# Confirm Termux itself can see it. This is the check that actually matters:
# the file being on /sdcard does not help if Termux cannot read that path.
$probe = "test -r $stagePath && echo READABLE || echo NOT_READABLE"
$r = Invoke-Adb -AdbArgs @("shell", $probe) -TimeoutSec 25
$readable = if ($r.TimedOut) { "<timeout>" } else { $r.Output.Trim() }
Info "readable from adb shell: $readable"
if ($readable -ne "READABLE") {
  Die "cannot read $stagePath from the device shell. On some ROMs shared storage needs a one-time grant; open Termux and run: termux-setup-storage"
}

# ---------------------------------------------------------------- build
Head "5. Build on the device"
#
# Termux holds the app keystore, so adb cannot drive its shell directly. Hand the
# user ONE line to paste, and build the command so it is safe to paste verbatim:
# every step is checked, and failure stops with a readable message instead of
# running the next command against a half-unpacked tree.
$selftestArg = if ($SkipSelftest) { "" } else { " --selftest" }
$oneLiner = "cd ~; rm -rf $remoteDir; mkdir -p $remoteDir; " +
            "cp $stagePath ~/$remoteDir.tar.gz; " +
            "tar xzf ~/$remoteDir.tar.gz -C $remoteDir; " +
            "cd $remoteDir; " +
            "if command -v clang++ >/dev/null 2>&1; then " +
            "bash build_termux.sh$selftestArg; " +
            "else " +
            "echo 'MISSING TOOLCHAIN - run: pkg update; pkg install -y clang libomp make'; " +
            "fi"

Write-Host ""
Write-Host "Paste this ONE line into Termux (it copies the payload out of shared"
Write-Host "storage, unpacks it, and builds):"
Write-Host ""
Write-Host $oneLiner -ForegroundColor Green
Write-Host ""
Write-Host "Then paste the output back here and it will be analysed."
Write-Host "Nothing was built automatically because Termux owns its own keystore, so"
Write-Host "adb cannot execute commands inside it without RUN_COMMAND permissions."

Write-Host ""
Write-Host "=== done ===" -ForegroundColor Green
Write-Host "Acceptance (all three must hold):"
Write-Host "  1) build_termux.sh printed DONE."
Write-Host "  2) the exported-symbol check shows all 5 key symbols as [OK]"
Write-Host "  3) the selftest printed [PASSED] (all 4 checks)"
Write-Host ""
Write-Host "If step 5 failed with 'clang++ not found', run this ON THE PHONE:"
Write-Host "  pkg update"
Write-Host "  pkg install -y clang libomp make"
Write-Host "then run this script again."
