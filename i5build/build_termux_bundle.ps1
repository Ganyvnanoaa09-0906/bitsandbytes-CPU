# build_termux_bundle.ps1 -- assemble what the phone needs, and nothing else.
#
# WHY THIS IS A SCRIPT AND NOT A HAND-TYPED COMMAND SEQUENCE
# The first bundle was assembled by hand and shipped without csrc/common.h, so the
# ARM64 build died on:
#     fatal error: 'common.h' file not found
# while every kernel was fine. The lesson is not "remember common.h", it is that
# hand-picking files from a directory the compiler reads wholesale is a mistake
# waiting to happen. So: copy the whole csrc tree (minus build detritus) and let
# the build decide what it needs.
#
# Usage:  pwsh -File build_termux_bundle.ps1

$ErrorActionPreference = 'Stop'

$src     = 'D:\work\bitsandbytes-CPU'
$pkgroot = "$src\bitsandbytes"          # holds csrc/, selftest_cpu.c, build_termux.sh
$stage   = 'D:\work\termux_bundle'
$serve   = 'D:\work\termux_serve'

Write-Host "=== cleaning staging ==="
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "$stage\bnb"              | Out-Null
New-Item -ItemType Directory -Force -Path "$stage\bnb\csrc"         | Out-Null
New-Item -ItemType Directory -Force -Path "$stage\bnb\bitsandbytes" | Out-Null
New-Item -ItemType Directory -Force -Path $serve                    | Out-Null

Write-Host "=== csrc: whole tree, minus .bak / .exe / .obj / CUDA-only .cu ==="
# The CPU build reads csrc/ as an include root, so anything it might #include has
# to be here. Size is trivial (~0.85 MB even unfiltered).
$copied = 0
Get-ChildItem "$pkgroot\csrc" -File | Where-Object {
    $_.Extension -notin '.bak', '.exe', '.obj', '.cu' -and $_.Name -notlike '*.bak*'
} | ForEach-Object {
    Copy-Item $_.FullName "$stage\bnb\csrc\" -Force
    $copied++
}
Write-Host "   $copied file(s) from csrc"

Write-Host "=== build entry points ==="
foreach ($f in @('selftest_cpu.c', 'build_termux.sh', 'build_linux.sh')) {
    if (-not (Test-Path "$pkgroot\$f")) { throw "missing $pkgroot\$f" }
    Copy-Item "$pkgroot\$f" "$stage\bnb\" -Force
}

Write-Host "=== the bitsandbytes package (build_termux.sh writes the .so here) ==="
# Exclude prebuilt Windows/shared objects: the phone builds its own.
Copy-Item "$pkgroot\bitsandbytes\*" "$stage\bnb\bitsandbytes\" -Recurse -Force `
    -Exclude @('__pycache__', '*.dll', '*.exp', '*.lib', '*.so', '*.pyd')

Write-Host "=== python-side test + helper modules ==="
$pyFiles = @(
    'run_all_tests.py', '_testpath.py', 'efst.py', 'train_1000_steps.py',
    'termux_check.py',
    'termux_train.py',
    'torch_part.py',
    'verify_bnb_intact.py', 'verify_zerocopy.py', 'verify_latent.py',
    'stress_opt.py', 'verify_e2e_train.py', 'verify_real_finetune.py',
    'integration_video_balancer.py', 'test_temporal_only_lora.py',
    'disk_balancer.py', 'latent_chunk_store.py', 'temporal_only_lora.py'
)
foreach ($f in $pyFiles) {
    $p = "$src\$f"
    if (-not (Test-Path $p)) { $p = "$pkgroot\$f" }
    if (Test-Path $p) { Copy-Item $p "$stage\bnb\" -Force }
    else { Write-Warning "not found, skipped: $f" }
}

Write-Host "=== sanity: can the build's includes be satisfied from the bundle? ==="
$incRoot = "$stage\bnb\csrc"
$needed = @('common.h', 'cpu_ops.h', 'cpu_cache.h')
foreach ($h in $needed) {
    if (Test-Path "$incRoot\$h") { Write-Host "   [OK]   csrc/$h" }
    else { throw "bundle is missing csrc/$h -- the ARM64 build would fail on it" }
}
foreach ($c in @('cpu_ops.cpp', 'cpu_gdn.cpp', 'pythonInterface.cpp')) {
    if (Test-Path "$incRoot\$c") { Write-Host "   [OK]   csrc/$c" }
    else { throw "bundle is missing csrc/$c" }
}

Write-Host "=== tarball ==="
$tar = "$serve\bnb_termux.tar.gz"
Remove-Item $tar -Force -ErrorAction SilentlyContinue
Push-Location $stage
try { tar -czf $tar bnb } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw "tar failed (exit $LASTEXITCODE)" }
$tarInfo = Get-Item $tar

$all = Get-ChildItem $stage -Recurse -File
Write-Host ("   {0} files staged, {1:N2} MB" -f $all.Count, (($all | Measure-Object Length -Sum).Sum / 1MB))
Write-Host ("   {0} = {1:N0} bytes" -f $tarInfo.Name, $tarInfo.Length)

Write-Host "=== refresh the served shell scripts ==="
foreach ($f in @('setup_termux_test.sh', 'get_termux_test.sh', 'bnb_launch.sh')) {
    $p = "$src\i5build\$f"
    if (Test-Path $p) { Copy-Item $p $serve -Force; Write-Host "   $f" }
}

Write-Host ""
Write-Host "=== served ==="
Get-ChildItem $serve | Select-Object Name, Length | Format-Table -AutoSize


