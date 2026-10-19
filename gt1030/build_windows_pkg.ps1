# Build the Windows iGPU-training zip package.
#   powershell -ExecutionPolicy Bypass -File gt1030\build_windows_pkg.ps1
#   optional: -ClblastDll <path> -OutDir <dir> -Version 0.1.0
param(
    [string]$ClblastDll = "D:\work\clblast_build\CLBlast-master\build\clblast.dll",
    [string]$OutDir = "D:\work\bitsandbytes-CPU\gt1030\dist",
    [string]$Version = "0.1.0"
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$pkg = Join-Path $repo 'gt1030\windows_pkg'
$dist = $OutDir
$name = "bitsandbytes-cpu-igpu-train-win-v$Version"
$stage = Join-Path $dist $name
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

Copy-Item (Join-Path $repo 'gt1030\igpu_train.py') $stage
foreach ($f in 'check_env.py', 'setup.bat', 'doctor.bat', 'train.bat',
               'README_WINDOWS.md', 'THIRD_PARTY_NOTICES.md') {
    Copy-Item (Join-Path $pkg $f) $stage
}
if (-not (Test-Path $ClblastDll)) {
    throw "CLBlast dll not found: $ClblastDll (pass -ClblastDll <path>)"
}
Copy-Item $ClblastDll $stage
$license = Join-Path (Split-Path -Parent $ClblastDll) '..\LICENSE'
if (Test-Path $license) {
    Copy-Item $license (Join-Path $stage 'CLBlast-LICENSE.txt')
} else {
    Write-Host "WARN: CLBlast LICENSE not found next to $ClblastDll" -ForegroundColor Yellow
}

$zip = Join-Path $dist "$name.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
Write-Host ""
Write-Host "package: $zip"
Write-Host ("size   : {0:N2} MB" -f ((Get-Item $zip).Length / 1MB))
Write-Host "sha256 : $hash"
Write-Host "files  :"
Get-ChildItem $stage -File | Sort-Object Name | ForEach-Object {
    Write-Host ("  {0,-30} {1,10:N1} KB" -f $_.Name, ($_.Length / 1KB))
}
