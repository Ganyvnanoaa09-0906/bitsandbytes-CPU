# publish.ps1 -- upload the built artifacts to PyPI from a token you type, not one that
# gets pasted into a chat or a shell history.
#
# WHY A PROMPT AND NOT A PARAMETER
# A token passed as -Token lands in the PowerShell history file and in the process
# command line, where any other process on the machine can read it. Read-Host -AsSecureString
# keeps it out of both, and it is exported to the twine child process only.
#
# USAGE
#     powershell -File tools\publish.ps1                 # upload dist\*, prompting
#     powershell -File tools\publish.ps1 -DryRun         # show what would be uploaded
#
# The token prompt expects the string from
#     PyPI -> Account settings -> API tokens -> Add API token  (scope: this project)
# It starts with "pypi-" and is about 150 characters.
#
# ASCII ONLY: this file must stay pure ASCII. It is read by PowerShell on Windows where
# the console codepage is 936, and a stray non-ASCII byte here is a parse error.

[CmdletBinding()]
param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$pkg = Split-Path -Parent $PSScriptRoot          # ...\bitsandbytes  (the project root)
$dist = Join-Path $pkg 'dist'

if (-not (Test-Path $dist)) {
    Write-Host "no dist\ directory; build first:" -ForegroundColor Yellow
    Write-Host "    python setup.py sdist bdist_wheel"
    exit 1
}

$files = Get-ChildItem $dist -File | Where-Object { $_.Name -match '\.(tar\.gz|whl)$' }
if (-not $files) {
    Write-Host "dist\ is empty; build first:  python setup.py sdist bdist_wheel" -ForegroundColor Yellow
    exit 1
}

Write-Host "artifacts:"
foreach ($f in $files) {
    Write-Host ("  {0,-58} {1,8:N1} KB" -f $f.Name, ($f.Length / 1KB))
}

# twine check first: a malformed long_description is rejected by the server after the
# upload has already started, which is a worse failure than a local one.
Write-Host "`ntwine check ..."
python -m twine check $files.FullName
if ($LASTEXITCODE -ne 0) {
    Write-Host "twine check failed; not uploading" -ForegroundColor Red
    exit 1
}

if ($DryRun) {
    Write-Host "`n-DryRun: nothing was uploaded." -ForegroundColor Cyan
    exit 0
}

Write-Host "`nPaste the API token (input is hidden, it is not written to history):"
$secure = Read-Host -AsSecureString -Prompt "  token"
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

if (-not $token) {
    Write-Host "empty token; aborting" -ForegroundColor Red
    exit 1
}
if ($token -notlike 'pypi-*') {
    Write-Host "that does not look like a PyPI API token (they start with 'pypi-')." -ForegroundColor Yellow
    Write-Host "A FIDO:/... string is a passkey handshake, not a token; twine cannot use it."
    exit 1
}

# Exported for the child process only; removed from this shell's environment afterwards.
$env:TWINE_USERNAME = '__token__'
$env:TWINE_PASSWORD = $token
try {
    python -m twine upload --non-interactive $files.FullName
    $rc = $LASTEXITCODE
} finally {
    Remove-Item Env:TWINE_PASSWORD -ErrorAction SilentlyContinue
    Remove-Item Env:TWINE_USERNAME -ErrorAction SilentlyContinue
}

if ($rc -ne 0) {
    Write-Host "`nupload failed (exit $rc)" -ForegroundColor Red
    Write-Host "if the version already exists, PyPI refuses it: bump the version in"
    Write-Host "setup.py and bitsandbytes/__init__.py, rebuild, and try again."
    exit $rc
}

Write-Host "`nuploaded. check https://pypi.org/project/bitsandbytes-cpu-fork/" -ForegroundColor Green
Write-Host "verify a clean install in a machine with no torch:"
Write-Host "    bash tools/verify_linux_cli.sh"
