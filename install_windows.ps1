param(
    [ValidateSet("cu128","cpu")] [string]$Backend = "cu128",
    [switch]$CoreOnly,
    [switch]$Rewards,
    [switch]$SkipModels
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
function Run-UV {
    & $script:uvExe @args
    if ($LASTEXITCODE -ne 0) { throw "uv failed with exit code $LASTEXITCODE" }
}
$localUV = Join-Path $PSScriptRoot ".venv\Scripts\uv.exe"
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (Test-Path -LiteralPath $localUV) {
    $script:uvExe = $localUV
} elseif ($uvCmd) {
    $script:uvExe = $uvCmd.Source
} else {
    $toolsDir = Join-Path $PSScriptRoot ".tools"
    New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
    $archive = Join-Path $toolsDir "uv.zip"
    Write-Host "Downloading the standalone uv installer..."
    Invoke-WebRequest -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $toolsDir -Force
    $script:uvExe = (Get-ChildItem -LiteralPath $toolsDir -Recurse -File -Filter uv.exe | Select-Object -First 1).FullName
    if (-not $script:uvExe) { throw "uv.exe not found after extraction" }
}
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    Run-UV python install 3.12
    Run-UV venv --python 3.12 .venv
}
Run-UV pip install --python $pythonExe torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 --index-url "https://download.pytorch.org/whl/$Backend"
Run-UV pip install --python $pythonExe -r requirements-core.txt -c constraints.txt
if (-not $CoreOnly) { Run-UV pip install --python $pythonExe -r requirements-extras.txt -c constraints.txt }
if ($Rewards) { Run-UV pip install --python $pythonExe -r requirements-rewards.txt -c constraints.txt }
Run-UV pip install --python $pythonExe --no-deps -e .
if (-not $SkipModels) {
    & $pythonExe jk_step.py models setup --checkpoint-dir checkpoints
    if ($LASTEXITCODE -ne 0) { throw "Model download failed; rerun this installer to resume." }
}
& $pythonExe jk_step.py doctor
if ($LASTEXITCODE -ne 0) { throw "Runtime check failed" }
Write-Host "JK-Step MR-FlowDPO ready. Open start_ui.bat, or use jk-step.bat --help."
