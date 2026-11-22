param(
    [string]$PythonExe = (Join-Path $HOME "Desktop\LittleBit-CUDA-venv\Scripts\python.exe")
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $PythonExe)) {
    throw "Existing LittleBit CUDA venv not found at $PythonExe. Supply -PythonExe with a working Python that has CuPy."
}
Set-Location $PSScriptRoot
& $PythonExe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed" }
& $PythonExe littlebitchat.py status
if ($LASTEXITCODE -ne 0) { throw "LittleBitChat status command failed" }
Write-Host "LittleBitChat installed locally. Trained Gemma3-27B LittleBit-2 weights are not included."
