# Create a virtual environment and install TBC dependencies.
# Usage: .\setup.ps1
$ErrorActionPreference = "Stop"

$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "Creating virtual environment in .venv\ ..."
& $python -m venv .venv

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done. To use the environment:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  streamlit run app.py"
