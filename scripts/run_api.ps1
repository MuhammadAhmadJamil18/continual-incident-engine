# Start FastAPI (from repo root: .\scripts\run_api.ps1)
Set-Location $PSScriptRoot\..
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Host "Create venv first: python -m venv .venv && .\.venv\Scripts\pip install -e `".[dev]`""
    exit 1
}
& .\.venv\Scripts\python.exe -m uvicorn incident_memory_engine.api.app:app --host 127.0.0.1 --port 8000 --reload
