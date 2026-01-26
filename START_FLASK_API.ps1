# Start Flask AI API Server on Port 5002
# This is the comprehensive analysis API (separate from FastAPI inference server)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  FLASK AI API SERVER (Analysis & Reports)" -ForegroundColor Cyan
Write-Host "  Starting on Port 5002" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$aiPath = "C:\Users\Suhelali\OneDrive\Desktop\Influencia\ai"

Set-Location $aiPath

Write-Host "Starting Flask API server..." -ForegroundColor Yellow
Write-Host "  Port: 5002" -ForegroundColor Gray
Write-Host "  Endpoints:" -ForegroundColor Gray
Write-Host "    /health" -ForegroundColor Gray
Write-Host "    /api/analyze" -ForegroundColor Gray
Write-Host "    /api/generate-report" -ForegroundColor Gray
Write-Host "    /api/generate-creator-report" -ForegroundColor Gray
Write-Host "    /api/match-score" -ForegroundColor Gray
Write-Host ""

python api_server.py 5002
