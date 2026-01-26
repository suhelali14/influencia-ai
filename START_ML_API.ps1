# Start the ML Inference API Server
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  Influencia ML API - Starting..." -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

cd ai\inference
python -m uvicorn api_server:app --host 0.0.0.0 --port 5001 --log-level info

Write-Host "`n" -ForegroundColor Yellow
Write-Host "Server stopped." -ForegroundColor Yellow
