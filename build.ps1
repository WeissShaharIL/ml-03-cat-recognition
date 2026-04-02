# Build frontends and start all services
Set-Location $PSScriptRoot

Write-Host "Building frontend-classifier..." -ForegroundColor Cyan
Set-Location frontend-classifier
npm install
npm run build
Set-Location ..

Write-Host "Building frontend-dash..." -ForegroundColor Cyan
Set-Location frontend-dash
npm install
npm run build
Set-Location ..

Write-Host "Starting Docker Compose..." -ForegroundColor Cyan
docker compose up --build