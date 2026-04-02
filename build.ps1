# Build all services and start
Set-Location $PSScriptRoot

Write-Host "Building and starting all services..." -ForegroundColor Cyan
docker compose up --build