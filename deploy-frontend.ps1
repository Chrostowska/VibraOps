# VibraOps Frontend Deployment Script for Windows
Write-Host "🔄 Deploying VibraOps Frontend with NGINX..." -ForegroundColor Green

# Stop the nginx container if it's running
Write-Host "📦 Stopping nginx container..." -ForegroundColor Yellow
docker-compose stop nginx

# Remove the nginx container to ensure clean restart
Write-Host "🗑️  Removing nginx container..." -ForegroundColor Yellow
docker-compose rm -f nginx

# Start the nginx container with new configuration
Write-Host "🚀 Starting nginx container with new configuration..." -ForegroundColor Yellow
docker-compose up -d nginx

# Check if nginx is running properly
Write-Host "🔍 Checking nginx status..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
docker-compose ps nginx

# Test the frontend
Write-Host "🌐 Testing frontend accessibility..." -ForegroundColor Green
Write-Host "Frontend should be available at: http://173.212.220.106" -ForegroundColor Cyan
Write-Host "API should be available at: http://173.212.220.106/api/" -ForegroundColor Cyan
Write-Host "Grafana should be available at: http://173.212.220.106/grafana/" -ForegroundColor Cyan
Write-Host "Prometheus should be available at: http://173.212.220.106/prometheus/" -ForegroundColor Cyan

# Show nginx logs
Write-Host "📋 Recent nginx logs:" -ForegroundColor Yellow
docker-compose logs --tail=20 nginx

Write-Host "✅ Deployment complete!" -ForegroundColor Green 