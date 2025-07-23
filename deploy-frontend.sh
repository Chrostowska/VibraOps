#!/bin/bash

echo "🔄 Deploying VibraOps Frontend with NGINX..."

# Stop the nginx container if it's running
echo "📦 Stopping nginx container..."
docker-compose stop nginx

# Remove the nginx container to ensure clean restart
echo "🗑️  Removing nginx container..."
docker-compose rm -f nginx

# Start the nginx container with new configuration
echo "🚀 Starting nginx container with new configuration..."
docker-compose up -d nginx

# Check if nginx is running properly
echo "🔍 Checking nginx status..."
sleep 5
docker-compose ps nginx

# Test the frontend
echo "🌐 Testing frontend accessibility..."
echo "Frontend should be available at: http://173.212.220.106"
echo "API should be available at: http://173.212.220.106/api/"
echo "Grafana should be available at: http://173.212.220.106/grafana/"
echo "Prometheus should be available at: http://173.212.220.106/prometheus/"

# Show nginx logs
echo "📋 Recent nginx logs:"
docker-compose logs --tail=20 nginx

echo "✅ Deployment complete!" 