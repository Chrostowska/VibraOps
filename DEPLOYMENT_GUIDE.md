# VibraOps Frontend Deployment Guide

## 🚨 Issue Resolution Summary

### Problem
The VibraOps frontend was not accessible via `http://173.212.220.106` because:
- The nginx container was running without a configuration file
- The expected file `./monitoring/nginx/default.conf` did not exist
- NGINX started without routing or serving anything, resulting in `ERR_CONNECTION_REFUSED`

### Solution Implemented
1. **Created NGINX Configuration**: Added `./monitoring/nginx/default.conf` with proper routing
2. **Updated Docker Compose**: Added frontend volume mounting to nginx container
3. **Created Deployment Scripts**: Added scripts for easy deployment

## 📁 Files Created/Modified

### 1. NGINX Configuration (`./monitoring/nginx/default.conf`)
- Static file serving for frontend (`/`)
- Reverse proxy for API (`/api/` → `vibraops-api:8000`)
- Health check endpoint (`/health`)
- Optional proxies for Grafana (`/grafana/`) and Prometheus (`/prometheus/`)
- Security headers and error handling

### 2. Docker Compose Updates (`docker-compose.yml`)
- Added volume mount: `./frontend:/usr/share/nginx/html`
- This ensures the frontend files are available to nginx

### 3. Deployment Scripts
- `deploy-frontend.sh` (Linux/Mac)
- `deploy-frontend.ps1` (Windows PowerShell)

## 🚀 Deployment Instructions

### Option 1: Using Deployment Script (Recommended)

#### Windows:
```powershell
.\deploy-frontend.ps1
```

#### Linux/Mac:
```bash
./deploy-frontend.sh
```

### Option 2: Manual Deployment

1. **Stop and remove nginx container:**
   ```bash
   docker-compose stop nginx
   docker-compose rm -f nginx
   ```

2. **Start nginx with new configuration:**
   ```bash
   docker-compose up -d nginx
   ```

3. **Verify deployment:**
   ```bash
   docker-compose ps nginx
   docker-compose logs nginx
   ```

## 🌐 Access Points

After successful deployment, the following endpoints will be available:

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | `http://173.212.220.106` | Main VibraOps interface |
| API | `http://173.212.220.106/api/` | Backend API endpoints |
| Health Check | `http://173.212.220.106/health` | API health status |
| Grafana | `http://173.212.220.106/grafana/` | Monitoring dashboard |
| Prometheus | `http://173.212.220.106/prometheus/` | Metrics collection |

## 🔧 Configuration Details

### NGINX Configuration Features
- **Static File Serving**: Serves frontend files from `/usr/share/nginx/html`
- **API Proxy**: Routes `/api/` requests to the vibraops-api service
- **SPA Support**: `try_files` directive for single-page application routing
- **Security Headers**: XSS protection, content type options, etc.
- **Error Handling**: Custom error pages for 404 and 5xx errors

### Volume Mounts
```yaml
volumes:
  - ./monitoring/nginx:/etc/nginx/conf.d    # NGINX configuration
  - ./frontend:/usr/share/nginx/html        # Frontend static files
  - ./ssl:/etc/ssl/certs                    # SSL certificates (if needed)
```

## 🐛 Troubleshooting

### Common Issues

1. **Frontend still not accessible**
   - Check if nginx container is running: `docker-compose ps nginx`
   - View nginx logs: `docker-compose logs nginx`
   - Verify frontend files exist: `ls -la frontend/`

2. **API proxy not working**
   - Ensure vibraops-api service is running: `docker-compose ps vibraops-api`
   - Check API logs: `docker-compose logs vibraops-api`
   - Test API directly: `curl http://localhost:8000/health`

3. **Permission issues**
   - Ensure frontend directory has proper permissions
   - Check nginx container logs for permission errors

### Debug Commands
```bash
# Check all service status
docker-compose ps

# View nginx configuration
docker exec vibraops-nginx cat /etc/nginx/conf.d/default.conf

# Test nginx configuration
docker exec vibraops-nginx nginx -t

# Check frontend files in container
docker exec vibraops-nginx ls -la /usr/share/nginx/html

# View real-time logs
docker-compose logs -f nginx
```

## 🔄 Next Steps

1. **Test all endpoints** to ensure they're working correctly
2. **Monitor logs** for any errors or issues
3. **Consider SSL setup** for production use
4. **Set up monitoring** to track frontend performance
5. **Implement CI/CD** for automated deployments

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the nginx logs: `docker-compose logs nginx`
3. Verify all services are running: `docker-compose ps`
4. Test individual components separately

---

**Status**: ✅ Frontend deployment issue resolved
**Last Updated**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss") 