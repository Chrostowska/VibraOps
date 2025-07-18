#!/bin/bash

# VibraOps Quick Start Script
# This script sets up and runs the complete VibraOps system

set -e

echo "🚀 VibraOps Quick Start Script"
echo "==============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "All prerequisites are installed!"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    mkdir -p logs models data config
    log_success "Directories created!"
}

# Install Python dependencies (for local development)
install_dependencies() {
    if [ "$1" = "local" ]; then
        log_info "Installing Python dependencies..."
        
        # Check if virtual environment exists
        if [ ! -d "venv" ]; then
            log_info "Creating virtual environment..."
            python3 -m venv venv
        fi
        
        # Activate virtual environment
        source venv/bin/activate
        
        # Install dependencies
        pip install --upgrade pip
        pip install -r requirements.txt
        
        log_success "Python dependencies installed!"
    fi
}

# Generate sample data and train models
setup_models() {
    if [ "$1" = "local" ]; then
        log_info "Setting up models (this may take a few minutes)..."
        
        # Activate virtual environment
        source venv/bin/activate
        
        # Generate sample data
        log_info "Generating sample vibration data..."
        python src/data_simulator.py
        
        # Train models
        log_info "Training anomaly detection models..."
        python src/models.py
        
        log_success "Models trained and ready!"
    else
        log_warning "Skipping model training for Docker deployment (models will be generated on first run)"
    fi
}

# Start services with Docker Compose
start_docker_services() {
    log_info "Starting services with Docker Compose..."
    
    # Build and start services
    docker-compose up --build -d
    
    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        log_success "Services are running!"
    else
        log_error "Some services failed to start. Check logs with: docker-compose logs"
        exit 1
    fi
}

# Start local development server
start_local_server() {
    log_info "Starting local development server..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start the API server
    log_info "Starting FastAPI server on http://localhost:8000"
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload &
    
    # Store PID for cleanup
    echo $! > .api_pid
    
    log_success "Local server started!"
}

# Test the API
test_api() {
    local base_url=$1
    log_info "Testing API at $base_url..."
    
    # Wait for API to be ready
    for i in {1..30}; do
        if curl -s "$base_url/health" >/dev/null 2>&1; then
            log_success "API is responding!"
            break
        fi
        
        if [ $i -eq 30 ]; then
            log_error "API is not responding after 30 attempts"
            return 1
        fi
        
        sleep 2
    done
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    curl -s "$base_url/health" | python3 -m json.tool
    
    # Test prediction endpoint with sample data
    log_info "Testing prediction endpoint with sample data..."
    curl -X POST "$base_url/predict" \
         -H "Content-Type: application/json" \
         -d '{
           "signal": [0.1, 0.2, 0.15, 0.18, 5.0, 0.12, 0.14, 0.16],
           "metadata": {"sensor_id": "test_sensor"}
         }' | python3 -m json.tool
    
    log_success "API tests completed!"
}

# Show service URLs
show_urls() {
    echo ""
    echo "🌐 Service URLs:"
    echo "==============="
    echo "• API Service:      http://localhost:8000"
    echo "• API Documentation: http://localhost:8000/docs"
    echo "• Grafana:          http://localhost:3000 (admin/admin123)"
    echo "• Prometheus:       http://localhost:9090"
    echo ""
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    if [ -f ".api_pid" ]; then
        kill $(cat .api_pid) 2>/dev/null || true
        rm .api_pid
    fi
}

# Main script
main() {
    local mode=${1:-docker}
    
    if [ "$mode" != "local" ] && [ "$mode" != "docker" ]; then
        echo "Usage: $0 [local|docker]"
        echo "  local:  Run in local development mode"
        echo "  docker: Run with Docker Compose (default)"
        exit 1
    fi
    
    echo "🎯 Starting VibraOps in $mode mode..."
    echo ""
    
    # Setup trap for cleanup
    trap cleanup EXIT
    
    # Run setup steps
    check_prerequisites
    create_directories
    install_dependencies "$mode"
    setup_models "$mode"
    
    if [ "$mode" = "docker" ]; then
        start_docker_services
        test_api "http://localhost:8000"
        show_urls
        
        echo "🎉 VibraOps is now running!"
        echo "💡 Use 'docker-compose logs -f' to view logs"
        echo "🛑 Use 'docker-compose down' to stop services"
        
    else
        start_local_server
        test_api "http://localhost:8000"
        show_urls
        
        echo "🎉 VibraOps is now running in local mode!"
        echo "💡 Check the terminal for API logs"
        echo "🛑 Press Ctrl+C to stop the server"
        
        # Wait for user interrupt
        wait
    fi
}

# Run main function with all arguments
main "$@" 