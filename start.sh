#!/bin/bash

# PDF Q&A System Startup Script
# This script provides various ways to start the application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="development"
PORT=8000
HOST="0.0.0.0"
WORKERS=1

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  PDF Q&A System Startup${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        print_error "pip is not installed"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment not found. Creating one..."
        python3 -m venv venv
    fi
    
    print_status "Dependencies check completed"
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install/upgrade dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Create necessary directories
    mkdir -p logs uploads chroma_db
    
    # Copy environment file if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_warning "No .env file found. Copying from .env.example"
            cp .env.example .env
            print_warning "Please edit .env file with your configuration"
        else
            print_error "No .env.example file found"
            exit 1
        fi
    fi
    
    print_status "Environment setup completed"
}

# Function to run health check
run_health_check() {
    print_status "Running health check..."
    
    # Wait for service to start
    sleep 5
    
    # Check if service is responding
    if curl -f "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
        print_status "Health check passed"
        return 0
    else
        print_error "Health check failed"
        return 1
    fi
}

# Function to start development server
start_development() {
    print_status "Starting development server..."
    
    export DEBUG=true
    export LOG_LEVEL=DEBUG
    export ENVIRONMENT=development
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start the server
    python main.py
}

# Function to start production server
start_production() {
    print_status "Starting production server..."
    
    export DEBUG=false
    export LOG_LEVEL=INFO
    export ENVIRONMENT=production
    export WORKERS=${WORKERS}
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start the server
    python run.py server
}

# Function to start with Docker
start_docker() {
    print_status "Starting with Docker..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Build and start containers
    if [ "$MODE" = "production" ]; then
        docker-compose --profile production up --build
    else
        docker-compose up --build
    fi
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run tests
    python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term
    
    print_status "Tests completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  dev         Start development server (default)"
    echo "  prod        Start production server"
    echo "  docker      Start with Docker"
    echo "  test        Run tests"
    echo "  health      Run health check"
    echo "  setup       Setup environment only"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT     Port to run on (default: 8000)"
    echo "  -h, --host HOST     Host to bind to (default: 0.0.0.0)"
    echo "  -w, --workers NUM   Number of workers for production (default: 1)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                  # Start development server"
    echo "  $0 prod -w 4        # Start production server with 4 workers"
    echo "  $0 docker           # Start with Docker"
    echo "  $0 test             # Run tests"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        dev|development)
            MODE="development"
            shift
            ;;
        prod|production)
            MODE="production"
            shift
            ;;
        docker)
            MODE="docker"
            shift
            ;;
        test)
            MODE="test"
            shift
            ;;
        health)
            MODE="health"
            shift
            ;;
        setup)
            MODE="setup"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_header

case $MODE in
    development|dev)
        check_dependencies
        setup_environment
        start_development
        ;;
    production|prod)
        check_dependencies
        setup_environment
        start_production
        ;;
    docker)
        start_docker
        ;;
    test)
        check_dependencies
        setup_environment
        run_tests
        ;;
    health)
        run_health_check
        ;;
    setup)
        check_dependencies
        setup_environment
        print_status "Environment setup completed. You can now start the server."
        ;;
    *)
        print_error "Invalid mode: $MODE"
        show_usage
        exit 1
        ;;
esac