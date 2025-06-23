#!/bin/bash

# BEAM Container Management Script
# Provides easy commands for managing BEAM containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "BEAM Container Management Script"
    echo ""
    echo "Usage: $0 COMMAND [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build           Build all containers"
    echo "  up              Start all containers"
    echo "  down            Stop all containers"
    echo "  restart         Restart all containers"
    echo "  logs [SERVICE]  Show logs (optionally for specific service)"
    echo "  exec SERVICE    Execute bash in a service container"
    echo "  health          Check health of all services"
    echo "  clean           Clean up containers and volumes"
    echo "  single          Run single all-in-one container"
    echo ""
    echo "Services: beam-core, zeek-processor, database"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build all containers"
    echo "  $0 up                       # Start multi-container setup"
    echo "  $0 exec beam-core          # Access BEAM core container"
    echo "  $0 logs zeek-processor     # Show Zeek logs"
    echo "  $0 single                  # Run single container"
}

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_requirements() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
}

build() {
    log "Building BEAM containers..."
    cd "$PROJECT_DIR"
    docker-compose build
    success "Containers built successfully"
}

up() {
    log "Starting BEAM containers..."
    cd "$PROJECT_DIR"
    docker-compose up -d
    success "Containers started successfully"
    
    # Wait a moment and check health
    sleep 5
    health
}

down() {
    log "Stopping BEAM containers..."
    cd "$PROJECT_DIR"
    docker-compose down
    success "Containers stopped successfully"
}

restart() {
    log "Restarting BEAM containers..."
    down
    sleep 2
    up
}

logs() {
    cd "$PROJECT_DIR"
    if [ -n "$1" ]; then
        log "Showing logs for $1..."
        docker-compose logs -f "$1"
    else
        log "Showing logs for all services..."
        docker-compose logs -f
    fi
}

exec_service() {
    if [ -z "$1" ]; then
        error "Service name required for exec command"
        echo "Available services: beam-core, zeek-processor, database"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    log "Executing bash in $1 container..."
    docker-compose exec "$1" bash
}

health() {
    log "Checking health of BEAM services..."
    cd "$PROJECT_DIR"
    
    services=("beam-core" "zeek-processor" "database")
    
    for service in "${services[@]}"; do
        if docker-compose ps -q "$service" | grep -q .; then
            if docker-compose exec -T "$service" /app/scripts/healthcheck.sh "$service" 2>/dev/null; then
                success "$service is healthy"
            else
                error "$service is unhealthy"
            fi
        else
            warning "$service is not running"
        fi
    done
}

clean() {
    log "Cleaning up BEAM containers and volumes..."
    cd "$PROJECT_DIR"
    
    read -p "This will remove all containers, networks, and volumes. Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v --remove-orphans
        docker-compose rm -f
        success "Cleanup completed"
    else
        log "Cleanup cancelled"
    fi
}

single() {
    log "Starting BEAM all-in-one container..."
    cd "$PROJECT_DIR"
    docker-compose --profile single-container up -d beam-all-in-one
    success "All-in-one container started successfully"
}

# Main command handling
case "$1" in
    build)
        check_requirements
        build
        ;;
    up)
        check_requirements
        up
        ;;
    down)
        check_requirements
        down
        ;;
    restart)
        check_requirements
        restart
        ;;
    logs)
        check_requirements
        logs "$2"
        ;;
    exec)
        check_requirements
        exec_service "$2"
        ;;
    health)
        check_requirements
        health
        ;;
    clean)
        check_requirements
        clean
        ;;
    single)
        check_requirements
        single
        ;;
    *)
        usage
        exit 1
        ;;
esac
