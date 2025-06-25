#!/bin/bash

# BEAM Environment Setup Script
# Shared script for ensuring dependencies and Docker containers are ready
# Used by both beam.sh and run-demo.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running${NC}"
        echo "Please start Docker Desktop and try again."
        exit 1
    fi
}

# Function to check Docker Compose availability
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
        echo -e "${RED}❌ Docker Compose is not available. Please install Docker Compose.${NC}"
        exit 1
    fi
    
    # Set the compose command to use
    if docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    export COMPOSE_CMD
}

# Function to check if uv is available and update lock file
ensure_dependencies() {
    echo -e "${BLUE}📦 Checking dependencies...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Check if uv is available
    if command -v uv >/dev/null 2>&1; then
        # Check if pyproject.toml is newer than uv.lock
        if [ ! -f "uv.lock" ] || [ "pyproject.toml" -nt "uv.lock" ]; then
            echo -e "${YELLOW}⚡ Updating dependency lock file...${NC}"
            if ! uv lock; then
                echo -e "${RED}❌ Failed to update uv.lock${NC}"
                exit 1
            fi
            echo -e "${GREEN}✓ Dependencies locked successfully${NC}"
        else
            echo -e "${GREEN}✓ Dependencies are up-to-date${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  uv not found - using existing uv.lock file${NC}"
        if [ ! -f "uv.lock" ]; then
            echo -e "${RED}❌ No uv.lock file found and uv not available${NC}"
            echo "Please install uv or ensure uv.lock exists"
            exit 1
        fi
    fi
}

# Function to build main BEAM containers if needed
ensure_beam_containers() {
    echo -e "${BLUE}🐳 Preparing BEAM containers...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Check if the beam-core image exists
    if ! docker images | grep -q "beam-beam-core"; then
        echo -e "${YELLOW}Building containers for the first time (this may take a few minutes)...${NC}"
        if ! $COMPOSE_CMD -f docker-compose.yml build beam-core zeek-processor database; then
            echo -e "${RED}❌ Failed to build containers${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Containers built successfully${NC}"
    else
        echo -e "${GREEN}✓ Containers are available${NC}"
        echo -e "${BLUE}💡 Tip: If you've updated dependencies, run 'docker-compose build' to rebuild containers${NC}"
    fi
}

# Function to build demo containers if needed
ensure_demo_containers() {
    local use_llama="${1:-true}"
    
    echo -e "${BLUE}📦 Checking BEAM Demo containers...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Check if demo container exists
    if ! docker images | grep -q "beam-beam-demo"; then
        echo -e "${YELLOW}Building demo containers for the first time...${NC}"
        if [ "$use_llama" == "true" ]; then
            $COMPOSE_CMD -f docker-compose.demo.yml build database llama-model beam-demo
        else
            $COMPOSE_CMD -f docker-compose.demo.yml build database beam-demo
        fi
        echo -e "${GREEN}✓ Demo containers built successfully${NC}"
    else
        echo -e "${GREEN}✓ Demo containers are available${NC}"
        echo -e "${BLUE}💡 Tip: If you've updated dependencies, run 'docker-compose -f docker-compose.demo.yml build' to rebuild${NC}"
    fi
}

# Function to setup and start Llama service
setup_llama_service() {
    echo -e "${BLUE}🤖 Starting local Llama model service...${NC}"
    
    cd "$PROJECT_ROOT"
    
    $COMPOSE_CMD -f docker-compose.demo.yml up -d --remove-orphans database llama-model
    
    # Wait for Llama to be ready and download model if needed
    echo -e "${YELLOW}⏳ Waiting for Llama service to start...${NC}"
    sleep 5
    
    # Check if model needs to be downloaded
    if ! $COMPOSE_CMD -f docker-compose.demo.yml exec llama-model ollama list | grep -q "llama3.2:1b"; then
        echo -e "${BLUE}📥 Downloading Llama 3.2 model (one-time setup, ~1.3GB)...${NC}"
        $COMPOSE_CMD -f docker-compose.demo.yml exec llama-model ollama pull llama3.2:1b
    fi
    
    echo -e "${GREEN}✓ Llama service ready${NC}"
}

# Main function to setup complete environment
setup_environment() {
    local mode="${1:-main}"  # main, demo, or demo-interactive
    local use_llama="${2:-true}"
    
    echo -e "${BLUE}🔧 Setting up BEAM environment...${NC}"
    
    # Always check Docker and dependencies first
    check_docker
    check_docker_compose
    ensure_dependencies
    
    case "$mode" in
        "main")
            ensure_beam_containers
            ;;
        "demo"|"demo-interactive")
            ensure_demo_containers "$use_llama"
            if [ "$use_llama" == "true" ]; then
                setup_llama_service
            fi
            ;;
        *)
            echo -e "${RED}❌ Unknown setup mode: $mode${NC}"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}✅ Environment setup complete${NC}"
}

# If script is run directly, setup main environment
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_environment "main"
fi