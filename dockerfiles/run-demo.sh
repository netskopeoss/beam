#!/bin/bash

# BEAM Demo Runner Script
# Makes it easy to run the BEAM demo with Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔒 BEAM Supply Chain Compromise Detection Demo${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    echo -e "${RED}❌ Docker Compose is not available. Please install Docker Compose.${NC}"
    exit 1
fi

# Function to run demo
run_demo() {
    echo -e "${YELLOW}🚀 Starting BEAM Demo...${NC}"
    echo ""
    
    # Create demo results directory
    mkdir -p demo_results
    
    # Use docker compose (new) or docker-compose (legacy)
    if docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    # Build and run the demo
    echo -e "${BLUE}📦 Building BEAM Demo container...${NC}"
    $COMPOSE_CMD -f docker-compose.demo.yml build beam-demo
    
    echo -e "${BLUE}🔍 Running Supply Chain Compromise Detection Demo...${NC}"
    $COMPOSE_CMD -f docker-compose.demo.yml run --rm beam-demo
    
    echo ""
    echo -e "${GREEN}✅ Demo completed successfully!${NC}"
    echo -e "${YELLOW}📄 Demo results saved to: ./demo_results/${NC}"
    echo -e "${BLUE}📊 Files created:${NC}"
    echo -e "${BLUE}   • beacon_demo_security_report.txt - Detailed security analysis${NC}"
    echo -e "${BLUE}   • beacon_demo_features.json - Extracted ML features${NC}"
    echo -e "${BLUE}   • beacon_demo_enriched.json - Enriched network events${NC}"
    echo -e "${BLUE}   • predictions/ - Per-application anomaly predictions${NC}"
}

# Function to run interactive shell
run_interactive() {
    echo -e "${YELLOW}🖥️  Starting interactive BEAM shell...${NC}"
    echo ""
    
    mkdir -p demo_results
    
    if docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    echo -e "${BLUE}📦 Building BEAM Demo container...${NC}"
    $COMPOSE_CMD -f docker-compose.demo.yml build beam-demo
    
    echo -e "${BLUE}🔍 Starting interactive shell...${NC}"
    echo -e "${YELLOW}💡 Inside the container, you can run:${NC}"
    echo -e "${YELLOW}   • python -c \"from beam.demo import run_demo; run_demo()\"${NC}"
    echo -e "${YELLOW}   • python -m beam --help${NC}"
    echo -e "${YELLOW}   • Explore the /app directory${NC}"
    echo ""
    
    $COMPOSE_CMD -f docker-compose.demo.yml run --rm beam-interactive
}

# Function to clean up
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up Docker containers and images...${NC}"
    
    if docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    $COMPOSE_CMD -f docker-compose.demo.yml down --rmi all --volumes --remove-orphans
    
    echo -e "${GREEN}✅ Cleanup completed!${NC}"
}

# Check command line arguments
case "${1:-demo}" in
    "demo")
        run_demo
        ;;
    "interactive")
        run_interactive
        ;;
    "cleanup")
        cleanup
        ;;
    "help")
        echo -e "${BLUE}BEAM Demo Runner${NC}"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  demo        Run the automated demo (default)"
        echo "  interactive Start an interactive shell"
        echo "  cleanup     Remove Docker containers and images"
        echo "  help        Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 demo        # Run the demo"
        echo "  $0 interactive # Start interactive shell"
        echo "  $0 cleanup     # Clean up Docker resources"
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo -e "${YELLOW}Run '$0 help' for usage information.${NC}"
        exit 1
        ;;
esac
