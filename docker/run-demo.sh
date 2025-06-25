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

# Get the directory where this script is located and source shared setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup-environment.sh"

# Function to run demo
run_demo() {
    local use_llama="${USE_LOCAL_LLM:-true}"
    
    if [ "$use_llama" == "true" ]; then
        echo -e "${YELLOW}🚀 Starting BEAM Demo with Local Llama Model...${NC}"
        echo -e "${BLUE}🤖 Using local AI model - no API keys required!${NC}"
    else
        echo -e "${YELLOW}🚀 Starting BEAM Demo with Gemini API...${NC}"
        echo -e "${BLUE}☁️  Using cloud-based Gemini model${NC}"
    fi
    echo ""
    
    # Setup environment (dependencies + containers + services)
    setup_environment "demo" "$use_llama"
    
    # Create demo results and logs directories
    mkdir -p demo_results logs
    
    echo -e "${BLUE}🔍 Running Supply Chain Compromise Detection Demo...${NC}"
    $COMPOSE_CMD -f docker-compose.demo.yml run --rm --remove-orphans beam-demo
    
    # Stop Llama service
    if [ "$use_llama" == "true" ]; then
        echo -e "${BLUE}🛑 Stopping Llama service...${NC}"
        $COMPOSE_CMD -f docker-compose.demo.yml stop llama-model
    fi
    
    echo ""
    echo -e "${GREEN}✅ Demo completed successfully!${NC}"
    echo -e "${YELLOW}📄 Demo results saved to: ./demo_results/${NC}"
    echo -e "${BLUE}📊 Files created:${NC}"
    echo -e "${BLUE}   • beacon_demo_security_report.txt - Detailed security analysis${NC}"
    echo -e "${BLUE}   • beacon_demo_features.json - Extracted ML features${NC}"
    echo -e "${BLUE}   • beacon_demo_enriched.json - Enriched network events${NC}"
    echo -e "${BLUE}   • predictions/ - Per-application anomaly predictions${NC}"
    echo -e "${BLUE}   • beam.log - Complete execution logs${NC}"
}

# Function to run interactive shell
run_interactive() {
    local use_llama="${USE_LOCAL_LLM:-true}"
    
    echo -e "${YELLOW}🖥️  Starting interactive BEAM shell...${NC}"
    if [ "$use_llama" == "true" ]; then
        echo -e "${BLUE}🤖 With local Llama model${NC}"
    else
        echo -e "${BLUE}☁️  With cloud-based Gemini model${NC}"
    fi
    echo ""
    
    # Setup environment (dependencies + containers + services)
    setup_environment "demo-interactive" "$use_llama"
    
    mkdir -p demo_results logs
    
    echo -e "${BLUE}🔍 Starting interactive shell...${NC}"
    echo -e "${YELLOW}💡 Inside the container, you can run:${NC}"
    echo -e "${YELLOW}   • python -c \"from beam.demo import run_demo; run_demo()\"${NC}"
    echo -e "${YELLOW}   • python -m beam --help${NC}"
    echo -e "${YELLOW}   • Explore the /app directory${NC}"
    echo ""
    
    $COMPOSE_CMD -f docker-compose.demo.yml run --rm --remove-orphans --profile interactive beam-interactive
    
    # Stop Llama service
    if [ "$use_llama" == "true" ]; then
        echo -e "${BLUE}🛑 Stopping Llama service...${NC}"
        $COMPOSE_CMD -f docker-compose.demo.yml stop llama-model
    fi
}

# Function to clean up
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up Docker containers and images...${NC}"
    
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
        echo "Environment Variables:"
        echo "  USE_LOCAL_LLM  Set to 'false' to use Gemini API instead of local Llama"
        echo "                 (default: true - uses local Llama model)"
        echo ""
        echo "Examples:"
        echo "  $0 demo                    # Run demo with local Llama (default)"
        echo "  USE_LOCAL_LLM=false $0 demo  # Run demo with Gemini API"
        echo "  $0 interactive             # Start interactive shell"
        echo "  $0 cleanup                 # Clean up Docker resources"
        echo ""
        echo "Note: Local Llama model requires ~1.3GB download on first run."
        echo "      Gemini API requires GEMINI_API_KEY environment variable."
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo -e "${YELLOW}Run '$0 help' for usage information.${NC}"
        exit 1
        ;;
esac
