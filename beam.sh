#!/bin/bash

# BEAM - Easy Docker wrapper for running BEAM commands
# This script handles all Docker complexity so you can use BEAM naturally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to show usage
usage() {
    echo "🔒 BEAM - Behavioral Evaluation of Application Metrics"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --train                     Train custom models (auto-discovers apps)"
    echo "  --input PATH               Path to input file or directory (default: ./data/input)"
    echo "  --use_custom_models        Explicitly use custom models for detection"
    echo "  --demo                     Run the interactive demo"
    echo "  --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run the interactive demo"
    echo "  $0 --demo"
    echo ""
    echo "  # Train models from a specific HAR file (auto-discovers apps)"
    echo "  $0 --train --input /path/to/traffic.har"
    echo ""
    echo "  # Train models for all apps found in a directory"
    echo "  $0 --train --input /path/to/pcap_files/"
    echo ""
    echo "  # Run detection (auto-detects if custom models available)"
    echo "  $0"
    echo ""
    echo "  # Explicitly use custom models"
    echo "  $0 --use_custom_models"
}

# Source the shared environment setup script
source "$SCRIPT_DIR/docker/setup-environment.sh"

# Function to convert host path to absolute path
get_absolute_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
    fi
}

# Parse command line arguments
BEAM_ARGS=""
INPUT_PATH=""
CUSTOM_INPUT=false
DEMO_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            usage
            exit 0
            ;;
        --input)
            INPUT_PATH="$2"
            CUSTOM_INPUT=true
            shift 2
            ;;
        -i)
            INPUT_PATH="$2"
            CUSTOM_INPUT=true
            shift 2
            ;;
        --demo)
            DEMO_MODE=true
            shift
            ;;
        *)
            BEAM_ARGS="$BEAM_ARGS $1"
            shift
            ;;
    esac
done

# Handle demo mode
if [ "$DEMO_MODE" = true ]; then
    echo "🔒 BEAM Supply Chain Compromise Detection Demo"
    echo "Running demo mode through unified interface..."
    echo ""
    
    # Use the existing demo system for now, but could be integrated further
    exec ./docker/run-demo.sh
fi

# Setup environment (dependencies + containers)
setup_environment "main"

# Handle input path
if [ "$CUSTOM_INPUT" = true ]; then
    if [ -z "$INPUT_PATH" ]; then
        echo -e "${RED}❌ --input requires a path${NC}"
        exit 1
    fi
    
    # Get absolute path
    ABS_INPUT_PATH=$(get_absolute_path "$INPUT_PATH")
    
    if [ ! -e "$ABS_INPUT_PATH" ]; then
        echo -e "${RED}❌ Input path does not exist: $INPUT_PATH${NC}"
        exit 1
    fi
    
    # Create a temporary directory for this session
    TEMP_DIR=$(mktemp -d "$SCRIPT_DIR/data/temp.XXXXXX")
    
    # Copy the input file(s) to the temp directory
    if [ -f "$ABS_INPUT_PATH" ]; then
        cp "$ABS_INPUT_PATH" "$TEMP_DIR/"
        echo -e "${GREEN}✓ Using input file: $(basename "$INPUT_PATH")${NC}"
    elif [ -d "$ABS_INPUT_PATH" ]; then
        cp -r "$ABS_INPUT_PATH"/* "$TEMP_DIR/" 2>/dev/null || true
        FILE_COUNT=$(find "$TEMP_DIR" -type f | wc -l | tr -d ' ')
        echo -e "${GREEN}✓ Using input directory with $FILE_COUNT files${NC}"
    fi
    
    # Update the input path for the container
    # The temp directory is created under $SCRIPT_DIR/data/, so it will be mounted at /app/data/
    TEMP_DIR_NAME=$(basename "$TEMP_DIR")
    CONTAINER_INPUT="/app/data/$TEMP_DIR_NAME"
    BEAM_ARGS="$BEAM_ARGS -i $CONTAINER_INPUT"
fi

# Prepare Docker run command
echo -e "${BLUE}🚀 Running BEAM...${NC}"
echo ""

# Start required services in the background
$COMPOSE_CMD -f "$SCRIPT_DIR/docker-compose.yml" up -d zeek-processor database >/dev/null 2>&1

# Run BEAM in the container
$COMPOSE_CMD -f "$SCRIPT_DIR/docker-compose.yml" run --rm \
    beam-core python -m beam $BEAM_ARGS

# Cleanup
if [ "$CUSTOM_INPUT" = true ] && [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

# Stop background services
$COMPOSE_CMD -f "$SCRIPT_DIR/docker-compose.yml" stop zeek-processor database >/dev/null 2>&1

echo ""
echo -e "${GREEN}✅ BEAM completed successfully!${NC}"

# Show where results are located
if [[ "$BEAM_ARGS" == *"--train"* ]]; then
    echo -e "${BLUE}📁 Custom models saved to: ./models/custom_models/${NC}"
else
    echo -e "${BLUE}📁 Detection results saved to: ./predictions/${NC}"
fi