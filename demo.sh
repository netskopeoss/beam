#!/bin/bash

# BEAM Quick Demo Launcher
# Place this in the root directory for easy access

echo "🔒 BEAM Supply Chain Compromise Detection"
echo "Starting demo with Docker..."
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the BEAM project root directory"
    exit 1
fi

# Run the demo
exec ./docker/run-demo.sh "$@"
