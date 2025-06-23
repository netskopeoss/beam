#!/bin/bash

# Container Health Check Script
# This script can be used to verify that containers are healthy and services are running

set -e

SERVICE="$1"

case "$SERVICE" in
    "beam-core")
        echo "Checking BEAM Core service..."
        python -m beam --help > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ BEAM Core is healthy"
            exit 0
        else
            echo "✗ BEAM Core is not responding"
            exit 1
        fi
        ;;
    
    "zeek")
        echo "Checking Zeek service..."
        zeek --version > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ Zeek is healthy"
            exit 0
        else
            echo "✗ Zeek is not available"
            exit 1
        fi
        ;;
    
    "database")
        echo "Checking Database service..."
        if [ -f "/app/data/mapper/user_agent_mapping.db" ]; then
            sqlite3 /app/data/mapper/user_agent_mapping.db "SELECT 1;" > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo "✓ Database is healthy"
                exit 0
            else
                echo "✗ Database is not accessible"
                exit 1
            fi
        else
            echo "✗ Database file not found"
            exit 1
        fi
        ;;
    
    *)
        echo "Usage: $0 {beam-core|zeek|database}"
        echo "Health check script for BEAM containers"
        exit 1
        ;;
esac
