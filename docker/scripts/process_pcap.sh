#!/bin/bash

# PCAP Processing Script for Zeek Container
# This script processes PCAP files using Zeek and outputs JSON logs

set -e

PCAP_FILE="$1"
OUTPUT_DIR="$2"

if [ -z "$PCAP_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <pcap_file> <output_directory>"
    echo "Example: $0 /app/data/input/sample.pcap /app/data/zeek"
    exit 1
fi

if [ ! -f "$PCAP_FILE" ]; then
    echo "Error: PCAP file $PCAP_FILE not found"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Processing PCAP file: $PCAP_FILE"
echo "Output directory: $OUTPUT_DIR"

# Run Zeek with JSON output format
cd "$OUTPUT_DIR"
zeek -r "$PCAP_FILE" \
    LogAscii::use_json=T \
    -e 'redef LogAscii::json_timestamps = JSON::TS_ISO8601;'

# Check if processing was successful
if [ $? -eq 0 ]; then
    echo "Zeek processing completed successfully"
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"/*.log 2>/dev/null || echo "No .log files found"
else
    echo "Error: Zeek processing failed"
    exit 1
fi
