# Zeek Network Analysis Container
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Zeek and dependencies
RUN apt-get update && apt-get install -y \
    curl=7.81.0-1ubuntu1.20 \
    wget=1.21.2-2ubuntu1.1 \
    && echo 'deb http://download.opensuse.org/repositories/security:/zeek/xUbuntu_22.04/ /' | tee /etc/apt/sources.list.d/security:zeek.list \
    && curl -fsSL https://download.opensuse.org/repositories/security:zeek/xUbuntu_22.04/Release.key | gpg --dearmor | tee /etc/apt/trusted.gpg.d/security_zeek.gpg > /dev/null \
    && apt-get update \
    && apt-get install -y zeek \
    && rm -rf /var/lib/apt/lists/*

# Set up Zeek environment
ENV PATH="/opt/zeek/bin:${PATH}"

# Create working directory
WORKDIR /app

# Create data directories
RUN mkdir -p /app/data/input \
    /app/data/zeek \
    /app/data/input_parsed

# Copy Zeek processing script
COPY docker/scripts/process_pcap.sh /app/process_pcap.sh
RUN chmod +x /app/process_pcap.sh

# Default command - keep container running for processing requests
CMD ["tail", "-f", "/dev/null"]
