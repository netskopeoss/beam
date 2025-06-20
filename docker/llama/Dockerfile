FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create app user
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

# Expose port for Ollama API
EXPOSE 11434

# Start Ollama server
CMD ["ollama", "serve"]