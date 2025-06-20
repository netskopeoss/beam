# BEAM Docker Setup

This directory contains Docker configurations for running BEAM (Behavioral Evaluation of Application Metrics) in containerized environments, with a focus on making the demo experience seamless across all platforms.

## 🚀 Quick Demo Start

**The fastest way to see BEAM in action:**

```bash
# Clone the repository
git clone <repository-url>
cd beam

# Run the demo (one command!)
./docker/run-demo.sh
```

That's it! The demo will automatically:
- Build the Docker container with all dependencies
- Run the supply chain compromise detection demo
- Show you how BEAM detects malicious behavior in network traffic
- Save results to `./demo_results/`

## 📋 Prerequisites

- Docker Desktop installed and running
- Docker Compose (included with Docker Desktop)
- ~2GB free disk space for container images

## 🐳 Container Options

### 1. Demo Container (Recommended for First-Time Users)
**File:** `Dockerfile.demo`
- **Purpose:** Optimized for running the interactive demo
- **Includes:** All demo data, pre-trained models, sample HAR files
- **Best for:** Quick evaluation, learning, showcasing BEAM capabilities

```bash
# Build and run demo
docker-compose -f docker-compose.demo.yml run --rm beam-demo

# Or use the convenience script
./docker/run-demo.sh demo
```

### 2. Core Application Container
**File:** `Dockerfile.beam-core`
- **Purpose:** BEAM application without demo data
- **Best for:** Production deployments, custom data analysis
- **Usage:** Mount your own data volumes

```bash
# Build core container
docker-compose build beam-core

# Run with your data
docker-compose run --rm beam-core python -m beam --help
```

### 3. All-in-One Container
**File:** `Dockerfile.all-in-one`
- **Purpose:** Complete BEAM environment with all tools
- **Includes:** BEAM + all analysis tools
- **Best for:** Complex analysis workflows, custom model training

```bash
# Build and run all-in-one
docker-compose build beam-all-in-one
docker-compose run --rm beam-all-in-one
```

## 🎯 Demo Experience

The demo container is designed to provide a **zero-friction experience**:

### What the Demo Shows
1. **Real Network Traffic:** Box cloud storage app with hidden malware
2. **Supply Chain Compromise:** Legitimate app infected with malicious code
3. **AI Detection:** How BEAM's ML models spot suspicious behavior
4. **Security Analysis:** Comprehensive threat assessment report

### Demo Commands
```bash
# Run automated demo
./docker/run-demo.sh demo

# Interactive exploration
./docker/run-demo.sh interactive

# Clean up containers
./docker/run-demo.sh cleanup
```

## 🏗️ Modern Dependencies with uv

All containers use **uv** for fast, reliable Python dependency management:

- **Faster installs:** uv is 10-100x faster than pip
- **Reliable builds:** Uses `uv.lock` for reproducible environments
- **Better caching:** Docker layers cache dependencies efficiently
- **No requirements.txt:** Uses modern `pyproject.toml` + `uv.lock`

## 📁 Directory Structure

```
docker/
├── Dockerfile.demo          # Demo-optimized container
├── Dockerfile.beam-core     # Core application container  
├── Dockerfile.all-in-one    # Full-featured container
├── Dockerfile.database      # Database services
├── Dockerfile.zeek          # Network analysis tools
├── run-demo.sh             # Demo runner script
├── scripts/                # Container initialization scripts
└── README.md              # This file

docker-compose.yml          # Multi-container setup
docker-compose.demo.yml     # Demo-focused setup
```

## 🛠️ Development Workflow

### For Demo Users
```bash
# Quick start
./docker/run-demo.sh

# Interactive exploration
./docker/run-demo.sh interactive
```

### For Developers
```bash
# Build development environment
docker-compose build beam-core

# Run with development data mounted
docker-compose run --rm -v $(pwd)/data:/app/data beam-core python -m beam

# Run tests
docker-compose run --rm beam-core python -m pytest
```

### For Custom Analysis
```bash
# Mount your HAR/PCAP files
docker-compose run --rm \
  -v /path/to/your/data:/app/data/input \
  -v /path/to/results:/app/predictions \
  beam-core python -m beam --input /app/data/input
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set Gemini API key for advanced analysis
export GEMINI_API_KEY="your-api-key"

# Optional: Set log level
export LOG_LEVEL="DEBUG"
```

### Volume Mounts
- **Input Data:** Mount to `/app/data/input`
- **Results:** Mount to `/app/predictions`
- **Models:** Mount to `/app/models`
- **Config:** Mount to `/app/config`

## 🚨 Troubleshooting

### Common Issues

**"Docker is not running"**
```bash
# macOS: Start Docker Desktop
open -a Docker

# Linux: Start Docker daemon
sudo systemctl start docker
```

**"Permission denied"**
```bash
# Make script executable
chmod +x docker/run-demo.sh
```

**"Container build fails"**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

**"Demo data missing"**
```bash
# Ensure demo data exists
ls -la demo/data/
```

### Getting Help

1. **Check logs:**
   ```bash
   docker-compose logs beam-demo
   ```

2. **Interactive debugging:**
   ```bash
   ./docker/run-demo.sh interactive
   ```

3. **Report issues:** Include Docker version, OS, and error logs

## 🎓 Learning Path

### 1. Start with Demo
- Run `./docker/run-demo.sh`
- Understand the supply chain compromise scenario
- Review the generated security reports

### 2. Explore Interactively
- Use `./docker/run-demo.sh interactive`
- Examine the code structure
- Run individual BEAM components

### 3. Analyze Your Data
- Place your HAR/PCAP files in `data/input/`
- Use the core container for analysis
- Train custom models for your applications

### 4. Production Deployment
- Use multi-container setup with `docker-compose.yml`
- Configure persistent volumes
- Set up monitoring and logging

## 📚 Next Steps

After running the demo:

1. **Read the main README.md** for detailed BEAM documentation
2. **Check out `models/custom_models/`** to learn about training
3. **Explore `src/beam/`** to understand the codebase
4. **Try your own data** with the core container

## 🤝 Contributing

When contributing Docker improvements:

1. Test all container variants
2. Ensure demo still works seamlessly
3. Update this README with changes
4. Consider impact on user experience

---

**🎯 Goal:** Make BEAM accessible to anyone with Docker installed, regardless of their Python/ML experience level.
