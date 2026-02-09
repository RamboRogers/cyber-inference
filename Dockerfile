# Cyber-Inference Dockerfile (CPU)
#
# Uses start.sh to handle dependency installation, environment setup,
# and server startup. llama.cpp is auto-installed by the application.
#
# Build:
#   docker build -t cyber-inference .
#
# Run:
#   docker run -d --name cyber-inference -p 8337:8337 \
#     -v cyber-models:/app/models -v cyber-data:/app/data \
#     cyber-inference

FROM python:3.12-slim

# Install runtime + build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    libcurl4-openssl-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pre-installed so start.sh skips this step)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md start.sh .python-version ./
COPY src/ ./src/

# Make start.sh executable
RUN chmod +x start.sh

# Pre-sync base dependencies during build for faster container startup
RUN uv sync

# Create data directories
RUN mkdir -p /app/data /app/models /app/bin /app/data/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CYBER_INFERENCE_DATA_DIR=/app/data \
    CYBER_INFERENCE_MODELS_DIR=/app/models \
    CYBER_INFERENCE_BIN_DIR=/app/bin \
    CYBER_INFERENCE_HOST=0.0.0.0 \
    CYBER_INFERENCE_PORT=8337 \
    CYBER_INFERENCE_LOG_LEVEL=INFO

EXPOSE 8337

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8337/health || exit 1

# Volumes for persistence
VOLUME ["/app/data", "/app/models"]

# start.sh handles uv sync, llama.cpp auto-install, and server launch
CMD ["./start.sh"]
