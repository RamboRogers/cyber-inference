# Cyber-Inference Dockerfile
# Multi-stage build for efficient image size

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv pip install .

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.12-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash cyber

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Create directories
RUN mkdir -p /app/data /app/models /app/bin /app/data/logs \
    && chown -R cyber:cyber /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CYBER_INFERENCE_DATA_DIR=/app/data \
    CYBER_INFERENCE_MODELS_DIR=/app/models \
    CYBER_INFERENCE_BIN_DIR=/app/bin \
    CYBER_INFERENCE_HOST=0.0.0.0 \
    CYBER_INFERENCE_PORT=8337 \
    CYBER_INFERENCE_LOG_LEVEL=INFO

# Expose port
EXPOSE 8337

# Switch to non-root user
USER cyber

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8337/health || exit 1

# Volume mounts for persistence
VOLUME ["/app/data", "/app/models", "/app/bin"]

# Default command
CMD ["python", "-m", "cyber_inference.cli", "serve", "--host", "0.0.0.0", "--port", "8337"]

