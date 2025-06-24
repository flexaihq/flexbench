FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set uv environment variables for production
ENV UV_COMPILE_BYTECODE=1 \
    UV_FROZEN=1 \
    UV_LINK_MODE=copy \
    UV_NO_INSTALLER_METADATA=1 \
    VIRTUAL_ENV=/app/.venv \
    PYTHONUNBUFFERED=1

# Copy only the flexbench core package (CLI is separate and not needed in Docker)
COPY src/flexbench/ ./src/flexbench/

# Create virtual environment and install only the flexbench core package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && \
    cd src/flexbench && uv pip install -e .

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set FlexBench environment variables
ENV LOG_LEVEL=DEBUG

# Health check for FlexBench readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import flexbench; print('FlexBench ready')" || exit 1

# Default command - use python -m flexbench (core package entry point)
CMD ["python", "-m", "flexbench"]
