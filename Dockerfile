FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
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

# Copy the flexbench module for installation
COPY src/ ./src/

# Create virtual environment and install the flexbench module
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && \
    uv pip install -e ./src/flexbench/

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set FlexBench module environment variables
ENV LOG_LEVEL=DEBUG

# Health check for FlexBench module readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import flexbench; print('FlexBench module ready')" || exit 1

# Default command - run the flexbench module
CMD ["python", "-m", "flexbench"]
