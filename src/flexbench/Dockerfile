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
COPY . .

# Create virtual environment and install the flexbench module with full dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && \
    uv sync --no-dev

# Add venv to PATH
ENV PATH=${VIRTUAL_ENV}/bin:$PATH

# Health check for FlexBench module readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import flexbench; print('FlexBench module ready')" || exit 1

# Default command - run the flexbench module
CMD ["python", "-m", "flexbench"]
