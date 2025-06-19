FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Set uv environment variables for production
ENV UV_COMPILE_BYTECODE=1 \
    UV_FROZEN=1 \
    UV_LINK_MODE=copy \
    UV_NO_INSTALLER_METADATA=1 \
    VIRTUAL_ENV=/app/.venv \
    PYTHONUNBUFFERED=1

# Install dependencies (excluding the project itself and dev deps)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY src/ ./src/

# Install the project itself
RUN uv pip install --no-deps -e .

# Set FlexBench environment variables
ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=INFO

# Health check for FlexBench readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import flexbench; print('FlexBench ready')" || exit 1

# Default command
CMD ["python", "-m", "flexbench"]
