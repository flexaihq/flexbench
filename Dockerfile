FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install FlexBench
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=INFO

# Run FlexBench
CMD ["python", "-m", "flexbench"]
