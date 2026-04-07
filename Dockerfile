# Use the official OpenEnv base image
FROM ghcr.io/meta-pytorch/openenv-base:latest

# Set working directory
WORKDIR /app

# Install uv for fast dependency management
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements to use Docker cache
COPY requirements.txt .

# Install dependencies directly into the container's environment
# This is much more robust than multi-stage copying for troubleshooting
RUN uv pip install --system --no-cache -r requirements.txt

# Verify installation (Look for 'groq' in your Docker build logs!)
RUN python -c "import groq; print('Groq successfully installed')"

# Copy the rest of your source code
# (.dockerignore ensures .git and .myenv aren't copied)
COPY . .

# Set environment variables
ENV PYTHONPATH="/app"
ENV ENABLE_WEB_INTERFACE=true

# Expose port and healthcheck
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch the FastAPI server with proxy support
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips='*'"]
