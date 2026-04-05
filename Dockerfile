# Use the official OpenEnv base image
FROM ghcr.io/meta-pytorch/openenv-base:latest

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies (from root requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Health check to ensure the model service is responsive
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch the FastAPI server using the modular entry point
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
