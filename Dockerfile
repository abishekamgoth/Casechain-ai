# ============================================================
# Dockerfile for Legal RAG Project
# ============================================================
#
# WHAT IS DOCKER?
# Docker packages your application + all its dependencies into a "container".
# A container is like a lightweight virtual machine that:
# - Has Python installed exactly the version you need
# - Has all your packages installed
# - Runs the same way on your laptop, your friend's laptop, and AWS
#
# Without Docker: "Works on my machine" problem
# With Docker: Same environment everywhere
#
# HOW TO USE:
#   docker build -t legal-rag .         # Build the image
#   docker run -p 8000:8000 legal-rag   # Run the container
# ============================================================

# Start from an official Python 3.10 image (slim = smaller size)
FROM python:3.10-slim

# Set working directory inside the container
# All subsequent commands run from this directory
WORKDIR /app

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Don't create .pyc files (keeps container clean)
# PYTHONUNBUFFERED: Print logs immediately (important for Docker log viewing)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# Some Python packages need C libraries to compile (like FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*   # Clean up to reduce image size

# Copy ONLY requirements first (Docker caching optimization)
# If requirements.txt doesn't change, Docker reuses the cached layer
# This makes rebuilds much faster during development
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Don't cache pip downloads (reduces image size)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This is done AFTER installing dependencies for caching efficiency
COPY . .

# Create directories that the app expects to exist
RUN mkdir -p data/faiss_index data/citation_graph mlflow_runs

# Expose the port FastAPI runs on
# Note: EXPOSE is documentation only — it doesn't actually open the port
# You need -p 8000:8000 in the docker run command for that
EXPOSE 8000

# Health check: Docker will run this periodically to verify the container is healthy
# If /health returns non-200, Docker marks the container as unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run when the container starts
# uvicorn = the ASGI server that runs FastAPI
# --host 0.0.0.0 = accept connections from outside the container
# --port 8000 = listen on port 8000
# --workers 2 = run 2 parallel workers (handle 2 requests simultaneously)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
