FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose ports
# 5000: FastAPI REST API
# 50051: gRPC server
EXPOSE 5000 50051

ENV MODEL_DIR=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/docs')" || exit 1

# Default to FastAPI server, can be overridden with docker-compose command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
