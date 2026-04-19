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

# Optional REST API port when running `uvicorn main:app`.
EXPOSE 5000

ENV MODEL_DIR=/app

# Default runtime matches diabetify-be integration via RabbitMQ.
CMD ["python", "main_mq.py"]
