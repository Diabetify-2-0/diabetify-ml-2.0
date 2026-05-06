FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# REST API port
EXPOSE 5000

# MODEL_DIR is overridden at runtime to point at the shared volume
ENV MODEL_DIR=/app/models

# Runs REST API (port 5000) + RabbitMQ worker in one process sharing one PredictionService
CMD ["python", "main_combined.py"]
