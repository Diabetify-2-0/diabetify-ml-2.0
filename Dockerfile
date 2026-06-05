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

# The bundled image includes xg_model.pkl and x_columns.pkl in /app.
# When running via docker-compose, MODEL_DIR is overridden to /app/models (volume).
# FALLBACK_MODEL_DIR points to the bundled location so the service stays healthy
# on a fresh deployment before any model is approved via MLOps.
ENV MODEL_DIR=/app
ENV FALLBACK_MODEL_DIR=/app

# Runs REST API (port 5000) + RabbitMQ worker in one process sharing one PredictionService
CMD ["python", "main_combined.py"]
