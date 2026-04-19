# diabetify-ml

ML prediction service for Diabetify.

## Runtime
The primary runtime is the RabbitMQ worker in `main_mq.py`.
It consumes `ml.prediction.request` and publishes responses to the queue requested by
the backend, normally `ml.prediction.hybrid_response`.

`main.py` is kept as an optional FastAPI API for local/manual prediction, health checks,
and model reload.

## Local Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run RabbitMQ Worker
```powershell
$env:RABBITMQ_URL = "amqp://admin:password123@localhost:5672/"
python main_mq.py
```

## Run Optional REST API
```powershell
uvicorn main:app --reload
```

## Quality Checks
```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m py_compile prediction_core.py main.py main_mq.py
```

## Docker
```powershell
docker compose up --build diabetify-ml-worker
docker compose --profile api up --build
```
