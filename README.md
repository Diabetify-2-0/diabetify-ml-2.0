# diabetify-ml

ML prediction service for Diabetify.

## Runtime
The primary runtime is the RabbitMQ worker in `main_mq.py`.
It consumes `ml.prediction.request` and publishes responses to the queue requested by
the backend via `reply_to`, normally `ml.prediction.response`.
The legacy queue `ml.prediction.hybrid_response` is still declared for compatibility.

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
python -m py_compile prediction_core.py main.py main_mq.py main_combined.py shared.py
python -m unittest discover -s tests
```

## Docker
The compose file reuses RabbitMQ from `diabetify-be` through host port `5672`.
It does not start its own broker.

```powershell
docker compose up --build -d diabetify-ml
```

The container also exposes the optional REST API on `http://localhost:5000`.
RabbitMQ management UI still comes from the backend stack at
`http://localhost:25672` with `admin` / `password123` by default.

`GET /health` now returns HTTP `503` whenever the model is not loaded or,
for the combined runtime, when the RabbitMQ worker is disconnected.

For full-stack local runs, start `diabetify-be` first so RabbitMQ is already
available on the host before the ML container boots.
