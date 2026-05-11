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
```

## Docker
The compose file is self-contained for local ML development. It starts a local
RabbitMQ broker for `diabetify-ml-worker`, so the worker can resolve
`rabbitmq:5672` inside the compose network.

```powershell
docker compose up --build diabetify-ml
```

RabbitMQ management UI is available at `http://localhost:15673` with
`admin` / `password123` by default.

If the backend compose is already running its own RabbitMQ on the same host
ports, stop one of the brokers before using this standalone compose file. For
full-stack runs, use a single shared broker from the backend or a top-level
compose file instead of starting two RabbitMQ containers.
