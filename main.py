import logging
import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from shared import prediction_service, runtime_status

logger = logging.getLogger(__name__)

_RELOAD_SECRET = os.getenv("RELOAD_SECRET", "")

app = FastAPI()


@app.on_event("startup")
def on_startup() -> None:
    if not _RELOAD_SECRET:
        logger.warning("RELOAD_SECRET is not set — /reload endpoint will reject all calls")


class PredictRequest(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        return prediction_service.predict(req.features)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except RuntimeError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except Exception as err:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {err}") from err


@app.post("/reload")
def reload_model(x_reload_secret: str = Header(default="")) -> dict:
    if not _RELOAD_SECRET or x_reload_secret != _RELOAD_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        prediction_service.reload()
        logger.info("Model reloaded successfully from %s", prediction_service.model_dir)
        return {"status": "ok", "message": "Model reloaded"}
    except Exception as err:
        logger.exception("Model reload failed")
        raise HTTPException(status_code=500, detail=f"Reload failed: {err}") from err


@app.get("/health")
def health() -> JSONResponse:
    health_payload = prediction_service.health()
    runtime_snapshot = runtime_status.snapshot()

    components = {}
    rabbitmq_required = runtime_snapshot["require_rabbitmq"]
    rabbitmq_initialized = runtime_snapshot["rabbitmq_initialized"]
    if rabbitmq_required or rabbitmq_initialized:
        components["rabbitmq"] = {
            "status": "healthy" if runtime_snapshot["rabbitmq_healthy"] else "unhealthy",
            **runtime_snapshot["rabbitmq_details"],
        }

    overall_healthy = health_payload["status"] == "healthy"
    if rabbitmq_required:
        overall_healthy = overall_healthy and runtime_snapshot["rabbitmq_healthy"]

    health_payload["status"] = "healthy" if overall_healthy else "unhealthy"
    health_payload["components"] = components
    return JSONResponse(
        status_code=200 if overall_healthy else 503,
        content=health_payload,
    )
