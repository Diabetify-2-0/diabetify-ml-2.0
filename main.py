import logging
import os
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from shared import prediction_service, runtime_status

logger = logging.getLogger(__name__)

_RELOAD_SECRET = os.getenv("RELOAD_SECRET", "")
_MLOPS_URL = os.getenv("MLOPS_URL", "")

_ARTIFACT_FILES = ("xg_model.pkl", "x_columns.pkl", "shap_background.parquet")


def _download_model_from_mlops(model_id: int) -> None:
    model_dir = prediction_service.model_dir
    os.makedirs(model_dir, exist_ok=True)
    with httpx.Client(timeout=30.0) as client:
        for fname in _ARTIFACT_FILES:
            url = f"{_MLOPS_URL}/internal/models/{model_id}/artifact/{fname}"
            resp = client.get(url, headers={"X-Reload-Secret": _RELOAD_SECRET})
            if resp.status_code == 404 and fname == "shap_background.parquet":
                logger.warning("shap_background.parquet not found for model %d — skipping", model_id)
                continue
            resp.raise_for_status()
            dest = os.path.join(model_dir, fname)
            with open(dest, "wb") as f:
                f.write(resp.content)
            logger.info("Downloaded %s from MLOps (model %d)", fname, model_id)
    logger.info("All model artifacts downloaded to %s", model_dir)

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


class ReloadRequest(BaseModel):
    model_id: Optional[int] = None


@app.post("/reload")
def reload_model(
    body: ReloadRequest = ReloadRequest(),
    x_reload_secret: str = Header(default=""),
) -> dict:
    if not _RELOAD_SECRET or x_reload_secret != _RELOAD_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        if _MLOPS_URL and body.model_id is not None:
            logger.info("Downloading model %d artifacts from MLOps at %s", body.model_id, _MLOPS_URL)
            _download_model_from_mlops(body.model_id)
        prediction_service.reload(model_id=body.model_id)
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
