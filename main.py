import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prediction_core import DEFAULT_MODEL_DIR, PredictionService

logger = logging.getLogger(__name__)

prediction_service = PredictionService(DEFAULT_MODEL_DIR)
app = FastAPI()


class PredictRequest(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        return prediction_service.predict(req.features)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {err}") from err


@app.post("/reload")
def reload_model() -> dict:
    try:
        prediction_service.reload()
        logger.info("Model reloaded successfully from %s", DEFAULT_MODEL_DIR)
        return {"status": "ok", "message": "Model reloaded"}
    except Exception as err:
        logger.exception("Model reload failed")
        raise HTTPException(status_code=500, detail=f"Reload failed: {err}") from err


@app.get("/health")
def health() -> dict:
    return prediction_service.health()
