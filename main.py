import logging
import os
import pickle

import numpy as np
import pandas as pd
import shap
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", ".")


def _load_artifacts(model_dir: str):
    with open(os.path.join(model_dir, "xg_model.pkl"), "rb") as f:
        _model = pickle.load(f)
    with open(os.path.join(model_dir, "x_columns.pkl"), "rb") as f:
        _x_columns = pickle.load(f)
    _background = pd.read_parquet(os.path.join(model_dir, "shap_background.parquet"))
    _explainer = shap.TreeExplainer(_model, feature_perturbation="tree_path_dependent")
    return _model, _x_columns, _background, _explainer


model, x_columns, background, explainer = _load_artifacts(MODEL_DIR)

app = FastAPI()

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(req: PredictRequest):
    start_time = time.time()

    X = pd.DataFrame([req.features], columns=x_columns)
    prediction = model.predict_proba(X)[0]
    shap_values_single = explainer.shap_values(X)
    abs_shap_single = np.abs(shap_values_single[0])
    abs_shap_single /= abs_shap_single.sum()

    explanation_items = [
        (
            feature,
            {
                "shap_value": float(shap),
                "contribution": float(contribution),
                "impact": 1 if shap > 0 else 0
            }
        )
        for feature, shap, contribution in zip(x_columns, shap_values_single[0], abs_shap_single)
    ]
    explanation_items_sorted = sorted(explanation_items, key=lambda x: x[1]["contribution"], reverse=True)
    explanation_dict_sorted = dict(explanation_items_sorted)

    elapsed_time = time.time() - start_time

    return {
        "prediction": float(prediction[1]),
        "explanation": explanation_dict_sorted,
        "elapsed_time_seconds": round(elapsed_time, 4)
    }


@app.post("/reload")
def reload_model():
    """Hot-reload model artifacts from disk (called by MLOps after promotion)."""
    global model, x_columns, background, explainer
    try:
        model, x_columns, background, explainer = _load_artifacts(MODEL_DIR)
        logger.info("Model reloaded successfully from %s", MODEL_DIR)
        return {"status": "ok", "message": "Model reloaded"}
    except Exception as e:
        logger.error("Model reload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
