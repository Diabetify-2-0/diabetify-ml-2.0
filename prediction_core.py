import datetime
import math
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap


DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", ".")


@dataclass
class PredictionService:
    model_dir: str = DEFAULT_MODEL_DIR

    def __post_init__(self) -> None:
        self.model: Any | None = None
        self.x_columns: list[str] = []
        self.explainer: Any | None = None
        try:
            self.reload()
        except FileNotFoundError:
            import logging
            logging.getLogger(__name__).warning(
                "Model files not found at %s — service unhealthy until /reload is called",
                self.model_dir,
            )

    def reload(self) -> None:
        with open(os.path.join(self.model_dir, "xg_model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(self.model_dir, "x_columns.pkl"), "rb") as f:
            self.x_columns = list(pickle.load(f))
        self.explainer = shap.TreeExplainer(self.model)

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy" if self.model is not None and self.explainer is not None else "unhealthy",
            "model_loaded": self.model is not None,
            "explainer_ready": self.explainer is not None,
            "feature_count": len(self.x_columns),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def predict(self, features: list[float]) -> dict[str, Any]:
        self._validate_features(features)
        assert self.model is not None
        assert self.explainer is not None

        start_time = time.time()
        converted_features = [self._safe_float(value) for value in features]
        x_frame = pd.DataFrame([converted_features], columns=self.x_columns)
        prediction = self.model.predict_proba(x_frame)[0]

        shap_values = self.explainer.shap_values(x_frame)
        shap_row = np.asarray(shap_values[0])
        abs_shap = np.abs(shap_row)
        total_shap = float(abs_shap.sum())
        if total_shap > 0:
            contributions = abs_shap / total_shap
        else:
            contributions = np.zeros_like(abs_shap)

        explanation = {}
        for index, (feature, shap_value, contribution) in enumerate(
            zip(self.x_columns, shap_row, contributions)
        ):
            safe_shap = self._safe_float(shap_value)
            safe_contribution = self._safe_float(contribution)
            explanation[str(feature)] = {
                "shap": safe_shap,
                "contribution": safe_contribution,
                "impact": 1 if safe_shap > 0 else 0,
                "value": converted_features[index],
            }

        elapsed_time = time.time() - start_time
        return {
            "prediction": self._safe_float(prediction[1]),
            "explanation": explanation,
            "elapsed_time": self._safe_float(elapsed_time),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def _validate_features(self, features: list[float]) -> None:
        expected = len(self.x_columns)
        if len(features) != expected:
            raise ValueError(f"Feature mismatch: expected {expected} features, received {len(features)}.")

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(converted) or math.isinf(converted):
            return 0.0
        return converted
