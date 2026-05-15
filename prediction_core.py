import datetime
import logging
import math
import os
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap


DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", ".")
logger = logging.getLogger(__name__)


@dataclass
class PredictionService:
    model_dir: str = DEFAULT_MODEL_DIR

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self.model: Any | None = None
        self.x_columns: list[str] = []
        self.explainer: Any | None = None
        self.last_reload_at: str | None = None
        self.last_reload_error: str | None = None
        try:
            self.reload()
        except FileNotFoundError:
            self.last_reload_error = f"Model files not found at {self.model_dir}"
            logger.warning(
                "Model files not found at %s — service unhealthy until /reload is called",
                self.model_dir,
            )
        except Exception as err:
            self.last_reload_error = str(err)
            logger.exception(
                "Initial model load failed from %s — service unhealthy until /reload succeeds",
                self.model_dir,
            )

    def reload(self) -> None:
        model_path = os.path.join(self.model_dir, "xg_model.pkl")
        columns_path = os.path.join(self.model_dir, "x_columns.pkl")
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(columns_path, "rb") as f:
                x_columns = list(pickle.load(f))
            if not x_columns:
                raise ValueError("x_columns.pkl must contain at least one feature name")
            explainer = shap.TreeExplainer(model)
        except Exception as err:
            self.last_reload_error = str(err)
            raise

        with self._lock:
            self.model = model
            self.x_columns = x_columns
            self.explainer = explainer
            self.last_reload_at = datetime.datetime.now().isoformat()
            self.last_reload_error = None

    def health(self) -> dict[str, Any]:
        with self._lock:
            model_loaded = self.model is not None
            explainer_ready = self.explainer is not None
            feature_count = len(self.x_columns)
            last_reload_at = self.last_reload_at
            last_reload_error = self.last_reload_error
            ready = model_loaded and explainer_ready and feature_count > 0

        return {
            "status": "healthy" if ready else "unhealthy",
            "model_loaded": model_loaded,
            "explainer_ready": explainer_ready,
            "feature_count": feature_count,
            "last_reload_at": last_reload_at,
            "last_reload_error": last_reload_error,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def predict(self, features: list[float]) -> dict[str, Any]:
        model, explainer, x_columns = self._snapshot_ready_state()
        converted_features = self._validate_features(features, x_columns)

        start_time = time.perf_counter()
        x_frame = pd.DataFrame([converted_features], columns=x_columns)
        prediction = model.predict_proba(x_frame)[0]

        shap_values = explainer.shap_values(x_frame)
        shap_row = self._extract_shap_row(shap_values)
        abs_shap = np.abs(shap_row)
        total_shap = float(abs_shap.sum())
        if total_shap > 0:
            contributions = abs_shap / total_shap
        else:
            contributions = np.zeros_like(abs_shap)

        explanation = {}
        for index, (feature, shap_value, contribution) in enumerate(
            zip(x_columns, shap_row, contributions)
        ):
            safe_shap = self._safe_float_for_output(shap_value)
            safe_contribution = self._safe_float_for_output(contribution)
            explanation[str(feature)] = {
                "shap": safe_shap,
                "contribution": safe_contribution,
                "impact": 1 if safe_shap > 0 else 0,
                "value": converted_features[index],
            }

        elapsed_time = time.perf_counter() - start_time
        positive_class_probability = prediction[1] if len(prediction) > 1 else prediction[0]
        return {
            "prediction": self._safe_float_for_output(positive_class_probability),
            "explanation": explanation,
            "elapsed_time": self._safe_float_for_output(elapsed_time),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def _snapshot_ready_state(self) -> tuple[Any, Any, list[str]]:
        with self._lock:
            if self.model is None or self.explainer is None or not self.x_columns:
                raise RuntimeError("Model is not loaded")
            return self.model, self.explainer, list(self.x_columns)

    def _validate_features(self, features: list[float], x_columns: list[str]) -> list[float]:
        expected = len(x_columns)
        if len(features) != expected:
            raise ValueError(f"Feature mismatch: expected {expected} features, received {len(features)}.")
        return [self._coerce_feature(value, index, x_columns[index]) for index, value in enumerate(features)]

    @staticmethod
    def _coerce_feature(value: Any, index: int, feature_name: str) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature '{feature_name}' at index {index} must be numeric, received {value!r}."
            ) from None
        if math.isnan(converted) or math.isinf(converted):
            raise ValueError(
                f"Feature '{feature_name}' at index {index} must be finite, received {value!r}."
            )
        return converted

    @staticmethod
    def _extract_shap_row(shap_values: Any) -> np.ndarray:
        if isinstance(shap_values, list):
            if not shap_values:
                raise RuntimeError("SHAP returned an empty list")
            shap_row = np.asarray(shap_values[-1][0])
        else:
            shap_array = np.asarray(shap_values)
            if shap_array.ndim == 3:
                shap_row = np.asarray(shap_array[0, :, -1])
            elif shap_array.ndim == 2:
                shap_row = np.asarray(shap_array[0])
            else:
                raise RuntimeError(f"Unexpected SHAP output shape: {shap_array.shape}")
        return shap_row.astype(float, copy=False)

    @staticmethod
    def _safe_float_for_output(value: Any) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(converted) or math.isinf(converted):
            return 0.0
        return converted
