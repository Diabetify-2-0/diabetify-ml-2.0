import json
import unittest
from unittest.mock import MagicMock, patch

from prediction_core import PredictionService


class DummyModel:
    def predict_proba(self, x_frame):
        return [[0.25, 0.75]]


class DummyExplainer:
    def shap_values(self, x_frame):
        return [[[0.1, -0.2]]]


class PredictionServiceTests(unittest.TestCase):
    def test_predict_rejects_non_numeric_features(self) -> None:
        service = PredictionService(model_dir="missing-model-dir")
        with service._lock:
            service.model = DummyModel()
            service.explainer = DummyExplainer()
            service.x_columns = ["age", "bmi"]

        with self.assertRaisesRegex(ValueError, "must be numeric"):
            service.predict(["bad", 24.0])

    def test_predict_rejects_non_finite_features(self) -> None:
        service = PredictionService(model_dir="missing-model-dir")
        with service._lock:
            service.model = DummyModel()
            service.explainer = DummyExplainer()
            service.x_columns = ["age", "bmi"]

        with self.assertRaisesRegex(ValueError, "must be finite"):
            service.predict([float("nan"), 24.0])

    def test_reload_updates_state_and_clears_error(self) -> None:
        mocked_file = MagicMock()
        mocked_file.__enter__.return_value = mocked_file
        with (
            patch("builtins.open", return_value=mocked_file),
            patch("prediction_core.pickle.load", side_effect=[DummyModel(), ["age", "bmi"]]),
            patch("prediction_core.shap.TreeExplainer", return_value=DummyExplainer()),
        ):
            service = PredictionService(model_dir="mocked-model-dir")

        health = service.health()
        self.assertEqual(health["status"], "healthy")
        self.assertIsNone(health["last_reload_error"])
        self.assertIsNotNone(health["last_reload_at"])

    def test_predict_returns_response_shape(self) -> None:
        service = PredictionService(model_dir="missing-model-dir")
        with service._lock:
            service.model = DummyModel()
            service.explainer = DummyExplainer()
            service.x_columns = ["age", "bmi"]

        result = service.predict([30.0, 24.0])
        self.assertEqual(result["prediction"], 0.75)
        self.assertEqual(set(result["explanation"].keys()), {"age", "bmi"})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
