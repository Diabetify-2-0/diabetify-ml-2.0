import json
import unittest

import main
from shared import prediction_service, runtime_status


class HealthEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_runtime = runtime_status.snapshot()
        with prediction_service._lock:
            self.original_model = prediction_service.model
            self.original_explainer = prediction_service.explainer
            self.original_x_columns = list(prediction_service.x_columns)
            self.original_last_reload_at = prediction_service.last_reload_at
            self.original_last_reload_error = prediction_service.last_reload_error

    def tearDown(self) -> None:
        runtime_status.set_require_rabbitmq(self.original_runtime["require_rabbitmq"])
        if self.original_runtime["rabbitmq_initialized"]:
            runtime_status.update_rabbitmq(
                self.original_runtime["rabbitmq_healthy"],
                self.original_runtime["rabbitmq_details"],
            )
        else:
            runtime_status.reset_rabbitmq()

        with prediction_service._lock:
            prediction_service.model = self.original_model
            prediction_service.explainer = self.original_explainer
            prediction_service.x_columns = self.original_x_columns
            prediction_service.last_reload_at = self.original_last_reload_at
            prediction_service.last_reload_error = self.original_last_reload_error

    def test_health_returns_503_when_model_not_loaded(self) -> None:
        runtime_status.set_require_rabbitmq(False)
        with prediction_service._lock:
            prediction_service.model = None
            prediction_service.explainer = None
            prediction_service.x_columns = []
            prediction_service.last_reload_error = "missing model"

        response = main.health()
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["status"], "unhealthy")

    def test_health_returns_503_when_rabbitmq_required_but_unhealthy(self) -> None:
        runtime_status.set_require_rabbitmq(True)
        runtime_status.update_rabbitmq(False, {"note": "connection lost"})
        with prediction_service._lock:
            prediction_service.model = object()
            prediction_service.explainer = object()
            prediction_service.x_columns = ["age"]
            prediction_service.last_reload_error = None

        response = main.health()
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["components"]["rabbitmq"]["status"], "unhealthy")

    def test_health_returns_200_when_required_components_are_healthy(self) -> None:
        runtime_status.set_require_rabbitmq(True)
        runtime_status.update_rabbitmq(True, {"note": "connected"})
        with prediction_service._lock:
            prediction_service.model = object()
            prediction_service.explainer = object()
            prediction_service.x_columns = ["age"]
            prediction_service.last_reload_error = None

        response = main.health()
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(payload["components"]["rabbitmq"]["status"], "healthy")


if __name__ == "__main__":
    unittest.main()
