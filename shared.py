import threading
from dataclasses import dataclass, field
from typing import Any

from prediction_core import DEFAULT_MODEL_DIR, PredictionService


@dataclass
class RuntimeStatus:
    require_rabbitmq: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _rabbitmq_initialized: bool = field(default=False, init=False, repr=False)
    _rabbitmq_healthy: bool = field(default=False, init=False, repr=False)
    _rabbitmq_details: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def set_require_rabbitmq(self, required: bool) -> None:
        with self._lock:
            self.require_rabbitmq = required

    def update_rabbitmq(self, healthy: bool, details: dict[str, Any]) -> None:
        with self._lock:
            self._rabbitmq_initialized = True
            self._rabbitmq_healthy = healthy
            self._rabbitmq_details = dict(details)

    def reset_rabbitmq(self) -> None:
        with self._lock:
            self._rabbitmq_initialized = False
            self._rabbitmq_healthy = False
            self._rabbitmq_details = {}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "require_rabbitmq": self.require_rabbitmq,
                "rabbitmq_initialized": self._rabbitmq_initialized,
                "rabbitmq_healthy": self._rabbitmq_healthy,
                "rabbitmq_details": dict(self._rabbitmq_details),
            }


prediction_service = PredictionService(DEFAULT_MODEL_DIR)
runtime_status = RuntimeStatus()
