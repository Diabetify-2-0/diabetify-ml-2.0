"""
Entry point that runs both the FastAPI REST server and the RabbitMQ worker
in the same process so they share one PredictionService instance.
Calling POST /reload reloads the model for all prediction paths.
"""
import logging
import signal
import threading

import uvicorn

from main import app
from main_mq import AsyncMLService
from shared import runtime_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_rest() -> None:
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")


def main() -> None:
    runtime_status.set_require_rabbitmq(True)
    mq_service = AsyncMLService()

    def handle_signal(signum, frame):
        mq_service.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    rest_thread = threading.Thread(target=run_rest, name="RestAPI", daemon=True)
    rest_thread.start()
    logger.info("REST API thread started on port 5000")

    mq_service.start()


if __name__ == "__main__":
    main()
