from __future__ import annotations

import datetime
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import pika
from pika.exceptions import AMQPConnectionError

from shared import prediction_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://admin:password123@localhost:5672/")
MAX_RABBITMQ_RETRIES = int(os.getenv("MAX_RABBITMQ_RETRIES", "5"))
RABBITMQ_RETRY_DELAY = int(os.getenv("RABBITMQ_RETRY_DELAY", "5"))
PREDICTION_REQUEST_QUEUE = os.getenv("PREDICTION_REQUEST_QUEUE", "ml.prediction.request")
PREDICTION_RESPONSE_QUEUE = os.getenv("PREDICTION_RESPONSE_QUEUE", "ml.prediction.response")
LEGACY_PREDICTION_RESPONSE_QUEUE = os.getenv(
    "LEGACY_PREDICTION_RESPONSE_QUEUE",
    "ml.prediction.hybrid_response",
)
HEALTH_REQUEST_QUEUE = os.getenv("HEALTH_REQUEST_QUEUE", "ml.health.request")
HEALTH_RESPONSE_QUEUE = os.getenv("HEALTH_RESPONSE_QUEUE", "ml.health.response")


class AsyncMLService:
    def __init__(self) -> None:
        self.prediction_service = prediction_service
        self.is_running = threading.Event()

        self.rabbit_connection: pika.BlockingConnection | None = None
        self.rabbit_channel: pika.channel.Channel | None = None
        self.rabbit_consumer_thread: threading.Thread | None = None

        self.prediction_request_queue = PREDICTION_REQUEST_QUEUE
        self.health_request_queue = HEALTH_REQUEST_QUEUE
        self.prediction_response_queue = PREDICTION_RESPONSE_QUEUE
        self.legacy_prediction_response_queue = LEGACY_PREDICTION_RESPONSE_QUEUE
        self.health_response_queue = HEALTH_RESPONSE_QUEUE

        self.messages_received = 0
        self.messages_processed = 0
        self.messages_failed = 0

    def start(self) -> None:
        self.is_running.set()
        if not self._setup_rabbitmq():
            logger.error("RabbitMQ setup failed after %s attempts", MAX_RABBITMQ_RETRIES)
            sys.exit(1)

        self.rabbit_consumer_thread = threading.Thread(
            target=self._start_consuming,
            name="RabbitMQConsumer",
            daemon=True,
        )
        self.rabbit_consumer_thread.start()
        logger.info("ML RabbitMQ worker started")

        try:
            while self.is_running.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        if not self.is_running.is_set():
            return

        self.is_running.clear()
        self._stop_consuming_threadsafe()
        if self.rabbit_consumer_thread and self.rabbit_consumer_thread.is_alive():
            self.rabbit_consumer_thread.join(timeout=5)
        self._close_rabbitmq_resources()
        logger.info("ML RabbitMQ worker stopped")

    def _setup_rabbitmq(self) -> bool:
        for attempt in range(1, MAX_RABBITMQ_RETRIES + 1):
            try:
                self.rabbit_connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
                self.rabbit_channel = self.rabbit_connection.channel()
                for queue in self._queues():
                    self.rabbit_channel.queue_declare(queue=queue, durable=True)
                self.rabbit_channel.basic_qos(prefetch_count=1)
                return True
            except AMQPConnectionError as err:
                logger.warning(
                    "RabbitMQ connection attempt %s/%s failed: %s",
                    attempt,
                    MAX_RABBITMQ_RETRIES,
                    err,
                )
                time.sleep(RABBITMQ_RETRY_DELAY)
        return False

    def _queues(self) -> list[str]:
        queues = [
            self.prediction_request_queue,
            self.prediction_response_queue,
            self.health_request_queue,
            self.health_response_queue,
        ]
        if self.legacy_prediction_response_queue != self.prediction_response_queue:
            queues.append(self.legacy_prediction_response_queue)
        return queues

    def _start_consuming(self) -> None:
        assert self.rabbit_channel is not None
        self.rabbit_channel.basic_consume(
            queue=self.prediction_request_queue,
            on_message_callback=lambda ch, method, props, body: self._handle_message(
                ch,
                method,
                props,
                body,
                self._process_prediction_request,
                self.prediction_response_queue,
            ),
        )
        self.rabbit_channel.basic_consume(
            queue=self.health_request_queue,
            on_message_callback=lambda ch, method, props, body: self._handle_message(
                ch,
                method,
                props,
                body,
                self._process_health_request,
                self.health_response_queue,
            ),
        )

        try:
            self.rabbit_channel.start_consuming()
        except Exception as err:
            if self.is_running.is_set():
                logger.exception("RabbitMQ consumer stopped unexpectedly: %s", err)
        finally:
            self._close_rabbitmq_resources()

    def _handle_message(
        self,
        ch: pika.channel.Channel,
        method: Any,
        properties: pika.BasicProperties,
        body: bytes,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
        fallback_reply_to_queue: str,
    ) -> None:
        correlation_id = properties.correlation_id
        reply_to_queue = properties.reply_to or fallback_reply_to_queue
        self.messages_received += 1

        try:
            request_data = json.loads(body.decode("utf-8"))
            if request_data.get("correlation_id"):
                correlation_id = str(request_data["correlation_id"])

            response_data = handler(request_data)
            response_data["correlation_id"] = correlation_id
            self._publish(ch, reply_to_queue, correlation_id, response_data)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.messages_processed += 1
        except Exception as err:
            self.messages_failed += 1
            error_response = {
                "error": str(err),
                "correlation_id": correlation_id,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            try:
                self._publish(ch, reply_to_queue, correlation_id, error_response)
            except Exception:
                logger.exception("Failed to publish ML error response")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def _stop_consuming_threadsafe(self) -> None:
        if not self.rabbit_connection or not self.rabbit_connection.is_open:
            return
        if not self.rabbit_channel or not self.rabbit_channel.is_open:
            return

        try:
            self.rabbit_connection.add_callback_threadsafe(self.rabbit_channel.stop_consuming)
        except Exception:
            logger.exception("Failed to request RabbitMQ consumer shutdown")

    def _close_rabbitmq_resources(self) -> None:
        if self.rabbit_channel and self.rabbit_channel.is_open:
            try:
                self.rabbit_channel.close()
            except Exception:
                logger.exception("Failed to close RabbitMQ channel cleanly")
        if self.rabbit_connection and self.rabbit_connection.is_open:
            try:
                self.rabbit_connection.close()
            except Exception:
                logger.exception("Failed to close RabbitMQ connection cleanly")

    def _publish(
        self,
        ch: pika.channel.Channel,
        routing_key: str,
        correlation_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        ch.basic_publish(
            exchange="",
            routing_key=routing_key,
            body=json.dumps(payload),
            properties=pika.BasicProperties(
                correlation_id=correlation_id,
                content_type="application/json",
                delivery_mode=2,
            ),
        )

    def _process_prediction_request(self, data: dict[str, Any]) -> dict[str, Any]:
        raw_features = data.get("features", [])
        if not isinstance(raw_features, list):
            raise ValueError("features must be a list")
        return self.prediction_service.predict(raw_features)

    def _process_health_request(self, data: dict[str, Any]) -> dict[str, Any]:
        health = self.prediction_service.health()
        health["details"] = {
            "rabbitmq_connected": bool(
                self.rabbit_connection and self.rabbit_connection.is_open
            ),
            "service_type": "rabbitmq_worker",
            "messages_received": self.messages_received,
            "messages_processed": self.messages_processed,
            "messages_failed": self.messages_failed,
        }
        return health


def main() -> None:
    service = AsyncMLService()

    def signal_handler(signum: int, frame: Any) -> None:
        service.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    service.start()


if __name__ == "__main__":
    main()
