import logging
import json
import pulsar
import threading

from QuantumPulse.app.services.qgan_service import qgan_service
from QuantumPulse.app.core.pulsar_client import pulsar_client # Assuming a shared client exists

logger = logging.getLogger(__name__)

COMMAND_TOPIC = "persistent://public/default/qgan-commands"
RESULTS_TOPIC = "persistent://public/default/qgan-results"
SUBSCRIPTION_NAME = "qgan-service-subscription"

_running = True

def handle_command(msg):
    """Parses a command message and executes the corresponding QGAN service method."""
    try:
        data = json.loads(msg.data().decode('utf-8'))
        command = data.get("command")
        request_id = data.get("request_id")
        params = data.get("params", {})

        if not all([command, request_id]):
            logger.error("Invalid command received: missing command or request_id", received_data=data)
            return

        logger.info("Processing QGAN command", command=command, request_id=request_id)

        result = None
        if command == "train":
            real_data = params.get("data")
            epochs = params.get("epochs", 20)
            if not real_data:
                raise ValueError("Training data is required for the 'train' command.")
            result = qgan_service.train(real_data, epochs=epochs)

        elif command == "generate_samples":
            n_samples = params.get("n_samples", 10)
            result = qgan_service.generate_samples(n_samples=n_samples)
        
        else:
            raise ValueError(f"Unknown command: {command}")

        # Publish the result back
        response = {
            "request_id": request_id,
            "status": "success",
            "result": result
        }
        pulsar_client.publish_message(RESULTS_TOPIC, response)
        logger.info("Successfully processed command and published result", request_id=request_id)

    except Exception as e:
        logger.error("Failed to process QGAN command", exc_info=True)
        response = {
            "request_id": data.get("request_id", "unknown"),
            "status": "error",
            "error": str(e)
        }
        pulsar_client.publish_message(RESULTS_TOPIC, response)


def consume_commands():
    """Main consumer loop for QGAN commands."""
    consumer = None
    try:
        # Assuming the shared client can create consumers or we create a new one
        client = pulsar.Client('pulsar://pulsar:6650')
        consumer = client.subscribe(COMMAND_TOPIC, SUBSCRIPTION_NAME)
        
        logger.info(f"QGAN Command Handler started. Listening on topic: {COMMAND_TOPIC}")

        while _running:
            try:
                msg = consumer.receive(timeout_millis=1000)
                handle_command(msg)
                consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception:
                logger.error("Error in QGAN consumer loop. Negative acknowledging.", exc_info=True)
                if 'msg' in locals():
                    consumer.negative_acknowledge(msg)
    
    finally:
        if consumer:
            consumer.close()
        if 'client' in locals() and client:
            client.close()
        logger.info("QGAN Command Handler shut down.")


def start_qgan_handler():
    """Starts the QGAN command consumer in a background thread."""
    handler_thread = threading.Thread(target=consume_commands, daemon=True)
    handler_thread.start()
    logger.info("QGAN handler thread initiated.")

def stop_qgan_handler():
    """Stops the QGAN command consumer."""
    global _running
    _running = False
    logger.info("Stopping QGAN handler...") 