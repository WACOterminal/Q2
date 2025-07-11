import structlog
import json
import uuid
import pulsar
import time
from agentQ.app.core.toolbox import Tool
from typing import Dict, Any, Optional

logger = structlog.get_logger(__name__)

PULSAR_URL = 'pulsar://pulsar:6650'
COMMAND_TOPIC = "persistent://public/default/qgan-commands"
RESULTS_TOPIC = "persistent://public/default/qgan-results"

def _invoke_qgan_service(command: str, params: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    """
    A private helper function to send a command to the QGAN service and await a response.
    """
    request_id = str(uuid.uuid4())
    logger.info("Invoking QGAN service", command=command, request_id=request_id)
    
    client = None
    producer = None
    consumer = None
    
    try:
        client = pulsar.Client(PULSAR_URL)
        producer = client.create_producer(COMMAND_TOPIC)
        consumer = client.subscribe(RESULTS_TOPIC, f"qgan-client-sub-{request_id}")

        # Construct and send the command
        command_payload = {
            "request_id": request_id,
            "command": command,
            "params": params
        }
        producer.send(json.dumps(command_payload).encode('utf-8'))

        # Wait for the specific response
        while True:
            msg = consumer.receive(timeout_millis=timeout * 1000)
            response = json.loads(msg.data().decode('utf-8'))
            consumer.acknowledge(msg)
            
            if response.get("request_id") == request_id:
                logger.info("Received response for QGAN request", request_id=request_id, status=response.get("status"))
                if response.get("status") == "error":
                    raise RuntimeError(f"QGAN service returned an error: {response.get('error')}")
                return response.get("result")
    
    except pulsar.Timeout:
        logger.error("Timed out waiting for QGAN service response", request_id=request_id)
        raise
    except Exception as e:
        logger.error("Failed to invoke QGAN service", exc_info=True)
        raise
    finally:
        if consumer:
            consumer.close()
        if producer:
            producer.close()
        if client:
            client.close()


def train_qgan_model(dataset_id: str, epochs: int = 20, config: Optional[Dict] = None) -> str:
    """
    Trains a Quantum Generative Adversarial Network (QGAN) on a given dataset.
    This tool sends a command to the QuantumPulse service to perform the training.

    Args:
        dataset_id (str): The ID or reference to the dataset to train on.
        epochs (int): The number of training epochs.

    Returns:
        str: A JSON string containing the result of the training job.
    """
    logger.info("Requesting QGAN model training", dataset_id=dataset_id, epochs=epochs)
    # In a real system, the dataset_id would be used to fetch data.
    # Here, we'll just pass some mock data for the service to use.
    mock_real_data = [[1.0, -1.0, 1.0, -1.0]] * 10
    params = {"data": mock_real_data, "epochs": epochs}
    
    try:
        result = _invoke_qgan_service("train", params)
        return json.dumps(result)
    except Exception as e:
        return f"Error: Failed to train QGAN model: {e}"


def generate_qgan_samples(model_id: str, num_samples: int, config: Optional[Dict] = None) -> str:
    """
    Generates new data samples using a trained QGAN model.
    This tool sends a command to the QuantumPulse service to perform the generation.

    Args:
        model_id (str): The ID of the trained QGAN model (currently ignored, uses singleton).
        num_samples (int): The number of samples to generate.

    Returns:
        str: A JSON string list of the generated data samples.
    """
    logger.info("Requesting QGAN sample generation", model_id=model_id, num_samples=num_samples)
    params = {"n_samples": num_samples}
    
    try:
        samples = _invoke_qgan_service("generate_samples", params)
        return json.dumps(samples)
    except Exception as e:
        return f"Error: Failed to generate QGAN samples: {e}"


# --- Tool Registration ---
train_qgan_tool = Tool(
    name="train_qgan_model",
    description="Trains a Quantum Generative Adversarial Network (QGAN) to learn a data distribution.",
    func=train_qgan_model
)

generate_samples_tool = Tool(
    name="generate_qgan_samples",
    description="Uses a trained QGAN to generate new, synthetic data samples.",
    func=generate_qgan_samples
)

quantum_ml_tools = [train_qgan_tool, generate_samples_tool] 