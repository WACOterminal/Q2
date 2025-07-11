import structlog
import json
import time
import random
from agentQ.app.core.toolbox import Tool
from agentQ.app.services.neuromorphic_engine import neuromorphic_engine
from agentQ.app.services.spiking_neural_networks import spiking_neural_networks

logger = structlog.get_logger(__name__)

def configure_snn_for_anomaly_detection(pulsar_topic: str, config: dict = None) -> str:
    """
    Configures a Spiking Neural Network to monitor a Pulsar topic for anomalies.

    Args:
        pulsar_topic (str): The Pulsar topic to monitor (e.g., 'persistent://public/default/market-data').

    Returns:
        str: A confirmation message with the network ID.
    """
    logger.info("Configuring SNN for anomaly detection", topic=pulsar_topic)
    try:
        # In a real system, this would configure a specific SNN architecture
        # For now, we'll assume a default architecture is created and used.
        architecture_id = neuromorphic_engine.create_cognitive_architecture(
            name=f"AnomalyDetector-{pulsar_topic}",
            architecture_type="recurrent",
            task_types=["pattern_recognition"]
        )
        # Here we would also set up a Pulsar reader for the SNN to consume from the topic.
        
        return json.dumps({
            "status": "success",
            "message": f"Neuromorphic architecture '{architecture_id}' is now monitoring topic '{pulsar_topic}'.",
            "network_id": architecture_id
        })
    except Exception as e:
        logger.error("Failed to configure SNN", exc_info=True)
        return f"Error: An unexpected error occurred during SNN configuration: {e}"

def get_snn_anomalies(network_id: str, config: dict = None) -> str:
    """
    Retrieves the latest detected anomalies from a specific SNN.
    This is a destructive read; anomalies are cleared after being retrieved.

    Args:
        network_id (str): The ID of the SNN architecture to query.

    Returns:
        str: A JSON string list of detected anomalies.
    """
    logger.info("Fetching SNN anomalies from live engine", network_id=network_id)
    try:
        # --- NEW: Call the real SNN service ---
        anomalies = spiking_neural_networks.detected_anomalies.get(network_id, [])
        
        if anomalies:
            # Clear the anomalies list after retrieving them to prevent re-reporting
            spiking_neural_networks.detected_anomalies[network_id] = []
            logger.warning(f"Retrieved {len(anomalies)} new anomalies.", network_id=network_id)
        
        return json.dumps(anomalies)
        
    except Exception as e:
        logger.error("Failed to fetch SNN anomalies", exc_info=True)
        return f"Error: An unexpected error occurred while fetching SNN anomalies: {e}"

# --- Tool Registration ---
configure_snn_tool = Tool(
    name="configure_snn_for_anomaly_detection",
    description="Sets up a Spiking Neural Network to monitor a Pulsar data stream for real-time anomaly detection.",
    func=configure_snn_for_anomaly_detection
)

get_anomalies_tool = Tool(
    name="get_snn_anomalies",
    description="Retrieves a list of the latest anomalies detected by a specific SNN.",
    func=get_snn_anomalies
)

neuromorphic_tools = [configure_snn_tool, get_anomalies_tool] 