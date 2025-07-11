import structlog
import json
import pulsar
from agentQ.app.core.toolbox import Tool
from typing import Dict, Any

logger = structlog.get_logger(__name__)

# This topic is listened to by the H2M service
PROPOSALS_TOPIC = "persistent://public/default/q.h2m.proactive_proposals"

def propose_assistance(user_id: str, suggestion_text: str, action_type: str, action_payload: Dict[str, Any], config: dict = None) -> str:
    """
    Proposes a helpful action to a specific user via the H2M service.

    Args:
        user_id (str): The ID of the user to send the suggestion to.
        suggestion_text (str): The natural language text of the suggestion.
        action_type (str): The type of action proposed (e.g., 'CREATE_WORKFLOW').
        action_payload (Dict[str, Any]): The data needed to execute the proposed action.

    Returns:
        str: A confirmation that the proposal has been sent.
    """
    logger.info("Proposing assistance to user", user_id=user_id, suggestion=suggestion_text)
    
    pulsar_url = config.get("pulsar_url", "pulsar://localhost:6650")
    client = None
    producer = None
    try:
        client = pulsar.Client(pulsar_url)
        producer = client.create_producer(PROPOSALS_TOPIC)
        
        message = {
            "user_id": user_id,
            "suggestion_text": suggestion_text,
            "action_type": action_type,
            "action_payload": action_payload,
        }
        
        producer.send(json.dumps(message).encode('utf-8'))
        
        return "Proposal sent to user successfully."
        
    except Exception as e:
        logger.error("Failed to propose assistance", exc_info=True)
        return f"Error: Could not send proposal. Reason: {e}"
    finally:
        if producer: producer.close()
        if client: client.close()


propose_assistance_tool = Tool(
    name="propose_assistance",
    description="Sends a proactive suggestion to a user to help them automate a task or improve their workflow.",
    func=propose_assistance
) 