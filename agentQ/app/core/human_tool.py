import logging
import pulsar
import json
from typing import Dict, Any
import uuid

from agentQ.app.core.toolbox import Tool

logger = logging.getLogger(__name__)

# --- Configuration ---
# In a real system, this would come from a shared config
PULSAR_SERVICE_URL = "pulsar://localhost:6650"
HUMAN_FEEDBACK_TOPIC = "persistent://public/default/q.agentq.human_feedback"
CLARIFICATION_TOPIC = "persistent://public/default/q.h2m.clarification_requests"
COPILOT_REQUEST_TOPIC = "persistent://public/default/q.h2m.copilot_requests"

# --- Tool Definition ---

def ask_human_for_clarification(question: str, conversation_id: str) -> str:
    """
    Asks a human for clarification or more information when you are stuck.
    
    Args:
        question (str): The specific question you need to ask the human.
        conversation_id (str): The ID of the current conversation, for routing the response.
        
    Returns:
        A confirmation that the question has been sent. The human's response will arrive in a subsequent turn.
    """
    client = None
    producer = None
    try:
        client = pulsar.Client(PULSAR_SERVICE_URL)
        producer = client.create_producer(HUMAN_FEEDBACK_TOPIC)
        
        message_payload = {
            "type": "clarification_request",
            "question": question,
            "conversation_id": conversation_id
        }
        
        producer.send(json.dumps(message_payload).encode('utf-8'))
        logger.info(f"Published human clarification request: {question}")
        
        return f"Your question has been sent to the human: '{question}'. Their response will appear in the conversation history."

    except Exception as e:
        logger.error(f"Failed to ask human for clarification: {e}", exc_info=True)
        return "Error: Could not send the question to the human."
    finally:
        if producer:
            producer.close()
        if client:
            client.close()


def request_copilot_approval(conversation_id: str, thought: str, proposed_action: Dict[str, Any]) -> str:
    """
    Sends the agent's current thought and proposed action to the human for approval
    and waits for a response. This is a blocking call.
    """
    client = None
    producer = None
    consumer = None
    try:
        client = pulsar.Client(PULSAR_SERVICE_URL)
        
        # Unique reply topic for this specific request
        reply_topic = f"persistent://public/default/q.h2m.copilot_replies.{uuid.uuid4().hex}"
        
        producer = client.create_producer(COPILOT_REQUEST_TOPIC)
        consumer = client.subscribe(reply_topic, subscription_name="agent-copilot-reply-sub")

        request_payload = {
            "type": "copilot_approval_request",
            "conversation_id": conversation_id,
            "thought": thought,
            "proposed_action": proposed_action,
            "reply_topic": reply_topic
        }
        
        producer.send(json.dumps(request_payload).encode('utf-8'))
        logger.info(f"Published co-pilot approval request for conversation: {conversation_id}")
        
        # Block and wait for the human's response
        logger.info("Waiting for co-pilot response...")
        msg = consumer.receive(timeout_millis=60000) # 60-second timeout
        response_payload = msg.data().decode('utf-8')
        consumer.acknowledge(msg)
        
        logger.info("Co-pilot response received.")
        return response_payload

    except Exception as e:
        logger.error(f"Failed during co-pilot approval request: {e}", exc_info=True)
        # Default to a "deny" response on error/timeout to be safe
        return json.dumps({"decision": "deny", "reason": f"Error or timeout: {e}"})
    finally:
        if producer: producer.close()
        if consumer: consumer.close()
        if client: client.close()

# --- Tool Registration Object ---

clarification_tool = Tool(
    name="ask_human_for_clarification",
    description="When you are stuck, require more information, or need a subjective opinion to proceed, use this tool to ask a human for help. Provide the conversation_id to ensure the response is routed correctly.",
    func=ask_human_for_clarification
)

copilot_approval_tool = Tool(
    name="request_copilot_approval",
    description="Pauses execution and asks a human co-pilot to approve, deny, or suggest changes to the agent's next action.",
    func=request_copilot_approval
)

# Export all human interaction tools
human_tools = [clarification_tool, copilot_approval_tool] 