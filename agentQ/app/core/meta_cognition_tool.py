import structlog
import json
from agentQ.app.core.toolbox import Tool

logger = structlog.get_logger(__name__)

def get_agent_thought_history(agent_personality: str, limit: int = 50, config: dict = None) -> str:
    """
    Retrieves the recent thought process history for a specific agent personality.
    This includes the final thought for each of the last N completed tasks.
    (This is a mock tool and returns generated data).

    Args:
        agent_personality (str): The personality of the agent to query (e.g., 'devops_agent').
        limit (int): The maximum number of recent thoughts to retrieve.

    Returns:
        str: A JSON string list of thought records.
    """
    logger.info("Fetching thought history for agent personality", agent_personality=agent_personality)
    
    # In a real system, this would query a database (e.g., Elasticsearch or Ignite)
    # where the full task results (including the 'thought' JSON blob) are stored.
    mock_history = [
        {
            "thought": "The user wants to restart a service. The `k8s_restart_deployment` tool is the most direct way to achieve this. I have the service name and namespace, so I can proceed.",
            "action": "k8s_restart_deployment",
            "timestamp": "2024-08-03T10:00:00Z"
        },
        {
            "thought": "The user asked for logs. The `elasticsearch_query` tool is appropriate. I will search for errors in the specified service.",
            "action": "elasticsearch_query",
            "timestamp": "2024-08-03T09:30:00Z"
        },
        {
            "thought": "The user wants to restart a service, but I don't have the namespace. I must first ask the user for the missing information.",
            "action": "ask_human_for_clarification",
            "timestamp": "2024-08-03T09:00:00Z"
        }
    ]
    
    return json.dumps(mock_history[:limit], indent=2)

meta_cognition_tool = Tool(
    name="get_agent_thought_history",
    description="Retrieves the recent reasoning history (thought processes) for a specific type of agent.",
    func=get_agent_thought_history
) 