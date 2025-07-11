import structlog
import json
from agentQ.app.core.toolbox import Tool
from agentQ.app.services.multi_agent_coordinator import multi_agent_coordinator

logger = structlog.get_logger(__name__)

def discover_peers(required_skills: str, config: dict = None) -> str:
    """
    Finds available agents with a specific set of required skills.

    Args:
        required_skills (str): A comma-separated string of skills (e.g., "sast_scan,k8s_deployment").

    Returns:
        str: A JSON string list of agent IDs that match the required skills, or an error message.
    """
    logger.info("Discovering peers with skills", required_skills=required_skills)
    try:
        skills_list = [skill.strip() for skill in required_skills.split(',')]
        # This calls a method on the coordinator service we initialized.
        capable_agents = multi_agent_coordinator.find_capable_agents(skills_list)
        
        return json.dumps([agent.agent_id for agent in capable_agents])
    except Exception as e:
        logger.error("Failed to discover peers", exc_info=True)
        return f"Error: An unexpected error occurred while discovering peers: {e}"

def propose_collaboration(agent_ids_json: str, goal_description: str, config: dict = None) -> str:
    """
    Proposes a collaboration between a group of agents to achieve a specific goal.

    Args:
        agent_ids_json (str): A JSON string list of agent IDs to include in the collaboration.
        goal_description (str): A clear, natural language description of the shared goal.

    Returns:
        str: A JSON string containing the collaboration session ID, or an error message.
    """
    logger.info("Proposing collaboration", goal=goal_description)
    try:
        agent_ids = json.loads(agent_ids_json)
        if not isinstance(agent_ids, list):
            return "Error: agent_ids_json must be a JSON list of strings."

        # This calls another method on the coordinator service.
        session = multi_agent_coordinator.start_collaboration_session(
            agent_ids=agent_ids,
            goal=goal_description
        )
        
        return json.dumps({"collaboration_session_id": session.session_id})
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for agent_ids_json."
    except Exception as e:
        logger.error("Failed to propose collaboration", exc_info=True)
        return f"Error: An unexpected error occurred while proposing collaboration: {e}"

discover_peers_tool = Tool(
    name="discover_peers",
    description="Finds available agents with a specific set of skills required for a task.",
    func=discover_peers
)

propose_collaboration_tool = Tool(
    name="propose_collaboration",
    description="Initiates a collaboration session with a group of agents for a shared goal.",
    func=propose_collaboration
)

collaboration_tools = [discover_peers_tool, propose_collaboration_tool] 