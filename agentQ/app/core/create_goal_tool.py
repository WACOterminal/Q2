
import logging
import httpx
from typing import Dict, Any, Optional

from agentQ.app.core.toolbox import Tool

logger = logging.getLogger(__name__)

def create_goal(prompt: str, parent_id: Optional[int] = None, config: dict = {}) -> str:
    """
    Creates a new high-level goal for the platform to pursue.
    Can optionally be linked to a parent goal.
    
    Args:
        prompt (str): The high-level objective for the new goal.
        parent_id (Optional[int]): The ID of the parent goal in OpenProject.
        
    Returns:
        A confirmation or error message.
    """
    logger.info(f"Creating a new sub-goal with prompt: '{prompt}'")
    
    try:
        manager_url = config.get('services', {}).get('managerq_url', 'http://localhost:8000')
        service_token = config.get('service_token')
        
        if not service_token:
            return "Error: Service token not available. Cannot create a sub-goal."

        headers = {
            "Authorization": f"Bearer {service_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "project_id": config.get('openproject_project_id', 1), # Assume a default project
            "parent_id": parent_id
        }
        
        with httpx.Client() as client:
            response = client.post(f"{manager_url}/api/v1/goals", json=payload, headers=headers)
            
            if response.status_code == 400: # Ambiguous goal
                error_details = response.json().get("detail", {})
                return f"Error: Could not create goal. It was ambiguous. The planner asks: {error_details.get('clarifying_question')}"
            
            response.raise_for_status()
            
            goal_status = response.json()
            return f"Successfully created new goal. Goal ID: {goal_status.get('goal_id')}, Status: {goal_status.get('status')}"

    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to create sub-goal via managerQ API: {e.response.text}", exc_info=True)
        return f"Error: An HTTP error occurred while creating the sub-goal: {e.response.text}"
    except httpx.RequestError as e:
        logger.error(f"A request error occurred while trying to reach managerQ: {e}", exc_info=True)
        return f"Error: Could not connect to the goal manager service: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while creating sub-goal: {e}", exc_info=True)
        return f"Error: An unexpected error occurred: {e}"

# --- Tool Registration Object ---

create_goal_tool = Tool(
    name="create_goal",
    description="Creates a new, high-level goal for the platform. Use this to delegate a complex, multi-step sub-problem to another agent. Provide a clear, detailed prompt for the new goal.",
    func=create_goal
) 