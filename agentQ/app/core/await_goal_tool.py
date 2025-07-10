
import logging
import time
import httpx
from typing import Dict, Any

from agentQ.app.core.toolbox import Tool

logger = logging.getLogger(__name__)

def await_goal(goal_id: int, context_key: str, parent_workflow_id: str, timeout: int = 300, poll_interval: int = 15, config: dict = {}) -> str:
    """
    Waits for a sub-goal to be completed and stores its result in the parent's shared context.
    
    Args:
        goal_id (int): The ID of the goal (work package) to wait for.
        context_key (str): The key under which to store the result in the shared context.
        parent_workflow_id (str): The ID of the parent workflow whose context should be updated.
        timeout (int): The maximum time to wait in seconds.
        poll_interval (int): The time to wait between checking the status.
        
    Returns:
        A string indicating the final status or an error/timeout message.
    """
    logger.info(f"Awaiting completion of goal (work package) {goal_id}...")
    
    start_time = time.time()
    
    integration_hub_url = config.get('services', {}).get('integrationhub_url', 'http://localhost:8002')
    manager_url = config.get('services', {}).get('managerq_url', 'http://localhost:8000')
    service_token = config.get('service_token')
    
    if not service_token:
        return "Error: Service token not available. Cannot await goal."

    headers = {
        "Authorization": f"Bearer {service_token}",
        "Content-Type": "application/json"
    }

    while time.time() - start_time < timeout:
        try:
            with httpx.Client() as client:
                response = client.get(f"{integration_hub_url}/api/v1/openproject/work-packages/{goal_id}", headers=headers)
                
                if response.status_code == 200:
                    work_package = response.json()
                    status = work_package.get("_links", {}).get("status", {}).get("name", "Unknown")
                    workflow_id = work_package.get("_links", {}).get("customField1", {}).get("title") # Assuming CF1 is workflow_id
                    
                    logger.info(f"Checked status for goal {goal_id}: {status}")
                    
                    if status in ["Closed", "Done", "Resolved"]:
                        if not workflow_id:
                            return f"Success: Goal {goal_id} is complete, but no associated workflow ID was found."
                        
                        logger.info(f"Goal {goal_id} complete. Fetching final result from workflow {workflow_id}...")
                        workflow_response = client.get(f"{manager_url}/api/v1/workflows/{workflow_id}", headers=headers)
                        
                        if workflow_response.status_code == 200:
                            workflow_data = workflow_response.json()
                            final_result = workflow_data.get("final_result", "No result found.")
                            
                            # --- New: Update parent workflow's context ---
                            logger.info(f"Updating parent workflow '{parent_workflow_id}' with result.")
                            context_update_payload = {context_key: final_result}
                            update_response = client.patch(
                                f"{manager_url}/api/v1/workflows/{parent_workflow_id}/context",
                                json=context_update_payload,
                                headers=headers
                            )
                            if update_response.status_code == 204:
                                return f"Success: Goal {goal_id} complete. Result stored in parent workflow's context key '{context_key}'."
                            else:
                                return f"Error: Goal {goal_id} complete, but failed to update parent workflow context. Status: {update_response.status_code}"
                            # ------------------------------------------
                        else:
                            return f"Success: Goal {goal_id} is complete, but failed to fetch final result. Status: {workflow_response.status_code}"
                else:
                    logger.warning(f"Failed to get status for goal {goal_id}. Status: {response.status_code}")

            time.sleep(poll_interval)
            
        except Exception as e:
            logger.error(f"An error occurred while awaiting goal {goal_id}: {e}", exc_info=True)
            return f"Error: An unexpected error occurred while waiting for goal {goal_id}."
            
    return f"Error: Timed out after {timeout} seconds waiting for goal {goal_id} to complete."


# --- Tool Registration Object ---

await_goal_tool = Tool(
    name="await_goal_completion",
    description="Pauses execution, waits for a sub-goal to complete, and stores its final result in the current workflow's shared context under the given key.",
    func=await_goal
) 