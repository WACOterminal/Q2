
import logging
from typing import Dict, Any

from agentQ.app.core.toolbox import Tool
from agentQ.app.core.integrationhub_tool import trigger_integration_flow

logger = logging.getLogger(__name__)

def update_work_package_comment(work_package_id: int, comment: str, config: dict = {}) -> str:
    """
    Adds a comment to a specific work package in OpenProject.
    
    Args:
        work_package_id (int): The ID of the work package to update.
        comment (str): The content of the comment to add.
        
    Returns:
        A success or error message from the IntegrationHub.
    """
    logger.info(f"Adding comment to OpenProject work package {work_package_id}")
    
    try:
        # This tool is a specific application of the generic 'trigger_integration_flow' tool.
        # It calls a pre-defined flow in IntegrationHub responsible for this action.
        result = trigger_integration_flow(
            flow_id="openproject_add_comment",
            parameters={
                "work_package_id": work_package_id,
                "comment_body": comment
            },
            config=config # Pass config down to the underlying tool
        )
        return f"Successfully posted comment to work package {work_package_id}. Response: {result}"
    except Exception as e:
        logger.error(f"Failed to add comment to work package {work_package_id}: {e}", exc_info=True)
        return f"Error: Failed to add comment. Reason: {e}"

# --- Tool Registration Object ---

openproject_comment_tool = Tool(
    name="update_work_package_comment",
    description="Adds a comment to an existing work package (e.g., a task or a goal ticket) in the OpenProject system. Use this to provide real-time status updates on your progress.",
    func=update_work_package_comment
) 