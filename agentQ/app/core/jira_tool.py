import logging
from typing import Dict, Any, Optional
from agentQ.app.core.toolbox import Tool
from agentQ.app.core.integrationhub_tool import trigger_integration_flow

logger = logging.getLogger(__name__)

def create_jira_issue(
    project_key: str,
    summary: str,
    description: str,
    issue_type: str = "Task",
    config: Dict[str, Any] = {}
) -> str:
    """
    Creates a new issue in Jira.
    
    Args:
        project_key (str): The key of the Jira project (e.g., "PROJ").
        summary (str): The summary or title of the issue.
        description (str): The detailed description of the issue.
        issue_type (str): The type of issue to create (e.g., "Task", "Bug", "Story").
        
    Returns:
        A string confirming the creation of the issue or an error message.
    """
    logger.info(f"Creating Jira issue in project {project_key}: {summary}")
    
    parameters = {
        "connector": "jira",
        "action": "create_issue",
        "project_key": project_key,
        "summary": summary,
        "description": description,
        "issue_type": issue_type
    }
    
    return trigger_integration_flow(flow_id="jira-issue-management", parameters=parameters, config=config)


def update_jira_issue(
    issue_key: str,
    summary: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    config: Dict[str, Any] = {}
) -> str:
    """
    Updates an existing issue in Jira.
    
    Args:
        issue_key (str): The key of the issue to update (e.g., "PROJ-123").
        summary (str, optional): The new summary for the issue.
        description (str, optional): The new description for the issue.
        status (str, optional): The new status for the issue (e.g., "In Progress", "Done").
        
    Returns:
        A string confirming the update of the issue or an error message.
    """
    logger.info(f"Updating Jira issue {issue_key}")
    
    parameters = {
        "connector": "jira",
        "action": "update_issue",
        "issue_key": issue_key,
        "summary": summary,
        "description": description,
        "status": status
    }
    
    return trigger_integration_flow(flow_id="jira-issue-management", parameters=parameters, config=config)


# --- Tool Registration ---

create_jira_issue_tool = Tool(
    name="create_jira_issue",
    description="Creates a new issue in Jira.",
    func=create_jira_issue
)

update_jira_issue_tool = Tool(
    name="update_jira_issue",
    description="Updates an existing issue in Jira.",
    func=update_jira_issue
) 