import httpx
import structlog
import json
from agentQ.app.core.toolbox import Tool

logger = structlog.get_logger(__name__)

def generate_workflow_from_prompt(prompt: str, config: dict = None) -> str:
    """
    Invokes the WorkflowEngine service to generate a workflow YAML from a natural language prompt.

    Args:
        prompt (str): The natural language description of the desired workflow.

    Returns:
        str: The generated workflow YAML as a string, or an error message.
    """
    workflow_engine_url = config.get("workflow_engine_url")
    if not workflow_engine_url:
        return "Error: workflow_engine_url is not configured."

    request_url = f"{workflow_engine_url}/v1/generate-workflow"
    logger.info("Calling WorkflowEngine service to generate workflow", url=request_url)

    try:
        with httpx.Client() as client:
            response = client.post(request_url, json={"description": prompt}, timeout=180.0)
            response.raise_for_status()
            # The service should return the YAML content directly
            return response.text
    except httpx.RequestError as e:
        logger.error("Failed to call WorkflowEngine service", error=str(e))
        return f"Error: Request to WorkflowEngine failed. Reason: {e}"
    except Exception as e:
        logger.error("An unexpected error occurred during workflow generation tool execution", exc_info=True)
        return f"Error: An unexpected error occurred: {e}"

workflow_generator_tool = Tool(
    name="generate_workflow_from_prompt",
    description="Generates a complete workflow YAML from a natural language description of a goal.",
    func=generate_workflow_from_prompt,
    config={"workflow_engine_url": "http://workflow-engine.q-platform.svc.cluster.local:8000"} # This needs to be a real service
) 