import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from managerQ.app.core.task_dispatcher import task_dispatcher
from shared.q_auth_parser import get_current_user, User

router = APIRouter()
logger = structlog.get_logger(__name__)

class WorkflowGenerationRequest(BaseModel):
    description: str

class WorkflowGenerationResponse(BaseModel):
    task_id: str
    message: str

@router.post("/generate", response_model=WorkflowGenerationResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_workflow(
    request: WorkflowGenerationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Accepts a natural language description and triggers a task for the planner agent
    to generate a workflow YAML.
    """
    logger.info("Received workflow generation request", description=request.description, user_id=current_user.id)
    
    try:
        # This prompt is sent to the planner_agent, which knows how to use the
        # `generate_workflow_from_prompt` tool.
        task_id = task_dispatcher.dispatch_task(
            prompt=request.description,
            agent_personality="planner_agent"
        )

        return WorkflowGenerationResponse(
            task_id=task_id,
            message="Workflow generation task has been dispatched. The result will be available shortly."
        )

    except Exception as e:
        logger.error("Failed to dispatch workflow generation task", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while dispatching the generation task."
        ) 