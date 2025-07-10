from fastapi import APIRouter, HTTPException, status, Depends
import logging
from pydantic import BaseModel
from typing import Optional

from managerQ.app.dependencies import get_task_dispatcher, get_result_listener
from managerQ.app.core.task_dispatcher import TaskDispatcher
from managerQ.app.core.result_listener import ResultListener

logger = logging.getLogger(__name__)
router = APIRouter()

class TaskRequest(BaseModel):
    prompt: str
    personality: str = "default"

class TaskResponse(BaseModel):
    task_id: str

class TaskResult(BaseModel):
    task_id: str
    result: Optional[str] = None
    status: str

@router.post("", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_task(
    request: TaskRequest,
    task_dispatcher: TaskDispatcher = Depends(get_task_dispatcher)
):
    """
    Accepts a user prompt and dispatches it to an available agent.
    """
    try:
        task_id = task_dispatcher.dispatch_task(
            personality=request.personality,
            prompt=request.prompt
        )
        return TaskResponse(task_id=task_id)
    except RuntimeError as e:
        logger.error(f"Task dispatch failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred while creating task: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred.")

@router.get("/{task_id}", response_model=TaskResult)
async def get_task_result(
    task_id: str,
    result_listener: ResultListener = Depends(get_result_listener)
):
    """
    Retrieves the result of a dispatched task.
    """
    try:
        result = await result_listener.wait_for_result(task_id)
        return TaskResult(task_id=task_id, result=result, status="COMPLETED")
    except TimeoutError:
        return TaskResult(task_id=task_id, status="PENDING")
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching task result: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred.") 