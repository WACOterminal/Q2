
import logging
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any

from managerQ.app.core.planner import planner, AmbiguousGoalError
from managerQ.app.core.goal_manager import goal_manager
from managerQ.app.models import GoalStatus
from managerQ.app.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

class CreateGoalRequest(BaseModel):
    prompt: str

class ClarifyGoalRequest(BaseModel):
    clarification: str

@router.post("", status_code=202, response_model=GoalStatus)
async def create_goal(
    request: CreateGoalRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Accepts a high-level goal, attempts to create a plan, and starts execution.
    If the goal is ambiguous, it returns a 400 error with a clarifying question.
    """
    try:
        user_id = current_user.get("sub")
        logger.info(f"Received goal from user {user_id}: '{request.prompt}'")
        
        goal_status = await goal_manager.process_new_goal(request.prompt, user_id)
        
        return goal_status

    except AmbiguousGoalError as e:
        logger.warning(f"Goal is ambiguous: {e.clarifying_question}")
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "AMBIGUOUS_GOAL",
                "message": e.message,
                "clarifying_question": e.clarifying_question,
            },
        )
    except Exception as e:
        logger.error(f"Error processing goal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process the goal.")


@router.post("/{goal_id}/clarify", response_model=GoalStatus)
async def clarify_goal(
    goal_id: str,
    request: ClarifyGoalRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Provides a clarifying answer to an ambiguous goal, allowing the planner to try again.
    """
    try:
        user_id = current_user.get("sub")
        logger.info(f"Received clarification for goal {goal_id} from user {user_id}")
        
        goal_status = await goal_manager.process_clarification(
            goal_id=goal_id,
            user_clarification=request.clarification,
            user_id=user_id
        )

        return goal_status

    except ValueError as e:
        logger.warning(f"Value error during clarification: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing clarification for goal {goal_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process the clarification.") 