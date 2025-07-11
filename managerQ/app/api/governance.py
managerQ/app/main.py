from fastapi import APIRouter, HTTPException, status
from managerQ.app.core.workflow_manager import workflow_manager

router = APIRouter()

@router.get("/vetoed-workflows/{workflow_id}")
async def get_vetoed_workflow_details(workflow_id: str):
    """
    Retrieves the full details of a specific workflow that was vetoed.
    """
    workflow = workflow_manager.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found.")
    
    if workflow.status.value != "FAILED" or not workflow.final_result.startswith("Ethical Veto:"):
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Workflow was not vetoed.")

    return workflow.dict() 