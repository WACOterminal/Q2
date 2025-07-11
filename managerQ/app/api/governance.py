from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Optional, List, Any, Dict
from managerQ.app.core.workflow_manager import workflow_manager
from managerQ.app.models import Workflow
from shared.q_auth_parser import get_current_user_is_admin
import yaml
import os
import json

router = APIRouter()

CONSTITUTION_PATH = "governance/platform_constitution.yaml"

@router.get("/constitution", response_model=Any)
async def get_constitution(is_admin: bool = Depends(get_current_user_is_admin)):
    """Retrieves the current platform constitution."""
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized.")
    try:
        with open(CONSTITUTION_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Constitution file not found.")

@router.post("/constitution", status_code=204)
async def update_constitution(
    constitution_data: Dict[str, Any] = Body(...),
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Updates the platform constitution file."""
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized.")
    try:
        with open(CONSTITUTION_PATH, 'w') as f:
            yaml.dump(constitution_data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write constitution file: {e}")


@router.get("/vetoed-workflows", response_model=List[Workflow])
async def get_vetoed_workflows(is_admin: bool = Depends(get_current_user_is_admin)):
    """
    Retrieves a list of all workflows that have been halted with a VETOED status.
    This endpoint is protected and only accessible by admins.
    """
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to view vetoed workflows.")
    
    # This relies on a new method in the WorkflowManager
    return workflow_manager.get_workflows_by_status("VETOED")


@router.get("/vetoed-workflows/{workflow_id}", response_model=Workflow)
async def get_vetoed_workflow_details(workflow_id: str, is_admin: bool = Depends(get_current_user_is_admin)):
    """
    Retrieves the full details for a single vetoed workflow.
    """
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to view vetoed workflows.")

    workflow = workflow_manager.get_workflow(workflow_id)
    if not workflow or workflow.status.value != "VETOED":
        raise HTTPException(status_code=404, detail="Vetoed workflow not found.")
    
    return workflow 

@router.get("/events/history")
async def get_event_history(
    timestamp: float,
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """
    Retrieves the persisted event tick closest to a given timestamp.
    """
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized.")
    
    try:
        # This is a conceptual implementation. A real Ignite query would be more complex,
        # likely using a continuous query or a scan query with a custom filter.
        # For now, we simulate fetching a single record.
        # This assumes the ObservabilityManager's Ignite cache is accessible here.
        # A better design would be a dedicated service.
        
        # from managerQ.app.core.observability_manager import observability_manager
        # record = observability_manager._event_history_cache.get(timestamp, with_-binary=True)
        
        # MOCK IMPLEMENTATION:
        mock_record = {
            "type": "TICK",
            "payload": [
                {"event_type": "NODE_CREATED", "data": {"id": "agent_abc", "label": "Agent-DevOps", "type": "agent"}}
            ],
            "timestamp": timestamp
        }
        return json.loads(json.dumps(mock_record))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch event history: {e}") 