
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# This is a conceptual import, assuming the connector is registered in a central place.
from ..connectors.openproject.openproject_connector import openproject_connector
from ..core.engine import engine

logger = logging.getLogger(__name__)
router = APIRouter()

class CreateGoalPayload(BaseModel):
    subject: str
    project_id: int
    parent_id: Optional[int] = None

@router.get("/work-packages", response_model=List[Dict[str, Any]])
async def list_work_packages(project_id: int = 1): # Default to project 1 for now
    """
    Fetches a list of work packages from OpenProject.
    """
    try:
        # We use the engine to run the connector's action
        result = await engine.run_action(
            connector_id="openproject",
            action_id="list_work_packages",
            credential_id="openproject-credentials", # This must be configured in Vault
            configuration={"project_id": project_id},
            data_context={}
        )
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch work packages from OpenProject: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch work packages.")

@router.get("/work-packages/{work_package_id}", response_model=Dict[str, Any])
async def get_work_package(work_package_id: int):
    """
    Fetches details for a single work package from OpenProject.
    """
    try:
        result = await engine.run_action(
            connector_id="openproject",
            action_id="get_work_package",
            credential_id="openproject-credentials",
            configuration={"work_package_id": work_package_id},
            data_context={}
        )
        return result
    except Exception as e:
        logger.error(f"Failed to fetch work package {work_package_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch work package {work_package_id}.")


@router.post("/work-packages", response_model=Dict[str, Any])
async def create_work_package(payload: CreateGoalPayload):
    """
    Creates a new work package (Goal) in OpenProject.
    """
    try:
        result = await engine.run_action(
            connector_id="openproject",
            action_id="create_work_package",
            credential_id="openproject-credentials", # This must be configured in Vault
            configuration={"project_id": payload.project_id},
            data_context={
                "subject": payload.subject,
                "parent_id": payload.parent_id
            }
        )
        return result
    except Exception as e:
        logger.error(f"Failed to create work package in OpenProject: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create work package.") 