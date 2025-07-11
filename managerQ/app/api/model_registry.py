# managerQ/app/api/model_registry.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional

# Import the new ModelRegistryService
from managerQ.app.core.model_registry_service import model_registry_service

router = APIRouter()

@router.post("/models/register")
async def register_model(
    model_name: str,
    version_id: str,
    artifact_path: str,
    metadata: Dict[str, Any],
    is_active: bool = False
):
    """
    Registers a new model version in the registry.
    """
    try:
        model_info = await model_registry_service.register_model_version(
            model_name=model_name,
            version_id=version_id,
            artifact_path=artifact_path,
            metadata=metadata,
            is_active=is_active
        )
        return {"status": "success", "message": "Model registered successfully", "model_info": model_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")

@router.get("/models/{model_name}/versions", response_model=List[Dict[str, Any]])
async def list_model_versions(model_name: str):
    """
    Lists all registered versions for a given model.
    """
    try:
        versions = await model_registry_service.list_model_versions(model_name)
        return versions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list model versions: {str(e)}")

@router.get("/models/{model_name}/{version_id}", response_model=Optional[Dict[str, Any]])
async def get_model_version(model_name: str, version_id: str):
    """
    Retrieves details for a specific model version.
    """
    try:
        model_version = await model_registry_service.get_model_version(model_name, version_id)
        if model_version is None:
            raise HTTPException(status_code=404, detail="Model version not found")
        return model_version
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model version: {str(e)}")

@router.post("/models/{model_name}/activate/{version_id}")
async def activate_model_version(model_name: str, version_id: str):
    """
    Activates a specific model version, making it the active model for inference.
    """
    try:
        success = await model_registry_service.activate_model_version(model_name, version_id)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to activate model version. Check if version exists.")
        return {"status": "success", "message": f"Model '{model_name}' version '{version_id}' activated."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate model version: {str(e)}")

@router.get("/models/{model_name}/active", response_model=Optional[Dict[str, Any]])
async def get_active_model(model_name: str):
    """
    Retrieves the currently active model version for a given model.
    """
    try:
        active_model = await model_registry_service.get_active_model(model_name)
        if active_model is None:
            raise HTTPException(status_code=404, detail="No active model found for this name.")
        return active_model
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active model: {str(e)}")

@router.get("/models/{model_name}/active/artifact-path", response_model=Optional[str])
async def get_active_model_artifact_path(model_name: str):
    """
    Retrieves the file path of the currently active model's artifact.
    """
    try:
        artifact_path = await model_registry_service.get_active_model_artifact_path(model_name)
        if artifact_path is None:
            raise HTTPException(status_code=404, detail="Artifact path not found for active model.")
        return artifact_path
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active model artifact path: {str(e)}") 