from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import os

from .sandbox_manager import sandbox_manager, SANDBOX_WORKSPACE_DIR

router = APIRouter()

class Sandbox(BaseModel):
    id: str
    status: str

class ExecutionRequest(BaseModel):
    command: str

class ExecutionResponse(BaseModel):
    exit_code: int
    output: str

class SandboxCreateRequest(BaseModel):
    network_enabled: bool = False

@router.post("/sandboxes", response_model=Sandbox, status_code=201)
def create_sandbox(request: SandboxCreateRequest):
    """
    Creates a new sandbox environment.
    """
    container = sandbox_manager.create_sandbox(network_enabled=request.network_enabled)
    if not container.id:
        raise HTTPException(status_code=500, detail="Failed to create sandbox container")
    return Sandbox(id=container.id, status=container.status)

@router.post("/sandboxes/{container_id}/execute", response_model=ExecutionResponse)
def execute_command(container_id: str, request: ExecutionRequest):
    """
    Executes a command in a specific sandbox.
    """
    container = sandbox_manager.get_sandbox(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    exit_code, output = sandbox_manager.execute_in_sandbox(container, request.command)
    return ExecutionResponse(exit_code=exit_code, output=output)

@router.post("/sandboxes/{container_id}/files/upload", status_code=204)
async def upload_file(container_id: str, file: UploadFile = File(...), path: str = ""):
    """
    Uploads a file to the sandbox's workspace.
    """
    container = sandbox_manager.get_sandbox(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    try:
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name
        
        filename = file.filename if file.filename else "uploaded_file"
        sandbox_manager.upload_file_to_sandbox(container, temp_path, os.path.join(path, filename))
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)

@router.get("/sandboxes/{container_id}/files/download")
async def download_file(container_id: str, path: str):
    """
    Downloads a file from the sandbox's workspace.
    """
    container = sandbox_manager.get_sandbox(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Sandbox not found")
        
    workspace_id = container.labels.get('qagi_sandbox_workspace_id')
    if not workspace_id:
        raise HTTPException(status_code=500, detail="Invalid sandbox state.")

    host_path = os.path.join(SANDBOX_WORKSPACE_DIR, workspace_id, path)
    
    if not os.path.isfile(host_path):
        raise HTTPException(status_code=404, detail="File not found in sandbox")

    return FileResponse(host_path, filename=os.path.basename(path))

@router.delete("/sandboxes/{container_id}", status_code=204)
def remove_sandbox(container_id: str):
    """
    Stops and removes a specific sandbox.
    """
    container = sandbox_manager.get_sandbox(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    sandbox_manager.remove_sandbox(container)
    return 