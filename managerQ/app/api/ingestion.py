import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
from managerQ.app.core.workflow_executor import workflow_executor
from shared.q_auth_parser import get_current_user, User

router = APIRouter()
logger = structlog.get_logger(__name__)

class WebIngestRequest(BaseModel):
    url: HttpUrl

@router.post("/web", status_code=202)
async def ingest_from_web(
    request: WebIngestRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Accepts a URL and triggers a workflow to ingest its content.
    """
    logger.info("Received web ingestion request", url=str(request.url), user_id=current_user.id)
    
    try:
        # The context that will be passed to the workflow
        context = {
            "source_url": str(request.url),
            "requesting_user_id": current_user.id
        }
        
        # Trigger the workflow and forget. The workflow will run in the background.
        workflow_executor.run_workflow(
            workflow_id="wf_live_data_ingestion",
            prompt_template="Ingest data from URL: {{ source_url }}",
            context=context,
        )

        return {"message": "Ingestion workflow started successfully.", "url": str(request.url)}

    except Exception as e:
        logger.error("Failed to start ingestion workflow", exc_info=True)
        # We use a generic error to avoid leaking implementation details
        raise HTTPException(status_code=500, detail="Internal server error while starting ingestion task.") 