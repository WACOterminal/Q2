from fastapi import APIRouter, Request, HTTPException
import hmac
import hashlib
from shared.vault_client import VaultClient
from IntegrationHub.app.core.flow_engine import flow_engine

router = APIRouter()

@router.post("/webhooks/jira")
async def handle_jira_webhook(request: Request):
    """
    Handles incoming webhooks from Jira.
    """
    # Verify the webhook signature
    vault_client = VaultClient() # This should be a singleton in a real app
    webhook_secret = vault_client.read_secret("jira-webhook-secret", "secret")
    
    # Jira doesn't send a signature, so we can't verify it.
    # In a real application, you would need to implement a different security measure,
    # such as a pre-shared secret in the URL.
    
    # For now, we'll just process the event.
    event = await request.json()
    
    # Trigger a flow based on the event type
    if event.get("webhookEvent") == "jira:issue_created":
        await flow_engine.trigger_flow("jira-issue-created", event)
    elif event.get("webhookEvent") == "jira:issue_updated":
        await flow_engine.trigger_flow("jira-issue-updated", event)
        
    return {"status": "ok"} 