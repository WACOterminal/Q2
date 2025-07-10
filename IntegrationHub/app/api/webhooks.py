from fastapi import APIRouter, Request, HTTPException
import hmac
import hashlib
import logging
from shared.vault_client import VaultClient
from app.core.engine import engine
from app.models.flow import Flow
from app.api.flows import PREDEFINED_FLOWS

router = APIRouter()
logger = logging.getLogger(__name__)

def verify_github_signature(payload_body: bytes, secret_token: str, signature_header: str) -> bool:
    """Verify GitHub webhook signature"""
    if not signature_header:
        return False
    
    try:
        hash_object = hmac.new(secret_token.encode('utf-8'), payload_body, hashlib.sha256)
        expected_signature = "sha256=" + hash_object.hexdigest()
        return hmac.compare_digest(expected_signature, signature_header)
    except Exception:
        return False

def verify_gitlab_signature(payload_body: bytes, secret_token: str, signature_header: str) -> bool:
    """Verify GitLab webhook signature"""
    if not signature_header:
        return False
    
    try:
        return hmac.compare_digest(secret_token, signature_header)
    except Exception:
        return False

@router.post("/webhooks/github")
async def handle_github_webhook(request: Request):
    """
    Handles incoming webhooks from GitHub with signature verification.
    """
    logger.info("Received GitHub webhook")
    
    # Get headers
    signature_header = request.headers.get("X-Hub-Signature-256")
    event_type = request.headers.get("X-GitHub-Event")
    delivery_id = request.headers.get("X-GitHub-Delivery")
    
    # Get the payload
    payload_body = await request.body()
    
    # Verify webhook signature
    vault_client = VaultClient()
    try:
        webhook_secret_data = vault_client.read_secret("github-webhook-secret", "token")
        webhook_secret = webhook_secret_data if isinstance(webhook_secret_data, str) else webhook_secret_data.get("token")
        
        if not verify_github_signature(payload_body, webhook_secret, signature_header or ""):
            logger.warning(f"GitHub webhook signature verification failed for delivery {delivery_id}")
            raise HTTPException(status_code=403, detail="Signature verification failed")
    except Exception as e:
        logger.error(f"Failed to verify GitHub webhook signature: {e}")
        raise HTTPException(status_code=500, detail="Signature verification error")
    
    # Parse the JSON payload
    try:
        event = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse GitHub webhook payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    logger.info(f"GitHub webhook event: {event_type}, delivery: {delivery_id}")
    
    # Handle different event types
    if event_type == "pull_request":
        action = event.get("action")
        if action in ["opened", "synchronize"]:
            await handle_github_pr_event(event)
    elif event_type == "push":
        await handle_github_push_event(event)
    elif event_type == "issues":
        action = event.get("action")
        if action in ["opened", "edited"]:
            await handle_github_issue_event(event)
    
    return {"status": "ok", "event_type": event_type, "delivery_id": delivery_id}

@router.post("/webhooks/gitlab")
async def handle_gitlab_webhook(request: Request):
    """
    Handles incoming webhooks from GitLab with signature verification.
    """
    logger.info("Received GitLab webhook")
    
    # Get headers
    signature_header = request.headers.get("X-Gitlab-Token")
    event_type = request.headers.get("X-Gitlab-Event")
    
    # Get the payload
    payload_body = await request.body()
    
    # Verify webhook signature
    vault_client = VaultClient()
    try:
        webhook_secret_data = vault_client.read_secret("gitlab-webhook-secret", "token")
        webhook_secret = webhook_secret_data if isinstance(webhook_secret_data, str) else webhook_secret_data.get("token")
        
        if not verify_gitlab_signature(payload_body, webhook_secret, signature_header or ""):
            logger.warning(f"GitLab webhook signature verification failed for event {event_type}")
            raise HTTPException(status_code=403, detail="Signature verification failed")
    except Exception as e:
        logger.error(f"Failed to verify GitLab webhook signature: {e}")
        raise HTTPException(status_code=500, detail="Signature verification error")
    
    # Parse the JSON payload
    try:
        event = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse GitLab webhook payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    logger.info(f"GitLab webhook event: {event_type}")
    
    # Handle different event types
    if event_type == "Merge Request Hook":
        action = event.get("object_attributes", {}).get("action")
        if action in ["open", "update"]:
            await handle_gitlab_mr_event(event)
    elif event_type == "Push Hook":
        await handle_gitlab_push_event(event)
    elif event_type == "Issue Hook":
        action = event.get("object_attributes", {}).get("action")
        if action in ["open", "update"]:
            await handle_gitlab_issue_event(event)
    elif event_type == "Pipeline Hook":
        await handle_gitlab_pipeline_event(event)
    
    return {"status": "ok", "event_type": event_type}

async def handle_github_pr_event(event: dict):
    """Handle GitHub pull request events"""
    pr_data = event.get("pull_request", {})
    repo_data = event.get("repository", {})
    
    # Extract relevant information
    flow_context = {
        "repo": repo_data.get("full_name"),
        "pr_number": pr_data.get("number"),
        "pr_title": pr_data.get("title"),
        "pr_body": pr_data.get("body"),
        "author": pr_data.get("user", {}).get("login"),
        "webhook_payload": event
    }
    
    # Trigger code review flow
    if "code_review_agent" in PREDEFINED_FLOWS:
        flow_config = PREDEFINED_FLOWS["code_review_agent"]
        flow_model = Flow(**flow_config)
        await engine.run_flow(flow_model, flow_context)

async def handle_github_push_event(event: dict):
    """Handle GitHub push events"""
    repo_data = event.get("repository", {})
    
    flow_context = {
        "repo": repo_data.get("full_name"),
        "ref": event.get("ref"),
        "commits": event.get("commits", []),
        "webhook_payload": event
    }
    
    # Trigger push processing if there's a matching flow
    # Could be used for automatic documentation updates, etc.
    logger.info(f"GitHub push to {flow_context['repo']} on {flow_context['ref']}")

async def handle_github_issue_event(event: dict):
    """Handle GitHub issue events"""
    issue_data = event.get("issue", {})
    repo_data = event.get("repository", {})
    
    flow_context = {
        "repo": repo_data.get("full_name"),
        "issue_number": issue_data.get("number"),
        "issue_title": issue_data.get("title"),
        "issue_body": issue_data.get("body"),
        "author": issue_data.get("user", {}).get("login"),
        "webhook_payload": event
    }
    
    logger.info(f"GitHub issue {flow_context['issue_number']} in {flow_context['repo']}")

async def handle_gitlab_mr_event(event: dict):
    """Handle GitLab merge request events"""
    mr_data = event.get("object_attributes", {})
    project_data = event.get("project", {})
    
    flow_context = {
        "project_id": project_data.get("id"),
        "project_name": project_data.get("path_with_namespace"),
        "merge_request_iid": mr_data.get("iid"),
        "merge_request_title": mr_data.get("title"),
        "merge_request_description": mr_data.get("description"),
        "author": event.get("user", {}).get("username"),
        "webhook_payload": event
    }
    
    # Create GitLab code review flow similar to GitHub
    # This would need a new flow definition similar to code_review_agent
    logger.info(f"GitLab MR {flow_context['merge_request_iid']} in {flow_context['project_name']}")

async def handle_gitlab_push_event(event: dict):
    """Handle GitLab push events"""
    project_data = event.get("project", {})
    
    flow_context = {
        "project_id": project_data.get("id"),
        "project_name": project_data.get("path_with_namespace"),
        "ref": event.get("ref"),
        "commits": event.get("commits", []),
        "webhook_payload": event
    }
    
    logger.info(f"GitLab push to {flow_context['project_name']} on {flow_context['ref']}")

async def handle_gitlab_issue_event(event: dict):
    """Handle GitLab issue events"""
    issue_data = event.get("object_attributes", {})
    project_data = event.get("project", {})
    
    flow_context = {
        "project_id": project_data.get("id"),
        "project_name": project_data.get("path_with_namespace"),
        "issue_iid": issue_data.get("iid"),
        "issue_title": issue_data.get("title"),
        "issue_description": issue_data.get("description"),
        "author": event.get("user", {}).get("username"),
        "webhook_payload": event
    }
    
    logger.info(f"GitLab issue {flow_context['issue_iid']} in {flow_context['project_name']}")

async def handle_gitlab_pipeline_event(event: dict):
    """Handle GitLab pipeline events"""
    pipeline_data = event.get("object_attributes", {})
    project_data = event.get("project", {})
    
    flow_context = {
        "project_id": project_data.get("id"),
        "project_name": project_data.get("path_with_namespace"),
        "pipeline_id": pipeline_data.get("id"),
        "pipeline_status": pipeline_data.get("status"),
        "ref": pipeline_data.get("ref"),
        "webhook_payload": event
    }
    
    # Could trigger notifications or follow-up actions based on pipeline status
    logger.info(f"GitLab pipeline {flow_context['pipeline_id']} in {flow_context['project_name']}: {flow_context['pipeline_status']}")

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
        await engine.run_flow_by_id("jira-issue-created", event)
    elif event.get("webhookEvent") == "jira:issue_updated":
        await engine.run_flow_by_id("jira-issue-updated", event)
        
    return {"status": "ok"} 