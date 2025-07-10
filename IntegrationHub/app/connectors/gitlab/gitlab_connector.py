import logging
from typing import Dict, Any, Optional, List
import httpx
from fastapi import HTTPException
import asyncio
import json

from app.models.connector import BaseConnector, ConnectorAction
from app.core.vault_client import vault_client

logger = logging.getLogger(__name__)

class GitLabConnector(BaseConnector):
    """
    A connector for interacting with the GitLab API.
    
    Supported actions:
    - create_issue: Create a new issue
    - update_issue: Update an existing issue  
    - get_issue: Get issue details
    - list_issues: List issues in a project
    - create_merge_request: Create a merge request
    - update_merge_request: Update a merge request
    - get_merge_request: Get merge request details
    - list_merge_requests: List merge requests
    - create_comment: Add comment to issue/MR
    - get_project_info: Get project information
    - get_file_contents: Get file contents from repository
    - create_branch: Create a new branch
    - create_commit: Create a commit
    - trigger_pipeline: Trigger a CI/CD pipeline
    - get_pipeline_status: Get pipeline status
    """

    @property
    def connector_id(self) -> str:
        return "gitlab"

    async def _get_client(self, credential_id: str) -> httpx.AsyncClient:
        """Helper to get an authenticated httpx client for GitLab API."""
        credential = await vault_client.get_credential(credential_id)
        access_token = credential.secrets.get("access_token")
        base_url = credential.secrets.get("base_url", "https://gitlab.com")

        if not access_token:
            raise ValueError("GitLab access token not found in credential secrets.")

        return httpx.AsyncClient(
            base_url=f"{base_url}/api/v4",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )

    async def execute(self, action: ConnectorAction, configuration: Dict[str, Any], data_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not action.credential_id:
            raise ValueError("credential_id must be provided for GitLab connector actions.")
        
        client = await self._get_client(action.credential_id)
        
        try:
            action_map = {
                "create_issue": self._create_issue,
                "update_issue": self._update_issue,
                "get_issue": self._get_issue,
                "list_issues": self._list_issues,
                "create_merge_request": self._create_merge_request,
                "update_merge_request": self._update_merge_request,
                "get_merge_request": self._get_merge_request,
                "list_merge_requests": self._list_merge_requests,
                "create_comment": self._create_comment,
                "get_project_info": self._get_project_info,
                "get_file_contents": self._get_file_contents,
                "create_branch": self._create_branch,
                "create_commit": self._create_commit,
                "trigger_pipeline": self._trigger_pipeline,
                "get_pipeline_status": self._get_pipeline_status,
            }

            if action.action_id in action_map:
                func = action_map[action.action_id]
                return await func(client, configuration, data_context)
            else:
                raise ValueError(f"Unsupported action for GitLab connector: {action.action_id}")

        except httpx.HTTPStatusError as e:
            logger.error(f"GitLab API error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"An unexpected error occurred in GitLabConnector: {e}", exc_info=True)
            raise
        finally:
            await client.aclose()

    async def _create_issue(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new issue in GitLab."""
        project_id = config.get("project_id")
        title = data.get("title")
        description = data.get("description", "")
        labels = data.get("labels", [])
        assignee_id = data.get("assignee_id")
        
        if not project_id or not title:
            raise ValueError("project_id and title are required.")

        payload = {
            "title": title,
            "description": description,
            "labels": labels if isinstance(labels, list) else [labels]
        }
        
        if assignee_id:
            payload["assignee_id"] = assignee_id

        response = await client.post(f"/projects/{project_id}/issues", json=payload)
        response.raise_for_status()
        return response.json()

    async def _update_issue(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing issue in GitLab."""
        project_id = config.get("project_id")
        issue_iid = config.get("issue_iid")
        
        if not project_id or not issue_iid:
            raise ValueError("project_id and issue_iid are required.")

        payload = {}
        if "title" in data:
            payload["title"] = data["title"]
        if "description" in data:
            payload["description"] = data["description"]
        if "labels" in data:
            payload["labels"] = data["labels"]
        if "state_event" in data:
            payload["state_event"] = data["state_event"]  # 'close' or 'reopen'

        response = await client.put(f"/projects/{project_id}/issues/{issue_iid}", json=payload)
        response.raise_for_status()
        return response.json()

    async def _get_issue(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get details of a specific issue."""
        project_id = config.get("project_id")
        issue_iid = config.get("issue_iid")
        
        if not project_id or not issue_iid:
            raise ValueError("project_id and issue_iid are required.")

        response = await client.get(f"/projects/{project_id}/issues/{issue_iid}")
        response.raise_for_status()
        return response.json()

    async def _list_issues(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """List issues in a project."""
        project_id = config.get("project_id")
        state = config.get("state", "opened")  # opened, closed, all
        labels = config.get("labels")
        
        if not project_id:
            raise ValueError("project_id is required.")

        params = {"state": state}
        if labels:
            params["labels"] = labels

        response = await client.get(f"/projects/{project_id}/issues", params=params)
        response.raise_for_status()
        return {"issues": response.json()}

    async def _create_merge_request(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new merge request."""
        project_id = config.get("project_id")
        title = data.get("title")
        description = data.get("description", "")
        source_branch = data.get("source_branch")
        target_branch = data.get("target_branch", "main")
        
        if not all([project_id, title, source_branch]):
            raise ValueError("project_id, title, and source_branch are required.")

        payload = {
            "title": title,
            "description": description,
            "source_branch": source_branch,
            "target_branch": target_branch
        }

        response = await client.post(f"/projects/{project_id}/merge_requests", json=payload)
        response.raise_for_status()
        return response.json()

    async def _update_merge_request(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing merge request."""
        project_id = config.get("project_id")
        merge_request_iid = config.get("merge_request_iid")
        
        if not project_id or not merge_request_iid:
            raise ValueError("project_id and merge_request_iid are required.")

        payload = {}
        if "title" in data:
            payload["title"] = data["title"]
        if "description" in data:
            payload["description"] = data["description"]
        if "state_event" in data:
            payload["state_event"] = data["state_event"]  # 'close', 'reopen', 'merge'

        response = await client.put(f"/projects/{project_id}/merge_requests/{merge_request_iid}", json=payload)
        response.raise_for_status()
        return response.json()

    async def _get_merge_request(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get details of a specific merge request."""
        project_id = config.get("project_id")
        merge_request_iid = config.get("merge_request_iid")
        
        if not project_id or not merge_request_iid:
            raise ValueError("project_id and merge_request_iid are required.")

        response = await client.get(f"/projects/{project_id}/merge_requests/{merge_request_iid}")
        response.raise_for_status()
        return response.json()

    async def _list_merge_requests(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """List merge requests in a project."""
        project_id = config.get("project_id")
        state = config.get("state", "opened")  # opened, closed, merged, all
        
        if not project_id:
            raise ValueError("project_id is required.")

        params = {"state": state}
        response = await client.get(f"/projects/{project_id}/merge_requests", params=params)
        response.raise_for_status()
        return {"merge_requests": response.json()}

    async def _create_comment(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comment on an issue or merge request."""
        project_id = config.get("project_id")
        resource_type = config.get("resource_type")  # "issues" or "merge_requests"
        resource_iid = config.get("resource_iid")
        body = data.get("body")
        
        if not all([project_id, resource_type, resource_iid, body]):
            raise ValueError("project_id, resource_type, resource_iid, and body are required.")

        payload = {"body": body}
        response = await client.post(f"/projects/{project_id}/{resource_type}/{resource_iid}/notes", json=payload)
        response.raise_for_status()
        return response.json()

    async def _get_project_info(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get project information."""
        project_id = config.get("project_id")
        
        if not project_id:
            raise ValueError("project_id is required.")

        response = await client.get(f"/projects/{project_id}")
        response.raise_for_status()
        return response.json()

    async def _get_file_contents(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get file contents from repository."""
        project_id = config.get("project_id")
        file_path = config.get("file_path")
        ref = config.get("ref", "main")
        
        if not project_id or not file_path:
            raise ValueError("project_id and file_path are required.")

        response = await client.get(f"/projects/{project_id}/repository/files/{file_path}/raw", params={"ref": ref})
        response.raise_for_status()
        return {"content": response.text}

    async def _create_branch(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new branch."""
        project_id = config.get("project_id")
        branch_name = data.get("branch_name")
        ref = data.get("ref", "main")
        
        if not project_id or not branch_name:
            raise ValueError("project_id and branch_name are required.")

        payload = {
            "branch": branch_name,
            "ref": ref
        }

        response = await client.post(f"/projects/{project_id}/repository/branches", json=payload)
        response.raise_for_status()
        return response.json()

    async def _create_commit(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a commit with file changes."""
        project_id = config.get("project_id")
        branch = data.get("branch")
        commit_message = data.get("commit_message")
        actions = data.get("actions", [])  # List of file actions
        
        if not all([project_id, branch, commit_message, actions]):
            raise ValueError("project_id, branch, commit_message, and actions are required.")

        payload = {
            "branch": branch,
            "commit_message": commit_message,
            "actions": actions
        }

        response = await client.post(f"/projects/{project_id}/repository/commits", json=payload)
        response.raise_for_status()
        return response.json()

    async def _trigger_pipeline(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a CI/CD pipeline."""
        project_id = config.get("project_id")
        ref = data.get("ref", "main")
        variables = data.get("variables", {})
        
        if not project_id:
            raise ValueError("project_id is required.")

        payload = {"ref": ref}
        if variables:
            payload["variables"] = variables

        response = await client.post(f"/projects/{project_id}/pipeline", json=payload)
        response.raise_for_status()
        return response.json()

    async def _get_pipeline_status(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get pipeline status."""
        project_id = config.get("project_id")
        pipeline_id = config.get("pipeline_id")
        
        if not project_id or not pipeline_id:
            raise ValueError("project_id and pipeline_id are required.")

        response = await client.get(f"/projects/{project_id}/pipelines/{pipeline_id}")
        response.raise_for_status()
        return response.json()


# Instantiate a single instance
gitlab_connector = GitLabConnector() 