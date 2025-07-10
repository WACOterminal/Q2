import logging
from typing import Dict, Any, Optional, List
import httpx
from fastapi import HTTPException
import asyncio
import json
import base64
from datetime import datetime

from app.models.connector import BaseConnector, ConnectorAction
from app.core.vault_client import vault_client

logger = logging.getLogger(__name__)

class OnlyOfficeConnector(BaseConnector):
    """
    A connector for interacting with OnlyOffice Document Server API.
    
    Supported actions:
    - create_document: Create a new document
    - get_document: Get document information
    - get_document_content: Get document content
    - update_document: Update document content
    - delete_document: Delete a document
    - share_document: Share a document with users
    - get_document_history: Get document edit history
    - get_document_versions: Get document versions
    - restore_document_version: Restore a specific version
    - get_document_permissions: Get document permissions
    - set_document_permissions: Set document permissions
    - get_collaboration_info: Get collaboration information
    - add_comment: Add a comment to document
    - get_comments: Get document comments
    - resolve_comment: Resolve a comment
    - get_document_changes: Get document changes
    - accept_changes: Accept document changes
    - reject_changes: Reject document changes
    - convert_document: Convert document to another format
    - get_document_thumbnail: Get document thumbnail
    """

    @property
    def connector_id(self) -> str:
        return "onlyoffice"

    async def _get_client(self, credential_id: str) -> httpx.AsyncClient:
        """Helper to get an authenticated httpx client for OnlyOffice API."""
        credential = await vault_client.get_credential(credential_id)
        api_key = credential.secrets.get("api_key")
        base_url = credential.secrets.get("base_url")
        secret_key = credential.secrets.get("secret_key")  # For JWT signing

        if not all([api_key, base_url]):
            raise ValueError("OnlyOffice api_key and base_url must be provided in credential secrets.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Add JWT token if secret_key is provided
        if secret_key:
            headers["X-OnlyOffice-Secret"] = secret_key

        return httpx.AsyncClient(
            base_url=base_url.rstrip('/'),
            headers=headers,
            timeout=30.0
        )

    async def execute(self, action: ConnectorAction, configuration: Dict[str, Any], data_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not action.credential_id:
            raise ValueError("credential_id must be provided for OnlyOffice connector actions.")
        
        client = await self._get_client(action.credential_id)
        
        try:
            action_map = {
                "create_document": self._create_document,
                "get_document": self._get_document,
                "get_document_content": self._get_document_content,
                "update_document": self._update_document,
                "delete_document": self._delete_document,
                "share_document": self._share_document,
                "get_document_history": self._get_document_history,
                "get_document_versions": self._get_document_versions,
                "restore_document_version": self._restore_document_version,
                "get_document_permissions": self._get_document_permissions,
                "set_document_permissions": self._set_document_permissions,
                "get_collaboration_info": self._get_collaboration_info,
                "add_comment": self._add_comment,
                "get_comments": self._get_comments,
                "resolve_comment": self._resolve_comment,
                "get_document_changes": self._get_document_changes,
                "accept_changes": self._accept_changes,
                "reject_changes": self._reject_changes,
                "convert_document": self._convert_document,
                "get_document_thumbnail": self._get_document_thumbnail,
            }

            if action.action_id in action_map:
                func = action_map[action.action_id]
                return await func(client, configuration, data_context)
            else:
                raise ValueError(f"Unsupported action for OnlyOffice connector: {action.action_id}")

        except httpx.HTTPStatusError as e:
            logger.error(f"OnlyOffice API error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"An unexpected error occurred in OnlyOfficeConnector: {e}", exc_info=True)
            raise
        finally:
            await client.aclose()

    async def _create_document(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new document."""
        title = data.get("title")
        document_type = config.get("document_type", "docx")  # docx, xlsx, pptx
        template_id = config.get("template_id")
        
        if not title:
            raise ValueError("title is required.")

        payload = {
            "title": title,
            "filetype": document_type,
            "createdat": datetime.now().isoformat(),
        }
        
        if template_id:
            payload["templateid"] = template_id

        response = await client.post("/api/2.0/files", json=payload)
        response.raise_for_status()
        return response.json()

    async def _get_document(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get document information."""
        document_id = config.get("document_id")
        
        if not document_id:
            raise ValueError("document_id is required.")

        response = await client.get(f"/api/2.0/files/{document_id}")
        response.raise_for_status()
        return response.json()

    async def _get_document_content(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get document content."""
        document_id = config.get("document_id")
        
        if not document_id:
            raise ValueError("document_id is required.")

        response = await client.get(f"/api/2.0/files/{document_id}/content")
        response.raise_for_status()
        
        return {
            "content": response.content,
            "content_type": response.headers.get("content-type"),
            "size": len(response.content)
        }

    async def _update_document(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Update document content."""
        document_id = config.get("document_id")
        content = data.get("content")
        
        if not document_id or content is None:
            raise ValueError("document_id and content are required.")

        response = await client.put(f"/api/2.0/files/{document_id}/content", content=content)
        response.raise_for_status()
        return response.json()

    async def _delete_document(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a document."""
        document_id = config.get("document_id")
        
        if not document_id:
            raise ValueError("document_id is required.")

        response = await client.delete(f"/api/2.0/files/{document_id}")
        response.raise_for_status()
        return {"status": "success", "deleted": document_id}

    async def _share_document(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Share a document with users."""
        document_id = config.get("document_id")
        users = data.get("users", [])  # List of user IDs or emails
        permissions = config.get("permissions", "read")  # read, write, comment
        
        if not document_id or not users:
            raise ValueError("document_id and users are required.")

        payload = {
            "share": [
                {
                    "shareTo": user,
                    "permissions": permissions
                }
                for user in users
            ]
        }

        response = await client.post(f"/api/2.0/files/{document_id}/share", json=payload)
        response.raise_for_status()
        return response.json()

    async def _get_document_history(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get document edit history."""
        document_id = config.get("document_id")
        
        if not document_id:
            raise ValueError("document_id is required.")

        response = await client.get(f"/api/2.0/files/{document_id}/history")
        response.raise_for_status()
        return response.json()

    async def _get_document_versions(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get document versions."""
        document_id = config.get("document_id")
        
        if not document_id:
            raise ValueError("document_id is required.")

        response = await client.get(f"/api/2.0/files/{document_id}/versions")
        response.raise_for_status()
        return response.json()

    async def _restore_document_version(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Restore a specific document version."""
        document_id = config.get("document_id")
        version_id = config.get("version_id")
        
        if not document_id or not version_id:
            raise ValueError("document_id and version_id are required.")

        payload = {"version": version_id}
        response = await client.post(f"/api/2.0/files/{document_id}/restore", json=payload)
        response.raise_for_status()
        return response.json()

    async def _get_document_permissions(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get document permissions."""
        document_id = config.get("document_id")
        
        if not document_id:
            raise ValueError("document_id is required.")

        response = await client.get(f"/api/2.0/files/{document_id}/permissions")
        response.raise_for_status()
        return response.json()

    async def _set_document_permissions(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Set document permissions."""
        document_id = config.get("document_id")
        permissions = data.get("permissions")
        
        if not document_id or not permissions:
            raise ValueError("document_id and permissions are required.")

        response = await client.put(f"/api/2.0/files/{document_id}/permissions", json=permissions)
        response.raise_for_status()
        return response.json()

    async def _get_collaboration_info(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get collaboration information."""
        document_id = config.get("document_id")
        
        if not document_id:
            raise ValueError("document_id is required.")

        response = await client.get(f"/api/2.0/files/{document_id}/collaboration")
        response.raise_for_status()
        return response.json()

    async def _add_comment(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a comment to document."""
        document_id = config.get("document_id")
        comment_text = data.get("comment")
        author = data.get("author")
        
        if not document_id or not comment_text:
            raise ValueError("document_id and comment are required.")

        payload = {
            "text": comment_text,
            "author": author or "System",
            "time": datetime.now().isoformat()
        }

        response = await client.post(f"/api/2.0/files/{document_id}/comments", json=payload)
        response.raise_for_status()
        return response.json()

    async def _get_comments(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get document comments."""
        document_id = config.get("document_id")
        
        if not document_id:
            raise ValueError("document_id is required.")

        response = await client.get(f"/api/2.0/files/{document_id}/comments")
        response.raise_for_status()
        return response.json()

    async def _resolve_comment(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a comment."""
        document_id = config.get("document_id")
        comment_id = config.get("comment_id")
        
        if not document_id or not comment_id:
            raise ValueError("document_id and comment_id are required.")

        payload = {"resolved": True}
        response = await client.put(f"/api/2.0/files/{document_id}/comments/{comment_id}", json=payload)
        response.raise_for_status()
        return response.json()

    async def _get_document_changes(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get document changes."""
        document_id = config.get("document_id")
        
        if not document_id:
            raise ValueError("document_id is required.")

        response = await client.get(f"/api/2.0/files/{document_id}/changes")
        response.raise_for_status()
        return response.json()

    async def _accept_changes(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Accept document changes."""
        document_id = config.get("document_id")
        change_ids = data.get("change_ids", [])
        
        if not document_id:
            raise ValueError("document_id is required.")

        payload = {"changes": change_ids, "action": "accept"}
        response = await client.post(f"/api/2.0/files/{document_id}/changes", json=payload)
        response.raise_for_status()
        return response.json()

    async def _reject_changes(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Reject document changes."""
        document_id = config.get("document_id")
        change_ids = data.get("change_ids", [])
        
        if not document_id:
            raise ValueError("document_id is required.")

        payload = {"changes": change_ids, "action": "reject"}
        response = await client.post(f"/api/2.0/files/{document_id}/changes", json=payload)
        response.raise_for_status()
        return response.json()

    async def _convert_document(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert document to another format."""
        document_id = config.get("document_id")
        output_format = config.get("output_format", "pdf")  # pdf, docx, xlsx, pptx
        
        if not document_id:
            raise ValueError("document_id is required.")

        payload = {"outputtype": output_format}
        response = await client.post(f"/api/2.0/files/{document_id}/convert", json=payload)
        response.raise_for_status()
        return response.json()

    async def _get_document_thumbnail(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get document thumbnail."""
        document_id = config.get("document_id")
        size = config.get("size", "medium")  # small, medium, large
        
        if not document_id:
            raise ValueError("document_id is required.")

        response = await client.get(f"/api/2.0/files/{document_id}/thumbnail", params={"size": size})
        response.raise_for_status()
        
        return {
            "thumbnail": response.content,
            "content_type": response.headers.get("content-type"),
            "size": len(response.content)
        }


# Instantiate a single instance
onlyoffice_connector = OnlyOfficeConnector() 