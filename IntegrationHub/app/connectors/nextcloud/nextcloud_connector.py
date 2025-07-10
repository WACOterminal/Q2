import logging
from typing import Dict, Any, Optional, List
import httpx
from fastapi import HTTPException
import asyncio
import json
import base64
from urllib.parse import quote

from app.models.connector import BaseConnector, ConnectorAction
from app.core.vault_client import vault_client

logger = logging.getLogger(__name__)

class NextCloudConnector(BaseConnector):
    """
    A connector for interacting with the NextCloud API and WebDAV.
    
    Supported actions:
    - list_files: List files and folders in a directory
    - get_file: Download a file's contents
    - upload_file: Upload a file to NextCloud
    - create_folder: Create a new folder
    - delete_file: Delete a file or folder
    - share_file: Create a share link for a file
    - unshare_file: Remove a share link
    - get_shares: Get all shares for a file
    - get_user_info: Get user information
    - search_files: Search for files by name
    - get_file_info: Get file metadata
    - move_file: Move/rename a file
    - copy_file: Copy a file
    - set_file_permissions: Set file permissions
    - get_activities: Get activity feed
    - create_comment: Add comment to a file
    - get_comments: Get comments for a file
    """

    @property
    def connector_id(self) -> str:
        return "nextcloud"

    async def _get_client(self, credential_id: str) -> httpx.AsyncClient:
        """Helper to get an authenticated httpx client for NextCloud API."""
        credential = await vault_client.get_credential(credential_id)
        username = credential.secrets.get("username")
        password = credential.secrets.get("password")
        base_url = credential.secrets.get("base_url")

        if not all([username, password, base_url]):
            raise ValueError("NextCloud username, password, and base_url must be provided in credential secrets.")

        # Create basic auth header
        auth_string = f"{username}:{password}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')

        return httpx.AsyncClient(
            base_url=base_url.rstrip('/'),
            headers={
                "Authorization": f"Basic {auth_b64}",
                "Content-Type": "application/json",
                "OCS-APIRequest": "true"
            },
            timeout=30.0
        )

    async def _get_webdav_client(self, credential_id: str) -> httpx.AsyncClient:
        """Helper to get an authenticated httpx client for NextCloud WebDAV."""
        credential = await vault_client.get_credential(credential_id)
        username = credential.secrets.get("username")
        password = credential.secrets.get("password")
        base_url = credential.secrets.get("base_url")

        if not all([username, password, base_url]):
            raise ValueError("NextCloud username, password, and base_url must be provided in credential secrets.")

        auth_string = f"{username}:{password}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')

        return httpx.AsyncClient(
            base_url=f"{base_url.rstrip('/')}/remote.php/dav/files/{username}",
            headers={
                "Authorization": f"Basic {auth_b64}",
            },
            timeout=30.0
        )

    async def execute(self, action: ConnectorAction, configuration: Dict[str, Any], data_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not action.credential_id:
            raise ValueError("credential_id must be provided for NextCloud connector actions.")
        
        try:
            action_map = {
                "list_files": self._list_files,
                "get_file": self._get_file,
                "upload_file": self._upload_file,
                "create_folder": self._create_folder,
                "delete_file": self._delete_file,
                "share_file": self._share_file,
                "unshare_file": self._unshare_file,
                "get_shares": self._get_shares,
                "get_user_info": self._get_user_info,
                "search_files": self._search_files,
                "get_file_info": self._get_file_info,
                "move_file": self._move_file,
                "copy_file": self._copy_file,
                "set_file_permissions": self._set_file_permissions,
                "get_activities": self._get_activities,
                "create_comment": self._create_comment,
                "get_comments": self._get_comments,
            }

            if action.action_id in action_map:
                func = action_map[action.action_id]
                return await func(action.credential_id, configuration, data_context)
            else:
                raise ValueError(f"Unsupported action for NextCloud connector: {action.action_id}")

        except httpx.HTTPStatusError as e:
            logger.error(f"NextCloud API error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"An unexpected error occurred in NextCloudConnector: {e}", exc_info=True)
            raise

    async def _list_files(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """List files and folders in a directory using WebDAV."""
        path = config.get("path", "/")
        client = await self._get_webdav_client(credential_id)
        
        try:
            response = await client.request(
                "PROPFIND",
                f"/{path.lstrip('/')}",
                headers={"Depth": "1"},
                content='<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:getcontenttype/><d:getlastmodified/><d:getcontentlength/><d:displayname/></d:prop></d:propfind>'
            )
            response.raise_for_status()
            
            # Parse WebDAV response (simplified)
            return {"status": "success", "files": response.text}
        finally:
            await client.aclose()

    async def _get_file(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Download a file's contents."""
        file_path = config.get("file_path")
        if not file_path:
            raise ValueError("file_path is required.")
            
        client = await self._get_webdav_client(credential_id)
        
        try:
            response = await client.get(f"/{file_path.lstrip('/')}")
            response.raise_for_status()
            
            return {
                "content": response.content,
                "content_type": response.headers.get("content-type"),
                "size": len(response.content)
            }
        finally:
            await client.aclose()

    async def _upload_file(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Upload a file to NextCloud."""
        file_path = config.get("file_path")
        content = data.get("content")
        
        if not file_path or content is None:
            raise ValueError("file_path and content are required.")
            
        client = await self._get_webdav_client(credential_id)
        
        try:
            response = await client.put(f"/{file_path.lstrip('/')}", content=content)
            response.raise_for_status()
            
            return {"status": "success", "file_path": file_path}
        finally:
            await client.aclose()

    async def _create_folder(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new folder."""
        folder_path = config.get("folder_path")
        if not folder_path:
            raise ValueError("folder_path is required.")
            
        client = await self._get_webdav_client(credential_id)
        
        try:
            response = await client.request("MKCOL", f"/{folder_path.lstrip('/')}")
            response.raise_for_status()
            
            return {"status": "success", "folder_path": folder_path}
        finally:
            await client.aclose()

    async def _delete_file(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a file or folder."""
        file_path = config.get("file_path")
        if not file_path:
            raise ValueError("file_path is required.")
            
        client = await self._get_webdav_client(credential_id)
        
        try:
            response = await client.delete(f"/{file_path.lstrip('/')}")
            response.raise_for_status()
            
            return {"status": "success", "deleted": file_path}
        finally:
            await client.aclose()

    async def _share_file(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a share link for a file."""
        file_path = config.get("file_path")
        share_type = config.get("share_type", 3)  # 3 = public link
        permissions = config.get("permissions", 1)  # 1 = read
        
        if not file_path:
            raise ValueError("file_path is required.")
            
        client = await self._get_client(credential_id)
        
        try:
            response = await client.post(
                "/ocs/v2.php/apps/files_sharing/api/v1/shares",
                data={
                    "path": file_path,
                    "shareType": share_type,
                    "permissions": permissions
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            
            return response.json()
        finally:
            await client.aclose()

    async def _unshare_file(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a share link."""
        share_id = config.get("share_id")
        if not share_id:
            raise ValueError("share_id is required.")
            
        client = await self._get_client(credential_id)
        
        try:
            response = await client.delete(f"/ocs/v2.php/apps/files_sharing/api/v1/shares/{share_id}")
            response.raise_for_status()
            
            return {"status": "success", "share_id": share_id}
        finally:
            await client.aclose()

    async def _get_shares(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get all shares for a file."""
        file_path = config.get("file_path")
        
        client = await self._get_client(credential_id)
        
        try:
            params = {}
            if file_path:
                params["path"] = file_path
                
            response = await client.get("/ocs/v2.php/apps/files_sharing/api/v1/shares", params=params)
            response.raise_for_status()
            
            return response.json()
        finally:
            await client.aclose()

    async def _get_user_info(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get user information."""
        client = await self._get_client(credential_id)
        
        try:
            response = await client.get("/ocs/v2.php/cloud/user")
            response.raise_for_status()
            
            return response.json()
        finally:
            await client.aclose()

    async def _search_files(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Search for files by name."""
        query = config.get("query")
        if not query:
            raise ValueError("query is required.")
            
        client = await self._get_client(credential_id)
        
        try:
            response = await client.get(f"/ocs/v2.php/apps/files_sharing/api/v1/search", params={"query": query})
            response.raise_for_status()
            
            return response.json()
        finally:
            await client.aclose()

    async def _get_file_info(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get file metadata."""
        file_path = config.get("file_path")
        if not file_path:
            raise ValueError("file_path is required.")
            
        client = await self._get_webdav_client(credential_id)
        
        try:
            response = await client.request(
                "PROPFIND",
                f"/{file_path.lstrip('/')}",
                headers={"Depth": "0"},
                content='<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:getcontenttype/><d:getlastmodified/><d:getcontentlength/><d:displayname/></d:prop></d:propfind>'
            )
            response.raise_for_status()
            
            return {"status": "success", "metadata": response.text}
        finally:
            await client.aclose()

    async def _move_file(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Move/rename a file."""
        source_path = config.get("source_path")
        destination_path = config.get("destination_path")
        
        if not source_path or not destination_path:
            raise ValueError("source_path and destination_path are required.")
            
        client = await self._get_webdav_client(credential_id)
        
        try:
            response = await client.request(
                "MOVE",
                f"/{source_path.lstrip('/')}",
                headers={"Destination": f"/{destination_path.lstrip('/')}"}
            )
            response.raise_for_status()
            
            return {"status": "success", "moved_from": source_path, "moved_to": destination_path}
        finally:
            await client.aclose()

    async def _copy_file(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Copy a file."""
        source_path = config.get("source_path")
        destination_path = config.get("destination_path")
        
        if not source_path or not destination_path:
            raise ValueError("source_path and destination_path are required.")
            
        client = await self._get_webdav_client(credential_id)
        
        try:
            response = await client.request(
                "COPY",
                f"/{source_path.lstrip('/')}",
                headers={"Destination": f"/{destination_path.lstrip('/')}"}
            )
            response.raise_for_status()
            
            return {"status": "success", "copied_from": source_path, "copied_to": destination_path}
        finally:
            await client.aclose()

    async def _set_file_permissions(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Set file permissions."""
        share_id = config.get("share_id")
        permissions = config.get("permissions")
        
        if not share_id or permissions is None:
            raise ValueError("share_id and permissions are required.")
            
        client = await self._get_client(credential_id)
        
        try:
            response = await client.put(
                f"/ocs/v2.php/apps/files_sharing/api/v1/shares/{share_id}",
                data={"permissions": permissions},
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            
            return response.json()
        finally:
            await client.aclose()

    async def _get_activities(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get activity feed."""
        limit = config.get("limit", 50)
        
        client = await self._get_client(credential_id)
        
        try:
            response = await client.get(f"/ocs/v2.php/apps/activity/api/v2/activity", params={"limit": limit})
            response.raise_for_status()
            
            return response.json()
        finally:
            await client.aclose()

    async def _create_comment(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Add comment to a file."""
        file_id = config.get("file_id")
        message = data.get("message")
        
        if not file_id or not message:
            raise ValueError("file_id and message are required.")
            
        client = await self._get_client(credential_id)
        
        try:
            response = await client.post(
                f"/ocs/v2.php/apps/comments/api/v1/comments",
                data={
                    "objectType": "files",
                    "objectId": file_id,
                    "message": message
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            
            return response.json()
        finally:
            await client.aclose()

    async def _get_comments(self, credential_id: str, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comments for a file."""
        file_id = config.get("file_id")
        
        if not file_id:
            raise ValueError("file_id is required.")
            
        client = await self._get_client(credential_id)
        
        try:
            response = await client.get(f"/ocs/v2.php/apps/comments/api/v1/comments", params={
                "objectType": "files",
                "objectId": file_id
            })
            response.raise_for_status()
            
            return response.json()
        finally:
            await client.aclose()


# Instantiate a single instance
nextcloud_connector = NextCloudConnector() 