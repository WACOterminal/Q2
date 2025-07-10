
import logging
from typing import Dict, Any, Optional, List
import httpx
from fastapi import HTTPException
import asyncio

from app.models.connector import BaseConnector, ConnectorAction
from app.core.vault_client import vault_client

logger = logging.getLogger(__name__)

class OpenProjectConnector(BaseConnector):
    """
    A connector for interacting with the OpenProject API.
    """

    @property
    def connector_id(self) -> str:
        return "openproject"

    async def _get_client(self, credential_id: str) -> httpx.AsyncClient:
        """
        Helper to get an authenticated httpx client for the OpenProject API.
        """
        credential = await vault_client.get_credential(credential_id)
        api_key = credential.secrets.get("api_key")
        base_url = credential.secrets.get("base_url")

        if not api_key or not base_url:
            raise ValueError("OpenProject API key or base URL not found in credential secrets.")

        return httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Basic {api_key}", # OpenProject uses API key as username, no password
                "Content-Type": "application/json"
            },
            timeout=30.0
        )

    async def execute(self, action: ConnectorAction, configuration: Dict[str, Any], data_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not action.credential_id:
            raise ValueError("credential_id must be provided for OpenProject connector actions.")
        
        client = await self._get_client(action.credential_id)
        
        try:
            action_map = {
                "update_work_package": self._update_work_package,
                "list_work_packages": self._list_work_packages,
                "create_work_package": self._create_work_package,
                "get_work_package": self._get_work_package,
            }

            if action.action_id in action_map:
                func = action_map[action.action_id]
                return await func(client, configuration, data_context)
            else:
                raise ValueError(f"Unsupported action for OpenProject connector: {action.action_id}")

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenProject API error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
        except Exception as e:
            logger.error(f"An unexpected error occurred in OpenProjectConnector: {e}", exc_info=True)
            raise
        finally:
            await client.aclose()
            
    async def _get_work_package(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetches a single work package by its ID.
        """
        work_package_id = config.get("work_package_id")
        if not work_package_id:
            raise ValueError("work_package_id is required.")
            
        response = await client.get(f"/api/v3/work_packages/{work_package_id}")
        response.raise_for_status()
        return response.json()

    async def _list_work_packages(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetches all work packages for a given project.
        """
        project_id = config.get("project_id")
        if not project_id:
            raise ValueError("project_id is required to list work packages.")
            
        # The OpenProject API uses a HAL+JSON format, so we need to parse the _embedded element.
        response = await client.get(f"/api/v3/projects/{project_id}/work_packages")
        response.raise_for_status()
        return response.json().get("_embedded", {}).get("elements", [])

    async def _update_work_package(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates a work package in OpenProject by adding a comment.
        """
        work_package_id = config.get("work_package_id")
        comment = data.get("comment_body")
        
        if not work_package_id or not comment:
            raise ValueError("work_package_id and comment_body are required.")

        # OpenProject requires comments to be posted to a separate notifications endpoint
        comment_payload = {
            "comment": {
                "raw": comment
            }
        }
        
        response = await client.post(f"/api/v3/work_packages/{work_package_id}/activities", json=comment_payload)
        response.raise_for_status()
        
        return response.json()

    async def _create_work_package(self, client: httpx.AsyncClient, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a new work package (a 'Goal') in a given project.
        Can optionally link it to a parent work package.
        """
        project_id = config.get("project_id")
        subject = data.get("subject")
        parent_id = data.get("parent_id") # New optional parameter
        type_id = config.get("type_id", 1) # Default to 'Task' type, should be configured

        if not project_id or not subject:
            raise ValueError("project_id and subject are required.")

        package_payload = {
            "subject": subject,
            "_links": {
                "type": {
                    "href": f"/api/v3/types/{type_id}"
                },
                "project": {
                    "href": f"/api/v3/projects/{project_id}"
                }
            }
        }
        
        # If a parent ID is provided, add it to the payload to create a child task
        if parent_id:
            package_payload["_links"]["parent"] = {
                "href": f"/api/v3/work_packages/{parent_id}"
            }
        
        response = await client.post("/api/v3/work_packages", json=package_payload)
        response.raise_for_status()
        return response.json()


# Instantiate a single instance
openproject_connector = OpenProjectConnector() 