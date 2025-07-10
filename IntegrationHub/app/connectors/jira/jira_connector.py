from typing import Optional
from IntegrationHub.app.core.base_connector import BaseConnector
from shared.vault_client import VaultClient
import httpx

class JiraConnector(BaseConnector):
    """
    A connector for interacting with the Jira API.
    """
    def __init__(self, config, vault_client: VaultClient):
        super().__init__(config)
        self.base_url = self.config.get("url")
        self.vault_client = vault_client
        self.auth = None

    async def _get_auth(self):
        if not self.auth:
            # The path to the secret in Vault, e.g., 'secret/data/jira'
            # This would be configured elsewhere, but we'll hardcode it for now.
            credentials = self.vault_client.read_secret_data(path="secret/data/jira")
            self.auth = (credentials['username'], credentials['api_token'])
        return self.auth

    async def create_issue(self, project_key: str, summary: str, description: str, issue_type: str = "Task"):
        """
        Creates a new issue in Jira.
        """
        auth = await self._get_auth()
        url = f"{self.base_url}/rest/api/2/issue"
        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": description,
                "issuetype": {"name": issue_type}
            }
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, auth=auth)
            response.raise_for_status()
            return response.json()

    async def update_issue(self, issue_key: str, summary: Optional[str] = None, description: Optional[str] = None, status: Optional[str] = None):
        """
        Updates an existing issue in Jira.
        """
        auth = await self._get_auth()
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
        fields = {}
        if summary:
            fields["summary"] = summary
        if description:
            fields["description"] = description
        
        payload = {"fields": fields}
        
        async with httpx.AsyncClient() as client:
            response = await client.put(url, json=payload, auth=auth)
            response.raise_for_status()
        
        if status:
            await self._transition_issue(issue_key, status)

    async def _transition_issue(self, issue_key: str, status_name: str):
        auth = await self._get_auth()
        # First, get available transitions
        transitions_url = f"{self.base_url}/rest/api/2/issue/{issue_key}/transitions"
        async with httpx.AsyncClient() as client:
            response = await client.get(transitions_url, auth=auth)
            response.raise_for_status()
            transitions = response.json().get("transitions", [])
        
        # Find the transition ID for the desired status
        transition_id = None
        for t in transitions:
            if t['name'].lower() == status_name.lower():
                transition_id = t['id']
                break
        
        if not transition_id:
            raise ValueError(f"Status '{status_name}' is not a valid transition for issue {issue_key}")
            
        # Perform the transition
        payload = {"transition": {"id": transition_id}}
        await client.post(transitions_url, json=payload, auth=auth) 