import httpx
from typing import Optional

class AgentSandboxClient:
    def __init__(self, base_url: str):
        self.base_url = f"{base_url}/api/v1"
        self.client = httpx.AsyncClient(timeout=300.0) # Long timeout for commands

    async def create_sandbox(self, network_enabled: bool = False) -> Optional[str]:
        """Creates a new sandbox and returns its ID."""
        try:
            response = await self.client.post(
                f"{self.base_url}/sandboxes",
                json={"network_enabled": network_enabled}
            )
            response.raise_for_status()
            return response.json().get("id")
        except httpx.HTTPError as e:
            # Add logging here in a real application
            print(f"Error creating sandbox: {e}")
            return None

    async def execute_command(self, container_id: str, command: str) -> Optional[dict]:
        """Executes a command in a sandbox."""
        try:
            response = await self.client.post(
                f"{self.base_url}/sandboxes/{container_id}/execute",
                json={"command": command}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error executing command in sandbox {container_id}: {e}")
            return None

    async def remove_sandbox(self, container_id: str):
        """Removes a sandbox."""
        try:
            response = await self.client.delete(f"{self.base_url}/sandboxes/{container_id}")
            response.raise_for_status()
        except httpx.HTTPError as e:
            print(f"Error removing sandbox {container_id}: {e}")

    async def close(self):
        await self.client.aclose() 