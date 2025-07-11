import structlog
import json
from agentQ.app.core.toolbox import Tool
from shared.vault_client import VaultClient

logger = structlog.get_logger(__name__)

def rotate_database_credentials(database_role_name: str, config: dict = None) -> str:
    """
    Connects to Vault and generates a new set of dynamic database credentials for a given role.

    Args:
        database_role_name (str): The name of the Vault database role (e.g., 'userprofile-db-role').

    Returns:
        str: A JSON string containing the new username, password, and lease duration.
    """
    logger.info("Rotating database credentials", role=database_role_name)
    try:
        # The VaultClient is already configured to use the agent's service role
        vault_client = VaultClient(role="agentq-role")
        
        # The path to the credentials endpoint for a database dynamic secret engine
        path = f"database/creds/{database_role_name}"
        
        new_creds = vault_client.read_secret_data(path)

        if not new_creds or "data" not in new_creds:
            raise Exception("Failed to retrieve new credentials from Vault.")

        credential_data = {
            "username": new_creds["data"]["username"],
            "password": new_creds["data"]["password"],
            "lease_duration": new_creds["lease_duration"]
        }
        
        # We only return the data, not the full lease object, to the agent
        return json.dumps(credential_data)

    except Exception as e:
        logger.error("Failed to rotate database credentials", exc_info=True)
        return f"Error: An unexpected error occurred during secret rotation: {e}"


rotate_creds_tool = Tool(
    name="rotate_database_credentials",
    description="Generates a new, dynamic set of database credentials from Vault for a specified role.",
    func=rotate_database_credentials
) 