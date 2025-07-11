import hvac
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# The standard path where a service account token is mounted in a pod
K8S_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
# The Vault role is now passed in, not hardcoded
# VAULT_K8S_ROLE = "q-platform-role"

class VaultClient:
    """A client for HashiCorp Vault that can use K8s auth."""

    def __init__(self, vault_addr: Optional[str] = None, role: Optional[str] = None):
        self._vault_addr = vault_addr or os.environ.get('VAULT_ADDR', '')
        self._role = role or os.environ.get('VAULT_ROLE', '')
        self.client = hvac.Client(url=self._vault_addr)
        self._authenticate()

    def _authenticate(self):
        """
        Authenticates with Vault. Prefers K8s auth, falls back to token auth.
        """
        # Case 1: Running in Kubernetes
        if os.path.exists(K8S_SA_TOKEN_PATH):
            logger.info("Detected Kubernetes environment. Attempting K8s auth.")
            if not self._role:
                raise ValueError("Vault role must be provided for Kubernetes auth.")
            try:
                with open(K8S_SA_TOKEN_PATH, 'r') as f:
                    jwt = f.read()
                
                self.client.auth.kubernetes.login(
                    role=self._role,
                    jwt=jwt
                )
                if self.client.is_authenticated():
                    logger.info("Vault client authenticated successfully using Kubernetes service account.")
                    return
            except Exception as e:
                logger.error(f"Kubernetes auth failed: {e}. Falling back to token auth.", exc_info=True)

        # Case 2: Fallback to token (for local dev)
        token = os.environ.get('VAULT_TOKEN')
        if token:
            self.client.token = token
            if self.client.is_authenticated():
                logger.info("Vault client authenticated successfully using VAULT_TOKEN.")
                return

        raise ConnectionError("Failed to authenticate with Vault. No valid k8s token or VAULT_TOKEN found.")

    def read_secret_data(self, path: str) -> dict:
        """
        Reads the entire data payload from a secret at a given path.

        Args:
            path: The path to the secret in Vault (e.g., 'secret/data/managerq/config').

        Returns:
            A dictionary containing the secret data.
        """
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data']
        except Exception as e:
            logger.error(f"Failed to read secret data from path '{path}': {e}", exc_info=True)
            raise

    def read_secret(self, path: str, key: str) -> str:
        """
        Reads a specific key from a secret at a given path.

        Args:
            path: The path to the secret in Vault (e.g., 'secret/data/openai').
            key: The key within the secret to retrieve.

        Returns:
            The secret value.
        """
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data'][key]
        except Exception as e:
            logger.error(f"Failed to read secret '{key}' from path '{path}': {e}", exc_info=True)
            raise

    def get_database_creds(self, db_role_name: str) -> dict:
        """
        Generates dynamic database credentials using a specific Vault role.

        Args:
            db_role_name: The name of the database role in Vault (e.g., 'cassandra-readonly').

        Returns:
            A dictionary containing the username, password, and lease duration.
        """
        try:
            # Assumes a database secrets engine is mounted at 'database/'
            response = self.client.secrets.database.generate_credentials(name=db_role_name)
            return response['data']
        except Exception as e:
            logger.error(f"Failed to generate database credentials for role '{db_role_name}': {e}", exc_info=True)
            raise

    def get_token(self, role: str, ttl: str) -> dict:
        """
        Requests a Vault token for a specific role and TTL.
        This is typically used for service-to-service authentication where a new token
        is needed for a specific purpose or extended duration.

        Args:
            role (str): The Vault role to request the token for.
            ttl (str): The desired time-to-live for the token (e.g., "1h", "30m").

        Returns:
            A dictionary containing the token details, including 'client_token'.
        """
        try:
            # Use the Kubernetes auth backend to request a token
            # This assumes the client is already authenticated via K8s service account
            response = self.client.auth.kubernetes.create_token(role=role, ttl=ttl)
            return response
        except Exception as e:
            logger.error(f"Failed to generate Vault token for role '{role}' with TTL '{ttl}': {e}", exc_info=True)
            raise 