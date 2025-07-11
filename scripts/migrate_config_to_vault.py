#!/usr/bin/env python3
"""
Enhanced configuration migration script for Q Platform.
Migrates all service configurations to HashiCorp Vault.
"""

import yaml
import hvac
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

class ConfigMigrator:
    def __init__(self, vault_addr: str, vault_token: str):
        self.vault_addr = vault_addr
        self.vault_token = vault_token
        self.client = hvac.Client(url=vault_addr, token=vault_token)
        
        # Verify Vault connection
        if not self.client.is_authenticated():
            raise ValueError("Failed to authenticate with Vault")
        
        print(f"✓ Connected to Vault at {vault_addr}")
    
    def migrate_config(self, config_path: str, secret_path: str, config_type: str = "yaml"):
        """
        Migrates a configuration file to Vault.
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                print(f"⚠️  Config file {config_path} does not exist, skipping...")
                return False
            
            if config_type == "yaml":
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif config_type == "json":
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            elif config_type == "env":
                config_data = self._parse_env_file(config_path)
            else:
                raise ValueError(f"Unsupported config type: {config_type}")

            # Store in Vault
            self.client.secrets.kv.v2.create_or_update_secret(
                path=secret_path,
                secret=config_data,
            )
            print(f"✓ Migrated {config_path} to Vault at {secret_path}")
            return True

        except Exception as e:
            print(f"❌ Error migrating {config_path}: {e}")
            return False
    
    def _parse_env_file(self, env_path: str) -> Dict[str, str]:
        """Parse environment file into dictionary."""
        config = {}
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        return config
    
    def create_service_secrets(self, service_name: str, secrets: Dict[str, Any]):
        """Create secrets for a service."""
        secret_path = f"secret/data/{service_name}"
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=secret_path,
                secret=secrets,
            )
            print(f"✓ Created secrets for {service_name}")
            return True
        except Exception as e:
            print(f"❌ Error creating secrets for {service_name}: {e}")
            return False
    
    def initialize_platform_secrets(self):
        """Initialize secrets for all Q Platform services."""
        services_secrets = {
            "manager-q": {
                "database_url": "postgresql://managerq:password@postgres:5432/managerq",
                "redis_url": "redis://redis:6379",
                "secret_key": "manager-q-secret-key-change-me",
                "api_key": "manager-q-api-key-change-me",
                "jwt_secret": "manager-q-jwt-secret-change-me"
            },
            "agentq": {
                "database_url": "postgresql://agentq:password@postgres:5432/agentq",
                "redis_url": "redis://redis:6379",
                "secret_key": "agentq-secret-key-change-me",
                "openai_api_key": "sk-placeholder-change-me",
                "anthropic_api_key": "placeholder-change-me"
            },
            "h2m-service": {
                "database_url": "postgresql://h2m:password@postgres:5432/h2m",
                "redis_url": "redis://redis:6379",
                "secret_key": "h2m-secret-key-change-me",
                "websocket_secret": "h2m-websocket-secret-change-me"
            },
            "quantumpulse": {
                "database_url": "postgresql://quantumpulse:password@postgres:5432/quantumpulse",
                "redis_url": "redis://redis:6379",
                "secret_key": "quantumpulse-secret-key-change-me",
                "model_api_key": "quantumpulse-model-api-key-change-me"
            },
            "vectorstore-q": {
                "database_url": "postgresql://vectorstore:password@postgres:5432/vectorstore",
                "milvus_host": "milvus",
                "milvus_port": "19530",
                "secret_key": "vectorstore-secret-key-change-me"
            },
            "knowledgegraphq": {
                "janusgraph_host": "janusgraph",
                "janusgraph_port": "8182",
                "cassandra_host": "cassandra",
                "cassandra_port": "9042",
                "secret_key": "knowledgegraph-secret-key-change-me"
            },
            "integrationhub": {
                "database_url": "postgresql://integrationhub:password@postgres:5432/integrationhub",
                "secret_key": "integrationhub-secret-key-change-me",
                "github_token": "ghp_placeholder-change-me",
                "jira_token": "placeholder-change-me",
                "slack_token": "xoxb-placeholder-change-me"
            },
            "userprofileq": {
                "database_url": "postgresql://userprofile:password@postgres:5432/userprofile",
                "secret_key": "userprofile-secret-key-change-me",
                "encryption_key": "userprofile-encryption-key-change-me"
            },
            "authq": {
                "database_url": "postgresql://authq:password@postgres:5432/authq",
                "secret_key": "authq-secret-key-change-me",
                "keycloak_secret": "authq-keycloak-secret-change-me"
            },
            "webapp-q": {
                "api_base_url": "https://api.q-platform.local",
                "keycloak_url": "https://keycloak.q-platform.local",
                "keycloak_realm": "q-platform",
                "keycloak_client_id": "q-webapp"
            },
            "workflowworker": {
                "database_url": "postgresql://workflowworker:password@postgres:5432/workflowworker",
                "redis_url": "redis://redis:6379",
                "secret_key": "workflowworker-secret-key-change-me"
            },
            "aiops": {
                "database_url": "postgresql://aiops:password@postgres:5432/aiops",
                "secret_key": "aiops-secret-key-change-me",
                "monitoring_api_key": "aiops-monitoring-api-key-change-me"
            },
            "agentsandbox": {
                "database_url": "postgresql://agentsandbox:password@postgres:5432/agentsandbox",
                "secret_key": "agentsandbox-secret-key-change-me",
                "sandbox_secret": "agentsandbox-sandbox-secret-change-me"
            }
        }
        
        success_count = 0
        for service_name, secrets in services_secrets.items():
            if self.create_service_secrets(service_name, secrets):
                success_count += 1
        
        print(f"✓ Initialized secrets for {success_count}/{len(services_secrets)} services")
        return success_count == len(services_secrets)
    
    def migrate_all_configs(self):
        """Migrate all configuration files to Vault."""
        config_mappings = [
            # Service configurations
            ("managerQ/config/manager.yaml", "secret/data/manager-q/config", "yaml"),
            ("agentQ/config/agent.yaml", "secret/data/agentq/config", "yaml"),
            ("H2M/config/h2m.yaml", "secret/data/h2m-service/config", "yaml"),
            ("VectorStoreQ/config/vectorstore.yaml", "secret/data/vectorstore-q/config", "yaml"),
            ("QuantumPulse/config/quantumpulse.yaml", "secret/data/quantumpulse/config", "yaml"),
            ("KnowledgeGraphQ/config/knowledgegraph.yaml", "secret/data/knowledgegraphq/config", "yaml"),
            ("IntegrationHub/config/integrationhub.yaml", "secret/data/integrationhub/config", "yaml"),
            ("UserProfileQ/config/userprofile.yaml", "secret/data/userprofileq/config", "yaml"),
            ("AuthQ/config/authq.yaml", "secret/data/authq/config", "yaml"),
            ("WebAppQ/config/webapp.yaml", "secret/data/webapp-q/config", "yaml"),
            ("WorkflowWorker/config/workflowworker.yaml", "secret/data/workflowworker/config", "yaml"),
            ("AIOps/config/aiops.yaml", "secret/data/aiops/config", "yaml"),
            ("AgentSandbox/config/agentsandbox.yaml", "secret/data/agentsandbox/config", "yaml"),
            
            # Environment files
            (".env", "secret/data/platform/env", "env"),
            ("docker-compose.env", "secret/data/platform/docker-env", "env"),
            
            # Infrastructure configurations
            ("infra/terraform/terraform.tfvars", "secret/data/infrastructure/terraform", "env"),
            ("helm/q-platform/values.yaml", "secret/data/helm/values", "yaml"),
            ("helm/q-platform/values-dev.yaml", "secret/data/helm/values-dev", "yaml"),
            ("helm/q-platform/values-prod.yaml", "secret/data/helm/values-prod", "yaml"),
        ]
        
        success_count = 0
        for config_path, secret_path, config_type in config_mappings:
            if self.migrate_config(config_path, secret_path, config_type):
                success_count += 1
        
        print(f"✓ Migrated {success_count}/{len(config_mappings)} configuration files")
        return success_count == len(config_mappings)

def main():
    parser = argparse.ArgumentParser(description='Migrate Q Platform configurations to Vault')
    parser.add_argument('--vault-addr', default=os.environ.get('VAULT_ADDR', 'http://127.0.0.1:8200'),
                        help='Vault server address')
    parser.add_argument('--vault-token', default=os.environ.get('VAULT_TOKEN'),
                        help='Vault authentication token')
    parser.add_argument('--init-secrets', action='store_true',
                        help='Initialize platform secrets')
    parser.add_argument('--migrate-configs', action='store_true',
                        help='Migrate configuration files')
    parser.add_argument('--all', action='store_true',
                        help='Run all migration tasks')
    
    args = parser.parse_args()
    
    if not args.vault_token:
        print("❌ VAULT_TOKEN environment variable must be set or provided via --vault-token")
        sys.exit(1)
    
    try:
        migrator = ConfigMigrator(args.vault_addr, args.vault_token)
        
        if args.all or args.init_secrets:
            print("\n📋 Initializing platform secrets...")
            migrator.initialize_platform_secrets()
        
        if args.all or args.migrate_configs:
            print("\n📋 Migrating configuration files...")
            migrator.migrate_all_configs()
        
        if not any([args.init_secrets, args.migrate_configs, args.all]):
            print("❌ No migration tasks specified. Use --init-secrets, --migrate-configs, or --all")
            sys.exit(1)
        
        print("\n✅ Configuration migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 