import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio
import os

from shared.q_knowledgegraph_client.client import KnowledgeGraphClient

logger = logging.getLogger(__name__)

class ModelRegistryService:
    """
    Manages the lifecycle and availability of ML models within the Q Platform.
    It tracks model versions, their metadata, storage locations, and active deployments.
    """
    def __init__(self, models_base_path: str = "models/registered_models", kg_client: Optional[KnowledgeGraphClient] = None):
        self.models_base_path = Path(models_base_path)
        self.models_base_path.mkdir(parents=True, exist_ok=True)
        self.registered_models: Dict[str, Dict[str, Any]] = {}
        self.active_models: Dict[str, str] = {} # model_name -> active_version_id
        self._lock = asyncio.Lock()
        self.kg_client = kg_client or KnowledgeGraphClient(base_url=os.getenv("KGQ_API_URL", "http://knowledgegraphq:8000")) # Corrected KG client initialization

    async def initialize(self):
        logger.info("Initializing ModelRegistryService...")
        await self._load_registry_state()
        # Removed await self.kg_client.initialize() as it does not exist
        logger.info(f"Loaded {len(self.registered_models)} models and {len(self.active_models)} active deployments.")

    async def shutdown(self):
        logger.info("Shutting down ModelRegistryService...")
        await self._save_registry_state()
        await self.kg_client.aclose() # Corrected method call
        logger.info("ModelRegistryService state saved.")

    async def _load_registry_state(self):
        state_file = self.models_base_path / "registry_state.json"
        if state_file.exists():
            try:
                async with self._lock:
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                        self.registered_models = state.get("registered_models", {})
                        self.active_models = state.get("active_models", {})
                logger.info("Model registry state loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model registry state: {e}", exc_info=True)

    async def _save_registry_state(self):
        state_file = self.models_base_path / "registry_state.json"
        try:
            async with self._lock:
                state = {
                    "registered_models": self.registered_models,
                    "active_models": self.active_models
                }
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2, default=str)
            logger.info("Model registry state saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save model registry state: {e}", exc_info=True)

    async def _store_model_in_kg(self, model_info: Dict[str, Any]):
        """Stores model metadata as a vertex in the Knowledge Graph."""
        try:
            vertex_data = {
                "model_id": model_info["version_id"],
                "model_name": model_info["model_name"],
                "artifact_path": model_info["artifact_path"],
                "registered_at": model_info["registered_at"],
                "status": model_info["status"],
                "metadata": model_info["metadata"],
            }
            await self.kg_client.add_vertex(
                "Model",
                model_info["version_id"],
                vertex_data
            )
            logger.info(f"Stored model {model_info['version_id']} in KG.")
        except Exception as e:
            logger.error(f"Failed to store model in KG: {e}", exc_info=True)

    async def _store_model_data_and_feature_lineage_in_kg(
        self,
        model_version_id: str,
        dataset_version_id: Optional[str] = None,
        feature_ids: Optional[List[str]] = None
    ):
        """Stores lineage relationships for a model in the Knowledge Graph."""
        try:
            if dataset_version_id:
                await self.kg_client.add_edge(
                    "trained_on",
                    model_version_id,
                    dataset_version_id,
                    {"relationship": "trained_on"}
                )
                logger.info(f"Created 'trained_on' edge from {model_version_id} to {dataset_version_id}.")

            if feature_ids:
                for feature_id in feature_ids:
                    await self.kg_client.add_edge(
                        "uses",
                        model_version_id,
                        feature_id,
                        {"relationship": "uses"}
                    )
                logger.info(f"Created 'uses' edges from {model_version_id} to {feature_ids}.")

        except Exception as e:
            logger.error(f"Failed to store model lineage in KG: {e}", exc_info=True)

    async def register_model_version(
        self,
        model_name: str,
        version_id: str,
        artifact_path: str,
        metadata: Dict[str, Any],
        is_active: bool = False,
        dataset_version_id: Optional[str] = None, # Added parameter
        feature_ids: Optional[List[str]] = None # Added parameter
    ) -> Dict[str, Any]:
        """
        Registers a new version for a model.
        """
        async with self._lock:
            if model_name not in self.registered_models:
                self.registered_models[model_name] = {}
            
            if version_id in self.registered_models[model_name]:
                logger.warning(f"Model version {version_id} for {model_name} already exists. Overwriting.")

            model_info = {
                "model_name": model_name, # Added model_name to model_info
                "version_id": version_id,
                "artifact_path": str(artifact_path),
                "metadata": metadata,
                "registered_at": datetime.utcnow().isoformat(),
                "status": "registered"
            }
            self.registered_models[model_name][version_id] = model_info
            
            if is_active:
                self.active_models[model_name] = version_id

            await self._save_registry_state()
            await self._store_model_in_kg(model_info) # Store model in KG
            await self._store_model_data_and_feature_lineage_in_kg( # Store lineage in KG
                model_version_id=version_id,
                dataset_version_id=dataset_version_id,
                feature_ids=feature_ids
            )
            logger.info(f"Registered model '{model_name}' version '{version_id}'.")
            return model_info

    async def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Lists all registered versions for a given model.
        """
        async with self._lock:
            versions = list(self.registered_models.get(model_name, {}).values())
            return sorted(versions, key=lambda x: x.get("registered_at"), reverse=True)

    async def get_model_version(self, model_name: str, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves details for a specific model version.
        """
        async with self._lock:
            return self.registered_models.get(model_name, {}).get(version_id)

    async def activate_model_version(self, model_name: str, version_id: str) -> bool:
        """
        Sets a specific model version as active for inference.
        """
        async with self._lock:
            if model_name not in self.registered_models or version_id not in self.registered_models[model_name]:
                logger.warning(f"Cannot activate: Model '{model_name}' version '{version_id}' not found.")
                return False
            
            self.active_models[model_name] = version_id
            await self._save_registry_state()
            logger.info(f"Activated model '{model_name}' version '{version_id}'.")
            return True

    async def get_active_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the currently active model version for a given model.
        """
        async with self._lock:
            active_version_id = self.active_models.get(model_name)
            if active_version_id:
                return self.registered_models.get(model_name, {}).get(active_version_id)
            return None

    async def get_active_model_artifact_path(self, model_name: str) -> Optional[str]:
        """
        Retrieves the file path of the currently active model's artifact.
        """
        async with self._lock:
            active_model = await self.get_active_model(model_name)
            return active_model.get("artifact_path") if active_model else None

# Global instance (for use within managerQ)
model_registry_service = ModelRegistryService() 