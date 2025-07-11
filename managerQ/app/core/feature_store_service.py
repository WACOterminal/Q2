"""
Feature Store Service for Q Platform

This service provides comprehensive feature management capabilities:
- Feature registration and discovery
- Feature transformation and engineering
- Online and offline feature serving
- Feature versioning and lineage tracking
- Integration with Apache Ignite for caching
- Real-time feature computation via Pulsar
- Feature quality monitoring and validation
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from collections import defaultdict
import hashlib
import re

# Feature engineering libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.vault_client import VaultClient
from .data_versioning_service import DataVersioningService

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Feature data types"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    EMBEDDING = "embedding"
    ARRAY = "array"
    JSON = "json"

class FeatureStatus(Enum):
    """Feature status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    ARCHIVED = "archived"

class TransformationType(Enum):
    """Transformation types"""
    SCALING = "scaling"
    ENCODING = "encoding"
    AGGREGATION = "aggregation"
    WINDOWING = "windowing"
    COMPUTATION = "computation"
    CUSTOM = "custom"

class FeatureServingMode(Enum):
    """Feature serving modes"""
    ONLINE = "online"
    OFFLINE = "offline"
    BATCH = "batch"
    STREAMING = "streaming"

@dataclass
class FeatureDefinition:
    """Feature definition metadata"""
    feature_id: str
    feature_name: str
    feature_type: FeatureType
    entity_type: str
    description: str
    feature_group: str
    status: FeatureStatus
    version: str
    created_at: datetime
    created_by: str
    source_table: Optional[str] = None
    source_column: Optional[str] = None
    transformation_config: Optional[Dict[str, Any]] = None
    validation_rules: Optional[Dict[str, Any]] = None
    serving_config: Optional[Dict[str, Any]] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    last_updated: Optional[datetime] = None
    updated_by: Optional[str] = None

@dataclass
class FeatureGroup:
    """Feature group metadata"""
    group_id: str
    group_name: str
    description: str
    entity_type: str
    features: List[str]
    serving_modes: List[FeatureServingMode]
    refresh_frequency: str
    created_at: datetime
    created_by: str
    status: FeatureStatus
    tags: List[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class FeatureTransformation:
    """Feature transformation definition"""
    transformation_id: str
    transformation_name: str
    transformation_type: TransformationType
    source_features: List[str]
    target_feature: str
    transformation_function: str
    transformation_config: Dict[str, Any]
    created_at: datetime
    created_by: str
    description: Optional[str] = None
    validation_rules: Optional[Dict[str, Any]] = None

@dataclass
class FeatureValue:
    """Feature value with metadata"""
    entity_id: str
    feature_id: str
    value: Any
    timestamp: datetime
    version: str
    computed_at: datetime
    metadata: Dict[str, Any] = None

@dataclass
class FeatureServingRequest:
    """Feature serving request"""
    request_id: str
    entity_ids: List[str]
    feature_names: List[str]
    serving_mode: FeatureServingMode
    as_of_timestamp: Optional[datetime] = None
    request_timestamp: datetime = None
    metadata: Dict[str, Any] = None

@dataclass
class FeatureQualityMetrics:
    """Feature quality metrics"""
    feature_id: str
    completeness: float
    uniqueness: float
    consistency: float
    validity: float
    timeliness: float
    accuracy: float
    computed_at: datetime
    sample_size: int
    anomaly_count: int
    quality_score: float

class FeatureStoreService:
    """
    Comprehensive Feature Store Service
    """
    
    def __init__(self, 
                 storage_path: str = "features",
                 kg_client: Optional[KnowledgeGraphClient] = None,
                 data_versioning_service: Optional[DataVersioningService] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Service dependencies
        self.kg_client = kg_client or KnowledgeGraphClient()
        self.data_versioning_service = data_versioning_service or DataVersioningService()
        
        # Feature registries
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.feature_transformations: Dict[str, FeatureTransformation] = {}
        self.feature_values: Dict[str, Dict[str, FeatureValue]] = defaultdict(dict)  # entity_id -> {feature_id: value}
        
        # Quality metrics
        self.quality_metrics: Dict[str, FeatureQualityMetrics] = {}
        
        # Transformation functions registry
        self.transformation_functions: Dict[str, Callable] = {}
        
        # Feature serving cache (simulate Ignite)
        self.feature_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Configuration
        self.config = {
            "cache_ttl": 3600,  # seconds
            "batch_size": 1000,
            "max_cache_size": 10000,
            "quality_check_frequency": 3600,
            "feature_freshness_threshold": 7200,  # seconds
            "default_serving_mode": FeatureServingMode.ONLINE
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Performance metrics
        self.feature_store_metrics = {
            "total_features": 0,
            "active_features": 0,
            "feature_groups": 0,
            "transformations": 0,
            "serving_requests": 0,
            "cache_hit_rate": 0.0,
            "average_latency": 0.0,
            "quality_score": 0.0
        }
        
        # Initialize built-in transformations
        self._register_builtin_transformations()
        
    async def initialize(self):
        """Initialize the feature store service"""
        logger.info("Initializing Feature Store Service")
        
        # Load existing features
        await self._load_features()
        
        # Initialize dependencies
        await self.kg_client.initialize()
        await self.data_versioning_service.initialize()
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        # Start background tasks
        self.background_tasks.add(asyncio.create_task(self._feature_quality_monitoring()))
        self.background_tasks.add(asyncio.create_task(self._feature_freshness_monitoring()))
        self.background_tasks.add(asyncio.create_task(self._cache_management()))
        self.background_tasks.add(asyncio.create_task(self._metrics_tracking()))
        
        logger.info("Feature Store Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the feature store service"""
        logger.info("Shutting down Feature Store Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save features
        await self._save_features()
        
        # Shutdown dependencies
        await self.data_versioning_service.shutdown()
        
        logger.info("Feature Store Service shut down successfully")
    
    # ===== FEATURE REGISTRATION =====
    
    async def register_feature(
        self,
        feature_name: str,
        feature_type: FeatureType,
        entity_type: str,
        description: str,
        feature_group: str,
        created_by: str,
        source_table: Optional[str] = None,
        source_column: Optional[str] = None,
        transformation_config: Optional[Dict[str, Any]] = None,
        validation_rules: Optional[Dict[str, Any]] = None,
        serving_config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new feature
        
        Args:
            feature_name: Name of the feature
            feature_type: Type of the feature
            entity_type: Entity type (e.g., "user", "product")
            description: Feature description
            feature_group: Feature group name
            created_by: Creator identifier
            source_table: Source table name
            source_column: Source column name
            transformation_config: Transformation configuration
            validation_rules: Validation rules
            serving_config: Serving configuration
            tags: Optional tags
            metadata: Optional metadata
            
        Returns:
            Feature ID
        """
        feature_id = f"feature_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Registering feature: {feature_name}")
        
        # Create feature definition
        feature_def = FeatureDefinition(
            feature_id=feature_id,
            feature_name=feature_name,
            feature_type=feature_type,
            entity_type=entity_type,
            description=description,
            feature_group=feature_group,
            status=FeatureStatus.ACTIVE,
            version="1.0",
            created_at=datetime.utcnow(),
            created_by=created_by,
            source_table=source_table,
            source_column=source_column,
            transformation_config=transformation_config or {},
            validation_rules=validation_rules or {},
            serving_config=serving_config or {},
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Store feature definition
        self.feature_definitions[feature_id] = feature_def
        
        # Store in KnowledgeGraph
        await self._store_feature_in_kg(feature_def)
        
        # Update feature group
        await self._update_feature_group(feature_group, feature_id, created_by)
        
        # Publish feature registered event
        await shared_pulsar_client.publish(
            "q.ml.feature.registered",
            {
                "feature_id": feature_id,
                "feature_name": feature_name,
                "feature_type": feature_type.value,
                "entity_type": entity_type,
                "feature_group": feature_group,
                "created_by": created_by,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Registered feature: {feature_id}")
        return feature_id
    
    async def get_feature(self, feature_id: str) -> Optional[FeatureDefinition]:
        """Get feature definition by ID"""
        return self.feature_definitions.get(feature_id)
    
    async def get_feature_by_name(self, feature_name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name"""
        for feature in self.feature_definitions.values():
            if feature.feature_name == feature_name:
                return feature
        return None
    
    async def list_features(
        self,
        entity_type: Optional[str] = None,
        feature_group: Optional[str] = None,
        status: Optional[FeatureStatus] = None,
        feature_type: Optional[FeatureType] = None,
        limit: int = 100
    ) -> List[FeatureDefinition]:
        """List features with optional filtering"""
        
        features = list(self.feature_definitions.values())
        
        if entity_type:
            features = [f for f in features if f.entity_type == entity_type]
        
        if feature_group:
            features = [f for f in features if f.feature_group == feature_group]
        
        if status:
            features = [f for f in features if f.status == status]
        
        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]
        
        # Sort by creation time (newest first)
        features.sort(key=lambda x: x.created_at, reverse=True)
        
        return features[:limit]
    
    async def update_feature(
        self,
        feature_id: str,
        updates: Dict[str, Any],
        updated_by: str
    ) -> bool:
        """Update feature definition"""
        
        if feature_id not in self.feature_definitions:
            return False
        
        feature = self.feature_definitions[feature_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(feature, key):
                setattr(feature, key, value)
        
        feature.last_updated = datetime.utcnow()
        feature.updated_by = updated_by
        
        # Update in KnowledgeGraph
        await self._update_feature_in_kg(feature_id, updates)
        
        # Publish feature updated event
        await shared_pulsar_client.publish(
            "q.ml.feature.updated",
            {
                "feature_id": feature_id,
                "updates": updates,
                "updated_by": updated_by,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return True
    
    # ===== FEATURE GROUPS =====
    
    async def create_feature_group(
        self,
        group_name: str,
        description: str,
        entity_type: str,
        serving_modes: List[FeatureServingMode],
        refresh_frequency: str,
        created_by: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a feature group"""
        
        group_id = f"group_{uuid.uuid4().hex[:12]}"
        
        feature_group = FeatureGroup(
            group_id=group_id,
            group_name=group_name,
            description=description,
            entity_type=entity_type,
            features=[],
            serving_modes=serving_modes,
            refresh_frequency=refresh_frequency,
            created_at=datetime.utcnow(),
            created_by=created_by,
            status=FeatureStatus.ACTIVE,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.feature_groups[group_id] = feature_group
        
        # Store in KnowledgeGraph
        await self._store_feature_group_in_kg(feature_group)
        
        return group_id
    
    async def _update_feature_group(
        self,
        group_name: str,
        feature_id: str,
        created_by: str
    ):
        """Update or create feature group with new feature"""
        
        # Find existing group
        existing_group = None
        for group in self.feature_groups.values():
            if group.group_name == group_name:
                existing_group = group
                break
        
        if existing_group:
            # Add feature to existing group
            if feature_id not in existing_group.features:
                existing_group.features.append(feature_id)
        else:
            # Create new group
            await self.create_feature_group(
                group_name=group_name,
                description=f"Feature group for {group_name}",
                entity_type="unknown",
                serving_modes=[FeatureServingMode.ONLINE, FeatureServingMode.OFFLINE],
                refresh_frequency="hourly",
                created_by=created_by
            )
    
    # ===== FEATURE TRANSFORMATIONS =====
    
    async def register_transformation(
        self,
        transformation_name: str,
        transformation_type: TransformationType,
        source_features: List[str],
        target_feature: str,
        transformation_function: str,
        transformation_config: Dict[str, Any],
        created_by: str,
        description: Optional[str] = None,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a feature transformation"""
        
        transformation_id = f"transform_{uuid.uuid4().hex[:12]}"
        
        transformation = FeatureTransformation(
            transformation_id=transformation_id,
            transformation_name=transformation_name,
            transformation_type=transformation_type,
            source_features=source_features,
            target_feature=target_feature,
            transformation_function=transformation_function,
            transformation_config=transformation_config,
            created_at=datetime.utcnow(),
            created_by=created_by,
            description=description,
            validation_rules=validation_rules or {}
        )
        
        self.feature_transformations[transformation_id] = transformation
        
        # Store in KnowledgeGraph
        await self._store_transformation_in_kg(transformation)
        
        return transformation_id
    
    async def apply_transformation(
        self,
        transformation_id: str,
        entity_ids: List[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply transformation to input data"""
        
        transformation = self.feature_transformations.get(transformation_id)
        if not transformation:
            raise ValueError(f"Transformation not found: {transformation_id}")
        
        # Get transformation function
        transform_func = self.transformation_functions.get(transformation.transformation_function)
        if not transform_func:
            raise ValueError(f"Transformation function not found: {transformation.transformation_function}")
        
        # Apply transformation
        result = await transform_func(
            input_data,
            transformation.transformation_config,
            entity_ids
        )
        
        return result
    
    def _register_builtin_transformations(self):
        """Register built-in transformation functions"""
        
        self.transformation_functions.update({
            "standard_scaling": self._standard_scaling,
            "min_max_scaling": self._min_max_scaling,
            "one_hot_encoding": self._one_hot_encoding,
            "label_encoding": self._label_encoding,
            "windowing_aggregation": self._windowing_aggregation,
            "time_based_features": self._time_based_features,
            "text_vectorization": self._text_vectorization,
            "pca_reduction": self._pca_reduction,
            "feature_selection": self._feature_selection,
            "custom_aggregation": self._custom_aggregation
        })
    
    # ===== FEATURE SERVING =====
    
    async def serve_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        serving_mode: FeatureServingMode = FeatureServingMode.ONLINE,
        as_of_timestamp: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Serve features for entities
        
        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names to serve
            serving_mode: Serving mode
            as_of_timestamp: Point-in-time timestamp
            
        Returns:
            Dictionary mapping entity_id to feature values
        """
        request_id = f"serve_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Serving features: {len(feature_names)} features for {len(entity_ids)} entities")
        
        # Track serving request
        self.feature_store_metrics["serving_requests"] += 1
        
        results = {}
        
        for entity_id in entity_ids:
            entity_features = {}
            
            for feature_name in feature_names:
                # Get feature definition
                feature_def = await self.get_feature_by_name(feature_name)
                if not feature_def:
                    logger.warning(f"Feature not found: {feature_name}")
                    continue
                
                # Try to serve from cache first
                if serving_mode == FeatureServingMode.ONLINE:
                    cached_value = await self._get_from_cache(entity_id, feature_def.feature_id)
                    if cached_value is not None:
                        entity_features[feature_name] = cached_value
                        continue
                
                # Serve from storage
                feature_value = await self._serve_feature_from_storage(
                    entity_id, feature_def, as_of_timestamp
                )
                
                if feature_value is not None:
                    entity_features[feature_name] = feature_value
                    
                    # Cache for online serving
                    if serving_mode == FeatureServingMode.ONLINE:
                        await self._cache_feature(entity_id, feature_def.feature_id, feature_value)
            
            results[entity_id] = entity_features
        
        # Publish serving event
        await shared_pulsar_client.publish(
            "q.ml.feature.served",
            {
                "request_id": request_id,
                "entity_count": len(entity_ids),
                "feature_count": len(feature_names),
                "serving_mode": serving_mode.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return results
    
    async def _serve_feature_from_storage(
        self,
        entity_id: str,
        feature_def: FeatureDefinition,
        as_of_timestamp: Optional[datetime] = None
    ) -> Optional[Any]:
        """Serve feature from storage"""
        
        # Get stored feature value
        feature_value = self.feature_values[entity_id].get(feature_def.feature_id)
        
        if feature_value:
            # Check timestamp constraint
            if as_of_timestamp and feature_value.timestamp > as_of_timestamp:
                return None
            
            return feature_value.value
        
        return None
    
    async def store_feature_values(
        self,
        entity_id: str,
        features: Dict[str, Any],
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store feature values for an entity"""
        
        timestamp = datetime.utcnow()
        
        for feature_name, value in features.items():
            # Get feature definition
            feature_def = await self.get_feature_by_name(feature_name)
            if not feature_def:
                continue
            
            # Create feature value
            feature_value = FeatureValue(
                entity_id=entity_id,
                feature_id=feature_def.feature_id,
                value=value,
                timestamp=timestamp,
                version=version,
                computed_at=timestamp,
                metadata=metadata or {}
            )
            
            # Store feature value
            self.feature_values[entity_id][feature_def.feature_id] = feature_value
            
            # Cache for online serving
            await self._cache_feature(entity_id, feature_def.feature_id, value)
        
        # Publish feature values stored event
        await shared_pulsar_client.publish(
            "q.ml.feature.values.stored",
            {
                "entity_id": entity_id,
                "feature_count": len(features),
                "version": version,
                "timestamp": timestamp.isoformat()
            }
        )
    
    # ===== FEATURE QUALITY =====
    
    async def compute_feature_quality(
        self,
        feature_id: str,
        sample_size: int = 1000
    ) -> FeatureQualityMetrics:
        """Compute quality metrics for a feature"""
        
        feature_def = self.feature_definitions.get(feature_id)
        if not feature_def:
            raise ValueError(f"Feature not found: {feature_id}")
        
        # Collect feature values
        feature_values = []
        for entity_values in self.feature_values.values():
            if feature_id in entity_values:
                feature_values.append(entity_values[feature_id].value)
        
        # Sample if needed
        if len(feature_values) > sample_size:
            feature_values = np.random.choice(feature_values, sample_size, replace=False)
        
        # Compute quality metrics
        completeness = self._compute_completeness(feature_values)
        uniqueness = self._compute_uniqueness(feature_values)
        consistency = self._compute_consistency(feature_values, feature_def.feature_type)
        validity = self._compute_validity(feature_values, feature_def.validation_rules)
        timeliness = self._compute_timeliness(feature_id)
        accuracy = self._compute_accuracy(feature_values, feature_def)
        
        # Count anomalies
        anomaly_count = self._count_anomalies(feature_values, feature_def)
        
        # Overall quality score
        quality_score = (completeness + uniqueness + consistency + validity + timeliness + accuracy) / 6
        
        # Create quality metrics
        quality_metrics = FeatureQualityMetrics(
            feature_id=feature_id,
            completeness=completeness,
            uniqueness=uniqueness,
            consistency=consistency,
            validity=validity,
            timeliness=timeliness,
            accuracy=accuracy,
            computed_at=datetime.utcnow(),
            sample_size=len(feature_values),
            anomaly_count=anomaly_count,
            quality_score=quality_score
        )
        
        # Store quality metrics
        self.quality_metrics[feature_id] = quality_metrics
        
        # Store in KnowledgeGraph
        await self._store_quality_metrics_in_kg(quality_metrics)
        
        return quality_metrics
    
    def _compute_completeness(self, values: List[Any]) -> float:
        """Compute completeness score"""
        if not values:
            return 0.0
        
        non_null_count = sum(1 for v in values if v is not None and v != "")
        return non_null_count / len(values)
    
    def _compute_uniqueness(self, values: List[Any]) -> float:
        """Compute uniqueness score"""
        if not values:
            return 0.0
        
        unique_count = len(set(values))
        return unique_count / len(values)
    
    def _compute_consistency(self, values: List[Any], feature_type: FeatureType) -> float:
        """Compute consistency score"""
        if not values:
            return 0.0
        
        # Check type consistency
        expected_type = self._get_python_type(feature_type)
        consistent_count = sum(1 for v in values if isinstance(v, expected_type))
        
        return consistent_count / len(values)
    
    def _compute_validity(self, values: List[Any], validation_rules: Dict[str, Any]) -> float:
        """Compute validity score"""
        if not values or not validation_rules:
            return 1.0
        
        valid_count = 0
        for value in values:
            if self._validate_value(value, validation_rules):
                valid_count += 1
        
        return valid_count / len(values)
    
    def _compute_timeliness(self, feature_id: str) -> float:
        """Compute timeliness score"""
        # Get latest feature values
        latest_timestamps = []
        for entity_values in self.feature_values.values():
            if feature_id in entity_values:
                latest_timestamps.append(entity_values[feature_id].timestamp)
        
        if not latest_timestamps:
            return 0.0
        
        latest_time = max(latest_timestamps)
        time_diff = (datetime.utcnow() - latest_time).total_seconds()
        
        # Score based on freshness threshold
        if time_diff <= self.config["feature_freshness_threshold"]:
            return 1.0
        else:
            return max(0.0, 1.0 - (time_diff / (2 * self.config["feature_freshness_threshold"])))
    
    def _compute_accuracy(self, values: List[Any], feature_def: FeatureDefinition) -> float:
        """Compute accuracy score (simplified)"""
        # For now, return 1.0 (would need ground truth for real accuracy)
        return 1.0
    
    def _count_anomalies(self, values: List[Any], feature_def: FeatureDefinition) -> int:
        """Count anomalies in feature values"""
        if not values:
            return 0
        
        # Simple anomaly detection based on statistical outliers
        if feature_def.feature_type == FeatureType.NUMERICAL:
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if len(numeric_values) < 2:
                return 0
            
            mean = np.mean(numeric_values)
            std = np.std(numeric_values)
            
            anomaly_count = 0
            for value in numeric_values:
                if abs(value - mean) > 3 * std:  # 3-sigma rule
                    anomaly_count += 1
            
            return anomaly_count
        
        return 0
    
    def _get_python_type(self, feature_type: FeatureType) -> type:
        """Get Python type for feature type"""
        type_mapping = {
            FeatureType.NUMERICAL: (int, float),
            FeatureType.CATEGORICAL: str,
            FeatureType.BINARY: bool,
            FeatureType.TEXT: str,
            FeatureType.TIMESTAMP: datetime,
            FeatureType.EMBEDDING: (list, np.ndarray),
            FeatureType.ARRAY: (list, np.ndarray),
            FeatureType.JSON: dict
        }
        return type_mapping.get(feature_type, object)
    
    def _validate_value(self, value: Any, validation_rules: Dict[str, Any]) -> bool:
        """Validate feature value against rules"""
        for rule_type, rule_value in validation_rules.items():
            if rule_type == "min_value" and value < rule_value:
                return False
            elif rule_type == "max_value" and value > rule_value:
                return False
            elif rule_type == "allowed_values" and value not in rule_value:
                return False
            elif rule_type == "pattern" and not re.match(rule_value, str(value)):
                return False
        
        return True
    
    # ===== CACHING =====
    
    async def _get_from_cache(self, entity_id: str, feature_id: str) -> Optional[Any]:
        """Get feature value from cache"""
        return self.feature_cache[entity_id].get(feature_id)
    
    async def _cache_feature(self, entity_id: str, feature_id: str, value: Any):
        """Cache feature value"""
        self.feature_cache[entity_id][feature_id] = {
            "value": value,
            "cached_at": datetime.utcnow()
        }
    
    async def _cache_management(self):
        """Manage feature cache"""
        while True:
            try:
                # Clean expired cache entries
                current_time = datetime.utcnow()
                ttl_seconds = self.config["cache_ttl"]
                
                for entity_id in list(self.feature_cache.keys()):
                    entity_cache = self.feature_cache[entity_id]
                    
                    for feature_id in list(entity_cache.keys()):
                        cached_item = entity_cache[feature_id]
                        if isinstance(cached_item, dict) and "cached_at" in cached_item:
                            if (current_time - cached_item["cached_at"]).total_seconds() > ttl_seconds:
                                del entity_cache[feature_id]
                    
                    # Remove empty entity caches
                    if not entity_cache:
                        del self.feature_cache[entity_id]
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cache management: {e}")
                await asyncio.sleep(300)
    
    # ===== TRANSFORMATION FUNCTIONS =====
    
    async def _standard_scaling(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Standard scaling transformation"""
        
        feature_col = config.get("feature_column")
        if not feature_col or feature_col not in data:
            return data
        
        scaler = StandardScaler()
        values = np.array(data[feature_col]).reshape(-1, 1)
        scaled_values = scaler.fit_transform(values).flatten()
        
        data[f"{feature_col}_scaled"] = scaled_values.tolist()
        return data
    
    async def _min_max_scaling(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Min-max scaling transformation"""
        
        feature_col = config.get("feature_column")
        if not feature_col or feature_col not in data:
            return data
        
        scaler = MinMaxScaler()
        values = np.array(data[feature_col]).reshape(-1, 1)
        scaled_values = scaler.fit_transform(values).flatten()
        
        data[f"{feature_col}_minmax"] = scaled_values.tolist()
        return data
    
    async def _one_hot_encoding(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """One-hot encoding transformation"""
        
        feature_col = config.get("feature_column")
        if not feature_col or feature_col not in data:
            return data
        
        # Get unique values
        unique_values = list(set(data[feature_col]))
        
        # Create one-hot encoded features
        for value in unique_values:
            encoded_col = f"{feature_col}_{value}"
            data[encoded_col] = [1 if x == value else 0 for x in data[feature_col]]
        
        return data
    
    async def _label_encoding(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Label encoding transformation"""
        
        feature_col = config.get("feature_column")
        if not feature_col or feature_col not in data:
            return data
        
        encoder = LabelEncoder()
        encoded_values = encoder.fit_transform(data[feature_col])
        
        data[f"{feature_col}_encoded"] = encoded_values.tolist()
        return data
    
    async def _windowing_aggregation(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Windowing aggregation transformation"""
        
        feature_col = config.get("feature_column")
        window_size = config.get("window_size", 7)
        agg_func = config.get("aggregation", "mean")
        
        if not feature_col or feature_col not in data:
            return data
        
        values = data[feature_col]
        windowed_values = []
        
        for i in range(len(values)):
            window_start = max(0, i - window_size + 1)
            window_values = values[window_start:i+1]
            
            if agg_func == "mean":
                windowed_values.append(np.mean(window_values))
            elif agg_func == "sum":
                windowed_values.append(np.sum(window_values))
            elif agg_func == "max":
                windowed_values.append(np.max(window_values))
            elif agg_func == "min":
                windowed_values.append(np.min(window_values))
            else:
                windowed_values.append(np.mean(window_values))
        
        data[f"{feature_col}_{agg_func}_{window_size}d"] = windowed_values
        return data
    
    async def _time_based_features(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Time-based features transformation"""
        
        timestamp_col = config.get("timestamp_column")
        if not timestamp_col or timestamp_col not in data:
            return data
        
        timestamps = pd.to_datetime(data[timestamp_col])
        
        # Extract time-based features
        data[f"{timestamp_col}_hour"] = timestamps.dt.hour.tolist()
        data[f"{timestamp_col}_day"] = timestamps.dt.day.tolist()
        data[f"{timestamp_col}_month"] = timestamps.dt.month.tolist()
        data[f"{timestamp_col}_year"] = timestamps.dt.year.tolist()
        data[f"{timestamp_col}_weekday"] = timestamps.dt.weekday.tolist()
        data[f"{timestamp_col}_is_weekend"] = (timestamps.dt.weekday >= 5).astype(int).tolist()
        
        return data
    
    async def _text_vectorization(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Text vectorization transformation"""
        
        text_col = config.get("text_column")
        max_features = config.get("max_features", 100)
        
        if not text_col or text_col not in data:
            return data
        
        vectorizer = TfidfVectorizer(max_features=max_features)
        vectors = vectorizer.fit_transform(data[text_col])
        
        # Add vectorized features
        for i, feature_name in enumerate(vectorizer.get_feature_names_out()):
            data[f"{text_col}_tfidf_{feature_name}"] = vectors[:, i].toarray().flatten().tolist()
        
        return data
    
    async def _pca_reduction(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """PCA dimensionality reduction"""
        
        feature_cols = config.get("feature_columns", [])
        n_components = config.get("n_components", 2)
        
        if not feature_cols:
            return data
        
        # Get feature matrix
        feature_matrix = np.array([data[col] for col in feature_cols if col in data]).T
        
        if feature_matrix.shape[1] == 0:
            return data
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(feature_matrix)
        
        # Add PCA features
        for i in range(n_components):
            data[f"pca_component_{i}"] = pca_features[:, i].tolist()
        
        return data
    
    async def _feature_selection(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Feature selection transformation"""
        
        feature_cols = config.get("feature_columns", [])
        target_col = config.get("target_column")
        k_best = config.get("k_best", 5)
        
        if not feature_cols or not target_col or target_col not in data:
            return data
        
        # Get feature matrix and target
        feature_matrix = np.array([data[col] for col in feature_cols if col in data]).T
        target = data[target_col]
        
        if feature_matrix.shape[1] == 0:
            return data
        
        # Apply feature selection
        selector = SelectKBest(score_func=f_classif, k=k_best)
        selected_features = selector.fit_transform(feature_matrix, target)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_cols = [feature_cols[i] for i in selected_indices]
        
        # Add selected features
        for i, col in enumerate(selected_cols):
            data[f"selected_{col}"] = selected_features[:, i].tolist()
        
        return data
    
    async def _custom_aggregation(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Custom aggregation transformation"""
        
        feature_col = config.get("feature_column")
        group_col = config.get("group_column")
        agg_func = config.get("aggregation", "mean")
        
        if not feature_col or not group_col or feature_col not in data or group_col not in data:
            return data
        
        # Create DataFrame for aggregation
        df = pd.DataFrame({
            "feature": data[feature_col],
            "group": data[group_col]
        })
        
        # Perform aggregation
        if agg_func == "mean":
            agg_result = df.groupby("group")["feature"].mean()
        elif agg_func == "sum":
            agg_result = df.groupby("group")["feature"].sum()
        elif agg_func == "count":
            agg_result = df.groupby("group")["feature"].count()
        elif agg_func == "std":
            agg_result = df.groupby("group")["feature"].std()
        else:
            agg_result = df.groupby("group")["feature"].mean()
        
        # Map back to original data
        group_to_agg = agg_result.to_dict()
        data[f"{feature_col}_{agg_func}_by_{group_col}"] = [
            group_to_agg.get(group, 0) for group in data[group_col]
        ]
        
        return data
    
    # ===== KNOWLEDGEGRAPH INTEGRATION =====
    
    async def _store_feature_in_kg(self, feature_def: FeatureDefinition):
        """Store feature definition in KnowledgeGraph"""
        
        try:
            vertex_data = {
                "feature_id": feature_def.feature_id,
                "feature_name": feature_def.feature_name,
                "feature_type": feature_def.feature_type.value,
                "entity_type": feature_def.entity_type,
                "description": feature_def.description,
                "feature_group": feature_def.feature_group,
                "status": feature_def.status.value,
                "version": feature_def.version,
                "created_at": feature_def.created_at.isoformat(),
                "created_by": feature_def.created_by,
                "source_table": feature_def.source_table,
                "source_column": feature_def.source_column,
                "transformation_config": feature_def.transformation_config,
                "validation_rules": feature_def.validation_rules,
                "serving_config": feature_def.serving_config,
                "tags": feature_def.tags,
                "metadata": feature_def.metadata
            }
            
            await self.kg_client.add_vertex(
                "Feature", 
                feature_def.feature_id, 
                vertex_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store feature in KnowledgeGraph: {e}")
    
    async def _store_feature_group_in_kg(self, feature_group: FeatureGroup):
        """Store feature group in KnowledgeGraph"""
        
        try:
            vertex_data = {
                "group_id": feature_group.group_id,
                "group_name": feature_group.group_name,
                "description": feature_group.description,
                "entity_type": feature_group.entity_type,
                "features": feature_group.features,
                "serving_modes": [mode.value for mode in feature_group.serving_modes],
                "refresh_frequency": feature_group.refresh_frequency,
                "created_at": feature_group.created_at.isoformat(),
                "created_by": feature_group.created_by,
                "status": feature_group.status.value,
                "tags": feature_group.tags,
                "metadata": feature_group.metadata
            }
            
            await self.kg_client.add_vertex(
                "FeatureGroup", 
                feature_group.group_id, 
                vertex_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store feature group in KnowledgeGraph: {e}")
    
    async def _store_transformation_in_kg(self, transformation: FeatureTransformation):
        """Store transformation in KnowledgeGraph"""
        
        try:
            vertex_data = {
                "transformation_id": transformation.transformation_id,
                "transformation_name": transformation.transformation_name,
                "transformation_type": transformation.transformation_type.value,
                "source_features": transformation.source_features,
                "target_feature": transformation.target_feature,
                "transformation_function": transformation.transformation_function,
                "transformation_config": transformation.transformation_config,
                "created_at": transformation.created_at.isoformat(),
                "created_by": transformation.created_by,
                "description": transformation.description,
                "validation_rules": transformation.validation_rules
            }
            
            await self.kg_client.add_vertex(
                "FeatureTransformation", 
                transformation.transformation_id, 
                vertex_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store transformation in KnowledgeGraph: {e}")
    
    async def _store_quality_metrics_in_kg(self, quality_metrics: FeatureQualityMetrics):
        """Store quality metrics in KnowledgeGraph"""
        
        try:
            vertex_data = {
                "feature_id": quality_metrics.feature_id,
                "completeness": quality_metrics.completeness,
                "uniqueness": quality_metrics.uniqueness,
                "consistency": quality_metrics.consistency,
                "validity": quality_metrics.validity,
                "timeliness": quality_metrics.timeliness,
                "accuracy": quality_metrics.accuracy,
                "computed_at": quality_metrics.computed_at.isoformat(),
                "sample_size": quality_metrics.sample_size,
                "anomaly_count": quality_metrics.anomaly_count,
                "quality_score": quality_metrics.quality_score
            }
            
            await self.kg_client.add_vertex(
                "FeatureQuality", 
                f"quality_{quality_metrics.feature_id}_{int(quality_metrics.computed_at.timestamp())}", 
                vertex_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store quality metrics in KnowledgeGraph: {e}")
    
    async def _update_feature_in_kg(self, feature_id: str, updates: Dict[str, Any]):
        """Update feature in KnowledgeGraph"""
        
        try:
            await self.kg_client.update_vertex(
                "Feature", 
                feature_id, 
                updates
            )
            
        except Exception as e:
            logger.error(f"Failed to update feature in KnowledgeGraph: {e}")
    
    # ===== BACKGROUND TASKS =====
    
    async def _feature_quality_monitoring(self):
        """Monitor feature quality"""
        
        while True:
            try:
                # Compute quality metrics for all active features
                active_features = [
                    f for f in self.feature_definitions.values()
                    if f.status == FeatureStatus.ACTIVE
                ]
                
                for feature in active_features:
                    try:
                        await self.compute_feature_quality(feature.feature_id)
                    except Exception as e:
                        logger.error(f"Quality monitoring failed for feature {feature.feature_id}: {e}")
                
                await asyncio.sleep(self.config["quality_check_frequency"])
                
            except Exception as e:
                logger.error(f"Error in feature quality monitoring: {e}")
                await asyncio.sleep(self.config["quality_check_frequency"])
    
    async def _feature_freshness_monitoring(self):
        """Monitor feature freshness"""
        
        while True:
            try:
                current_time = datetime.utcnow()
                threshold = self.config["feature_freshness_threshold"]
                
                # Check freshness for all features
                for feature_id, feature_def in self.feature_definitions.items():
                    if feature_def.status != FeatureStatus.ACTIVE:
                        continue
                    
                    # Get latest feature values
                    latest_timestamps = []
                    for entity_values in self.feature_values.values():
                        if feature_id in entity_values:
                            latest_timestamps.append(entity_values[feature_id].timestamp)
                    
                    if latest_timestamps:
                        latest_time = max(latest_timestamps)
                        time_diff = (current_time - latest_time).total_seconds()
                        
                        if time_diff > threshold:
                            # Publish freshness alert
                            await shared_pulsar_client.publish(
                                "q.ml.feature.freshness.alert",
                                {
                                    "feature_id": feature_id,
                                    "feature_name": feature_def.feature_name,
                                    "time_since_update": time_diff,
                                    "threshold": threshold,
                                    "timestamp": current_time.isoformat()
                                }
                            )
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in feature freshness monitoring: {e}")
                await asyncio.sleep(1800)
    
    async def _metrics_tracking(self):
        """Track feature store metrics"""
        
        while True:
            try:
                # Update metrics
                self.feature_store_metrics["total_features"] = len(self.feature_definitions)
                self.feature_store_metrics["active_features"] = len([
                    f for f in self.feature_definitions.values()
                    if f.status == FeatureStatus.ACTIVE
                ])
                self.feature_store_metrics["feature_groups"] = len(self.feature_groups)
                self.feature_store_metrics["transformations"] = len(self.feature_transformations)
                
                # Calculate cache hit rate (simplified)
                total_cache_entries = sum(len(cache) for cache in self.feature_cache.values())
                if total_cache_entries > 0:
                    # Simulate cache hit rate
                    self.feature_store_metrics["cache_hit_rate"] = 0.85
                
                # Calculate quality score
                if self.quality_metrics:
                    avg_quality = np.mean([m.quality_score for m in self.quality_metrics.values()])
                    self.feature_store_metrics["quality_score"] = avg_quality
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in metrics tracking: {e}")
                await asyncio.sleep(300)
    
    # ===== STORAGE MANAGEMENT =====
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for feature store"""
        
        topics = [
            "q.ml.feature.registered",
            "q.ml.feature.updated",
            "q.ml.feature.served",
            "q.ml.feature.values.stored",
            "q.ml.feature.freshness.alert",
            "q.ml.feature.quality.alert"
        ]
        
        logger.info("Feature store Pulsar topics configured")
    
    async def _load_features(self):
        """Load existing features from storage"""
        
        features_file = self.storage_path / "features.json"
        if features_file.exists():
            try:
                with open(features_file, 'r') as f:
                    features_data = json.load(f)
                
                for feature_data in features_data:
                    feature = FeatureDefinition(**feature_data)
                    self.feature_definitions[feature.feature_id] = feature
                
                logger.info(f"Loaded {len(self.feature_definitions)} features")
            except Exception as e:
                logger.error(f"Failed to load features: {e}")
    
    async def _save_features(self):
        """Save features to storage"""
        
        features_file = self.storage_path / "features.json"
        try:
            features_data = [asdict(f) for f in self.feature_definitions.values()]
            
            with open(features_file, 'w') as f:
                json.dump(features_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.feature_definitions)} features")
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
    
    # ===== PUBLIC API =====
    
    async def get_feature_groups(
        self,
        entity_type: Optional[str] = None,
        status: Optional[FeatureStatus] = None,
        limit: int = 100
    ) -> List[FeatureGroup]:
        """Get feature groups with optional filtering"""
        
        groups = list(self.feature_groups.values())
        
        if entity_type:
            groups = [g for g in groups if g.entity_type == entity_type]
        
        if status:
            groups = [g for g in groups if g.status == status]
        
        # Sort by creation time (newest first)
        groups.sort(key=lambda x: x.created_at, reverse=True)
        
        return groups[:limit]
    
    async def get_feature_quality_metrics(
        self,
        feature_id: Optional[str] = None,
        limit: int = 100
    ) -> List[FeatureQualityMetrics]:
        """Get feature quality metrics"""
        
        metrics = list(self.quality_metrics.values())
        
        if feature_id:
            metrics = [m for m in metrics if m.feature_id == feature_id]
        
        # Sort by computation time (newest first)
        metrics.sort(key=lambda x: x.computed_at, reverse=True)
        
        return metrics[:limit]
    
    async def get_feature_store_metrics(self) -> Dict[str, Any]:
        """Get feature store service metrics"""
        
        return {
            "service_metrics": self.feature_store_metrics,
            "config": self.config,
            "cache_size": sum(len(cache) for cache in self.feature_cache.values()),
            "stored_values": sum(len(values) for values in self.feature_values.values())
        }

# Global instance
feature_store_service = FeatureStoreService() 