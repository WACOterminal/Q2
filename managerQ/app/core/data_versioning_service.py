"""
Data Versioning Service for Q Platform

This service provides comprehensive data versioning, lineage tracking, and drift detection:
- Data version management with hashing and checksums
- Integration with KnowledgeGraphQ for lineage tracking
- Data drift detection and monitoring
- Dataset dependency tracking
- Model-data relationship management
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from collections import defaultdict

# Statistical libraries for drift detection
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jensen_shannon_distance

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)

class DataVersionStatus(Enum):
    """Data version status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    CORRUPTED = "corrupted"

class DriftSeverity(Enum):
    """Data drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DataVersion:
    """Data version metadata"""
    version_id: str
    dataset_name: str
    version_tag: str
    file_path: str
    checksum: str
    schema_hash: str
    row_count: int
    column_count: int
    file_size: int
    status: DataVersionStatus
    created_at: datetime
    created_by: str
    description: Optional[str] = None
    parent_version_id: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class DataLineage:
    """Data lineage information"""
    lineage_id: str
    source_version_id: str
    target_version_id: str
    transformation_type: str
    transformation_config: Dict[str, Any]
    created_at: datetime
    created_by: str
    description: Optional[str] = None

@dataclass
class DataDriftReport:
    """Data drift detection report"""
    report_id: str
    dataset_name: str
    baseline_version_id: str
    current_version_id: str
    drift_severity: DriftSeverity
    drift_score: float
    affected_columns: List[str]
    drift_details: Dict[str, Any]
    recommendations: List[str]
    created_at: datetime

@dataclass
class DatasetDependency:
    """Dataset dependency relationship"""
    dependency_id: str
    source_dataset: str
    target_dataset: str
    dependency_type: str
    relationship_strength: float
    created_at: datetime
    metadata: Dict[str, Any] = None

class DataVersioningService:
    """
    Data Versioning Service for comprehensive data management
    """
    
    def __init__(self, 
                 storage_path: str = "data/versions",
                 kg_client: Optional[KnowledgeGraphClient] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # KnowledgeGraph client for lineage tracking
        self.kg_client = kg_client or KnowledgeGraphClient()
        
        # Data registries
        self.data_versions: Dict[str, DataVersion] = {}
        self.data_lineage: Dict[str, DataLineage] = {}
        self.drift_reports: Dict[str, DataDriftReport] = {}
        self.dataset_dependencies: Dict[str, DatasetDependency] = {}
        
        # Active dataset tracking
        self.active_versions: Dict[str, str] = {}  # dataset_name -> version_id
        
        # Drift detection configuration
        self.drift_config = {
            "statistical_tests": ["kolmogorov_smirnov", "chi_square", "jensen_shannon"],
            "drift_thresholds": {
                "low": 0.1,
                "medium": 0.3,
                "high": 0.5,
                "critical": 0.7
            },
            "monitoring_frequency": 3600,  # seconds
            "alert_channels": ["q.ml.data.drift.alert"]
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Performance metrics
        self.versioning_metrics = {
            "total_versions": 0,
            "active_datasets": 0,
            "drift_alerts": 0,
            "lineage_entries": 0,
            "storage_size": 0
        }
        
    async def initialize(self):
        """Initialize the data versioning service"""
        logger.info("Initializing Data Versioning Service")
        
        # Load existing data versions
        await self._load_versions()
        
        # Initialize KnowledgeGraph client
        await self.kg_client.initialize()
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        # Start background tasks
        self.background_tasks.add(asyncio.create_task(self._drift_monitoring()))
        self.background_tasks.add(asyncio.create_task(self._lineage_tracking()))
        self.background_tasks.add(asyncio.create_task(self._metrics_tracking()))
        
        logger.info("Data Versioning Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the data versioning service"""
        logger.info("Shutting down Data Versioning Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save data versions
        await self._save_versions()
        
        logger.info("Data Versioning Service shut down successfully")
    
    # ===== VERSION MANAGEMENT =====
    
    async def create_data_version(
        self,
        dataset_name: str,
        file_path: str,
        version_tag: str,
        created_by: str,
        description: Optional[str] = None,
        parent_version_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new data version
        
        Args:
            dataset_name: Name of the dataset
            file_path: Path to the data file
            version_tag: Version tag (e.g., "v1.0", "latest")
            created_by: Creator identifier
            description: Optional description
            parent_version_id: Parent version for lineage
            tags: Optional tags
            metadata: Optional metadata
            
        Returns:
            Version ID
        """
        version_id = f"dv_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Creating data version: {dataset_name} -> {version_tag}")
        
        # Calculate file checksums and metadata
        file_info = await self._analyze_file(file_path)
        
        # Create data version
        data_version = DataVersion(
            version_id=version_id,
            dataset_name=dataset_name,
            version_tag=version_tag,
            file_path=file_path,
            checksum=file_info["checksum"],
            schema_hash=file_info["schema_hash"],
            row_count=file_info["row_count"],
            column_count=file_info["column_count"],
            file_size=file_info["file_size"],
            status=DataVersionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            created_by=created_by,
            description=description,
            parent_version_id=parent_version_id,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Store version
        self.data_versions[version_id] = data_version
        
        # Update active version
        self.active_versions[dataset_name] = version_id
        
        # Create lineage relationship if parent exists
        if parent_version_id:
            await self._create_lineage_relationship(
                parent_version_id, version_id, "version_update", created_by
            )
        
        # Store in KnowledgeGraph
        await self._store_version_in_kg(data_version)
        
        # Publish version created event
        await shared_pulsar_client.publish(
            "q.ml.data.version.created",
            {
                "version_id": version_id,
                "dataset_name": dataset_name,
                "version_tag": version_tag,
                "file_path": file_path,
                "checksum": file_info["checksum"],
                "created_by": created_by,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Created data version: {version_id}")
        return version_id
    
    async def get_data_version(self, version_id: str) -> Optional[DataVersion]:
        """Get data version by ID"""
        return self.data_versions.get(version_id)
    
    async def get_active_version(self, dataset_name: str) -> Optional[DataVersion]:
        """Get active version for a dataset"""
        version_id = self.active_versions.get(dataset_name)
        if version_id:
            return self.data_versions.get(version_id)
        return None
    
    async def list_versions(
        self,
        dataset_name: Optional[str] = None,
        status: Optional[DataVersionStatus] = None,
        limit: int = 100
    ) -> List[DataVersion]:
        """List data versions with optional filtering"""
        versions = list(self.data_versions.values())
        
        if dataset_name:
            versions = [v for v in versions if v.dataset_name == dataset_name]
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x.created_at, reverse=True)
        
        return versions[:limit]
    
    async def update_version_status(
        self,
        version_id: str,
        status: DataVersionStatus,
        updated_by: str
    ) -> bool:
        """Update version status"""
        if version_id not in self.data_versions:
            return False
        
        self.data_versions[version_id].status = status
        
        # Update in KnowledgeGraph
        await self._update_version_in_kg(version_id, {"status": status.value})
        
        # Publish status update event
        await shared_pulsar_client.publish(
            "q.ml.data.version.status_updated",
            {
                "version_id": version_id,
                "status": status.value,
                "updated_by": updated_by,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return True
    
    # ===== LINEAGE TRACKING =====
    
    async def _create_lineage_relationship(
        self,
        source_version_id: str,
        target_version_id: str,
        transformation_type: str,
        created_by: str,
        transformation_config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> str:
        """Create lineage relationship between versions"""
        
        lineage_id = f"lineage_{uuid.uuid4().hex[:12]}"
        
        lineage = DataLineage(
            lineage_id=lineage_id,
            source_version_id=source_version_id,
            target_version_id=target_version_id,
            transformation_type=transformation_type,
            transformation_config=transformation_config or {},
            created_at=datetime.utcnow(),
            created_by=created_by,
            description=description
        )
        
        self.data_lineage[lineage_id] = lineage
        
        # Store in KnowledgeGraph
        await self._store_lineage_in_kg(lineage)
        
        return lineage_id
    
    async def get_lineage_upstream(self, version_id: str) -> List[DataLineage]:
        """Get upstream lineage for a version"""
        return [
            lineage for lineage in self.data_lineage.values()
            if lineage.target_version_id == version_id
        ]
    
    async def get_lineage_downstream(self, version_id: str) -> List[DataLineage]:
        """Get downstream lineage for a version"""
        return [
            lineage for lineage in self.data_lineage.values()
            if lineage.source_version_id == version_id
        ]
    
    async def get_lineage_tree(self, version_id: str) -> Dict[str, Any]:
        """Get complete lineage tree for a version"""
        
        # Get all related lineage entries
        upstream = await self.get_lineage_upstream(version_id)
        downstream = await self.get_lineage_downstream(version_id)
        
        # Build tree structure
        tree = {
            "version_id": version_id,
            "version_info": self.data_versions.get(version_id),
            "upstream": [],
            "downstream": []
        }
        
        # Add upstream relationships
        for lineage in upstream:
            upstream_tree = await self.get_lineage_tree(lineage.source_version_id)
            tree["upstream"].append({
                "lineage_info": lineage,
                "tree": upstream_tree
            })
        
        # Add downstream relationships
        for lineage in downstream:
            downstream_tree = await self.get_lineage_tree(lineage.target_version_id)
            tree["downstream"].append({
                "lineage_info": lineage,
                "tree": downstream_tree
            })
        
        return tree
    
    # ===== DRIFT DETECTION =====
    
    async def detect_data_drift(
        self,
        baseline_version_id: str,
        current_version_id: str,
        columns: Optional[List[str]] = None
    ) -> DataDriftReport:
        """
        Detect data drift between two versions
        
        Args:
            baseline_version_id: Baseline version ID
            current_version_id: Current version ID
            columns: Optional list of columns to analyze
            
        Returns:
            Data drift report
        """
        report_id = f"drift_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Detecting data drift: {baseline_version_id} -> {current_version_id}")
        
        # Load data versions
        baseline_version = self.data_versions.get(baseline_version_id)
        current_version = self.data_versions.get(current_version_id)
        
        if not baseline_version or not current_version:
            raise ValueError("Invalid version IDs")
        
        # Load datasets
        baseline_data = await self._load_dataset(baseline_version.file_path)
        current_data = await self._load_dataset(current_version.file_path)
        
        # Analyze drift
        drift_results = await self._analyze_drift(
            baseline_data, current_data, columns
        )
        
        # Create drift report
        drift_report = DataDriftReport(
            report_id=report_id,
            dataset_name=baseline_version.dataset_name,
            baseline_version_id=baseline_version_id,
            current_version_id=current_version_id,
            drift_severity=drift_results["severity"],
            drift_score=drift_results["score"],
            affected_columns=drift_results["affected_columns"],
            drift_details=drift_results["details"],
            recommendations=drift_results["recommendations"],
            created_at=datetime.utcnow()
        )
        
        # Store report
        self.drift_reports[report_id] = drift_report
        
        # Store in KnowledgeGraph
        await self._store_drift_report_in_kg(drift_report)
        
        # Publish drift detection event
        await shared_pulsar_client.publish(
            "q.ml.data.drift.detected",
            {
                "report_id": report_id,
                "dataset_name": baseline_version.dataset_name,
                "drift_severity": drift_results["severity"].value,
                "drift_score": drift_results["score"],
                "affected_columns": drift_results["affected_columns"],
                "baseline_version": baseline_version_id,
                "current_version": current_version_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Drift detection completed: {report_id}")
        return drift_report
    
    async def _analyze_drift(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze data drift between datasets"""
        
        if columns is None:
            columns = baseline_data.columns.tolist()
        
        drift_scores = {}
        affected_columns = []
        drift_details = {}
        
        for column in columns:
            if column not in baseline_data.columns or column not in current_data.columns:
                continue
                
            baseline_col = baseline_data[column].dropna()
            current_col = current_data[column].dropna()
            
            # Calculate drift score for this column
            column_drift = await self._calculate_column_drift(baseline_col, current_col)
            drift_scores[column] = column_drift["score"]
            drift_details[column] = column_drift["details"]
            
            # Check if drift exceeds threshold
            if column_drift["score"] > self.drift_config["drift_thresholds"]["low"]:
                affected_columns.append(column)
        
        # Calculate overall drift score
        overall_score = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        
        # Determine severity
        severity = DriftSeverity.LOW
        if overall_score > self.drift_config["drift_thresholds"]["critical"]:
            severity = DriftSeverity.CRITICAL
        elif overall_score > self.drift_config["drift_thresholds"]["high"]:
            severity = DriftSeverity.HIGH
        elif overall_score > self.drift_config["drift_thresholds"]["medium"]:
            severity = DriftSeverity.MEDIUM
        
        # Generate recommendations
        recommendations = await self._generate_drift_recommendations(
            severity, affected_columns, drift_details
        )
        
        return {
            "score": overall_score,
            "severity": severity,
            "affected_columns": affected_columns,
            "details": drift_details,
            "recommendations": recommendations
        }
    
    async def _calculate_column_drift(
        self,
        baseline_col: pd.Series,
        current_col: pd.Series
    ) -> Dict[str, Any]:
        """Calculate drift score for a single column"""
        
        drift_score = 0.0
        details = {}
        
        # Determine if column is numeric or categorical
        if baseline_col.dtype in ['int64', 'float64']:
            # Numeric column - use statistical tests
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p_value = stats.ks_2samp(baseline_col, current_col)
            details["ks_statistic"] = ks_stat
            details["ks_p_value"] = ks_p_value
            
            # Mean and std comparison
            baseline_mean = baseline_col.mean()
            current_mean = current_col.mean()
            baseline_std = baseline_col.std()
            current_std = current_col.std()
            
            mean_diff = abs(baseline_mean - current_mean) / (baseline_std + 1e-8)
            std_diff = abs(baseline_std - current_std) / (baseline_std + 1e-8)
            
            details["mean_difference"] = mean_diff
            details["std_difference"] = std_diff
            details["baseline_mean"] = baseline_mean
            details["current_mean"] = current_mean
            details["baseline_std"] = baseline_std
            details["current_std"] = current_std
            
            # Combine scores
            drift_score = max(ks_stat, mean_diff, std_diff)
            
        else:
            # Categorical column - use distribution comparison
            
            baseline_counts = baseline_col.value_counts(normalize=True)
            current_counts = current_col.value_counts(normalize=True)
            
            # Get all unique values
            all_values = set(baseline_counts.index) | set(current_counts.index)
            
            # Create aligned distributions
            baseline_dist = np.array([baseline_counts.get(val, 0) for val in all_values])
            current_dist = np.array([current_counts.get(val, 0) for val in all_values])
            
            # Jensen-Shannon divergence
            js_divergence = jensen_shannon_distance(baseline_dist, current_dist)
            
            details["js_divergence"] = js_divergence
            details["baseline_unique_count"] = len(baseline_counts)
            details["current_unique_count"] = len(current_counts)
            details["new_categories"] = list(set(current_counts.index) - set(baseline_counts.index))
            details["missing_categories"] = list(set(baseline_counts.index) - set(current_counts.index))
            
            drift_score = js_divergence
        
        return {
            "score": drift_score,
            "details": details
        }
    
    async def _generate_drift_recommendations(
        self,
        severity: DriftSeverity,
        affected_columns: List[str],
        drift_details: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on drift analysis"""
        
        recommendations = []
        
        if severity == DriftSeverity.CRITICAL:
            recommendations.append("CRITICAL: Immediate model retraining recommended")
            recommendations.append("Review data collection and preprocessing pipelines")
            recommendations.append("Consider model rollback if performance degrades")
        
        elif severity == DriftSeverity.HIGH:
            recommendations.append("HIGH: Schedule model retraining within 24 hours")
            recommendations.append("Increase monitoring frequency for affected features")
            recommendations.append("Analyze root cause of distribution changes")
        
        elif severity == DriftSeverity.MEDIUM:
            recommendations.append("MEDIUM: Schedule model retraining within 1 week")
            recommendations.append("Monitor model performance metrics closely")
            recommendations.append("Consider feature engineering adjustments")
        
        else:
            recommendations.append("LOW: Continue monitoring, no immediate action required")
            recommendations.append("Document changes for future reference")
        
        # Column-specific recommendations
        if len(affected_columns) > 0:
            recommendations.append(f"Focus on columns: {', '.join(affected_columns[:5])}")
        
        return recommendations
    
    # ===== DEPENDENCY TRACKING =====
    
    async def create_dataset_dependency(
        self,
        source_dataset: str,
        target_dataset: str,
        dependency_type: str,
        relationship_strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create dataset dependency relationship"""
        
        dependency_id = f"dep_{uuid.uuid4().hex[:12]}"
        
        dependency = DatasetDependency(
            dependency_id=dependency_id,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            dependency_type=dependency_type,
            relationship_strength=relationship_strength,
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.dataset_dependencies[dependency_id] = dependency
        
        # Store in KnowledgeGraph
        await self._store_dependency_in_kg(dependency)
        
        return dependency_id
    
    async def get_dataset_dependencies(
        self,
        dataset_name: str,
        direction: str = "both"
    ) -> List[DatasetDependency]:
        """Get dataset dependencies"""
        
        dependencies = []
        
        if direction in ["upstream", "both"]:
            dependencies.extend([
                dep for dep in self.dataset_dependencies.values()
                if dep.target_dataset == dataset_name
            ])
        
        if direction in ["downstream", "both"]:
            dependencies.extend([
                dep for dep in self.dataset_dependencies.values()
                if dep.source_dataset == dataset_name
            ])
        
        return dependencies
    
    # ===== UTILITY METHODS =====
    
    async def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze file and extract metadata"""
        
        file_path = Path(file_path)
        
        # Calculate file checksum
        checksum = await self._calculate_file_checksum(file_path)
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Load and analyze data
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Calculate schema hash
            schema_info = {
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "shape": df.shape
            }
            schema_hash = hashlib.sha256(
                json.dumps(schema_info, sort_keys=True).encode()
            ).hexdigest()
            
            return {
                "checksum": checksum,
                "schema_hash": schema_hash,
                "row_count": len(df),
                "column_count": len(df.columns),
                "file_size": file_size,
                "schema_info": schema_info
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return {
                "checksum": checksum,
                "schema_hash": "",
                "row_count": 0,
                "column_count": 0,
                "file_size": file_size,
                "schema_info": {}
            }
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset from file"""
        
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # ===== KNOWLEDGEGRAPH INTEGRATION =====
    
    async def _store_version_in_kg(self, data_version: DataVersion):
        """Store data version in KnowledgeGraph"""
        
        try:
            # Create vertex for data version
            vertex_data = {
                "version_id": data_version.version_id,
                "dataset_name": data_version.dataset_name,
                "version_tag": data_version.version_tag,
                "checksum": data_version.checksum,
                "schema_hash": data_version.schema_hash,
                "row_count": data_version.row_count,
                "column_count": data_version.column_count,
                "file_size": data_version.file_size,
                "status": data_version.status.value,
                "created_at": data_version.created_at.isoformat(),
                "created_by": data_version.created_by,
                "description": data_version.description,
                "tags": data_version.tags,
                "metadata": data_version.metadata
            }
            
            await self.kg_client.add_vertex(
                "DataVersion", 
                data_version.version_id, 
                vertex_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store version in KnowledgeGraph: {e}")
    
    async def _store_lineage_in_kg(self, lineage: DataLineage):
        """Store lineage relationship in KnowledgeGraph"""
        
        try:
            # Create edge for lineage relationship
            edge_data = {
                "lineage_id": lineage.lineage_id,
                "transformation_type": lineage.transformation_type,
                "transformation_config": lineage.transformation_config,
                "created_at": lineage.created_at.isoformat(),
                "created_by": lineage.created_by,
                "description": lineage.description
            }
            
            await self.kg_client.add_edge(
                "DataLineage",
                lineage.source_version_id,
                lineage.target_version_id,
                edge_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store lineage in KnowledgeGraph: {e}")
    
    async def _store_drift_report_in_kg(self, drift_report: DataDriftReport):
        """Store drift report in KnowledgeGraph"""
        
        try:
            # Create vertex for drift report
            vertex_data = {
                "report_id": drift_report.report_id,
                "dataset_name": drift_report.dataset_name,
                "drift_severity": drift_report.drift_severity.value,
                "drift_score": drift_report.drift_score,
                "affected_columns": drift_report.affected_columns,
                "drift_details": drift_report.drift_details,
                "recommendations": drift_report.recommendations,
                "created_at": drift_report.created_at.isoformat()
            }
            
            await self.kg_client.add_vertex(
                "DriftReport", 
                drift_report.report_id, 
                vertex_data
            )
            
            # Create edges to versions
            await self.kg_client.add_edge(
                "DriftAnalysis",
                drift_report.baseline_version_id,
                drift_report.report_id,
                {"role": "baseline"}
            )
            
            await self.kg_client.add_edge(
                "DriftAnalysis",
                drift_report.current_version_id,
                drift_report.report_id,
                {"role": "current"}
            )
            
        except Exception as e:
            logger.error(f"Failed to store drift report in KnowledgeGraph: {e}")
    
    async def _store_dependency_in_kg(self, dependency: DatasetDependency):
        """Store dataset dependency in KnowledgeGraph"""
        
        try:
            # Create edge for dependency relationship
            edge_data = {
                "dependency_id": dependency.dependency_id,
                "dependency_type": dependency.dependency_type,
                "relationship_strength": dependency.relationship_strength,
                "created_at": dependency.created_at.isoformat(),
                "metadata": dependency.metadata
            }
            
            await self.kg_client.add_edge(
                "DatasetDependency",
                dependency.source_dataset,
                dependency.target_dataset,
                edge_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store dependency in KnowledgeGraph: {e}")
    
    async def _update_version_in_kg(self, version_id: str, updates: Dict[str, Any]):
        """Update version in KnowledgeGraph"""
        
        try:
            await self.kg_client.update_vertex(
                "DataVersion", 
                version_id, 
                updates
            )
            
        except Exception as e:
            logger.error(f"Failed to update version in KnowledgeGraph: {e}")
    
    # ===== BACKGROUND TASKS =====
    
    async def _drift_monitoring(self):
        """Monitor for data drift in active datasets"""
        
        while True:
            try:
                # Get all active datasets
                active_datasets = list(self.active_versions.keys())
                
                for dataset_name in active_datasets:
                    current_version_id = self.active_versions[dataset_name]
                    current_version = self.data_versions[current_version_id]
                    
                    # Find previous version for comparison
                    previous_versions = [
                        v for v in self.data_versions.values()
                        if v.dataset_name == dataset_name 
                        and v.version_id != current_version_id
                        and v.status == DataVersionStatus.ACTIVE
                    ]
                    
                    if previous_versions:
                        # Get most recent previous version
                        previous_version = max(previous_versions, key=lambda x: x.created_at)
                        
                        # Check if drift detection is due
                        time_since_last_check = datetime.utcnow() - current_version.created_at
                        
                        if time_since_last_check.total_seconds() > self.drift_config["monitoring_frequency"]:
                            # Perform drift detection
                            try:
                                await self.detect_data_drift(
                                    previous_version.version_id,
                                    current_version_id
                                )
                            except Exception as e:
                                logger.error(f"Drift detection failed for {dataset_name}: {e}")
                
                await asyncio.sleep(self.drift_config["monitoring_frequency"])
                
            except Exception as e:
                logger.error(f"Error in drift monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _lineage_tracking(self):
        """Track and update lineage relationships"""
        
        while True:
            try:
                # Update lineage metrics
                self.versioning_metrics["lineage_entries"] = len(self.data_lineage)
                
                # Cleanup old lineage entries
                cutoff_time = datetime.utcnow() - timedelta(days=365)
                
                old_lineage = [
                    lineage_id for lineage_id, lineage in self.data_lineage.items()
                    if lineage.created_at < cutoff_time
                ]
                
                for lineage_id in old_lineage:
                    del self.data_lineage[lineage_id]
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                logger.error(f"Error in lineage tracking: {e}")
                await asyncio.sleep(3600)
    
    async def _metrics_tracking(self):
        """Track versioning metrics"""
        
        while True:
            try:
                # Update metrics
                self.versioning_metrics["total_versions"] = len(self.data_versions)
                self.versioning_metrics["active_datasets"] = len(self.active_versions)
                
                # Calculate storage size
                total_size = sum(v.file_size for v in self.data_versions.values())
                self.versioning_metrics["storage_size"] = total_size
                
                # Count drift alerts
                critical_reports = [
                    r for r in self.drift_reports.values()
                    if r.drift_severity == DriftSeverity.CRITICAL
                ]
                self.versioning_metrics["drift_alerts"] = len(critical_reports)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in metrics tracking: {e}")
                await asyncio.sleep(300)
    
    # ===== STORAGE MANAGEMENT =====
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for data versioning"""
        
        topics = [
            "q.ml.data.version.created",
            "q.ml.data.version.status_updated",
            "q.ml.data.drift.detected",
            "q.ml.data.drift.alert",
            "q.ml.data.lineage.created"
        ]
        
        logger.info("Data versioning Pulsar topics configured")
    
    async def _load_versions(self):
        """Load existing versions from storage"""
        
        versions_file = self.storage_path / "versions.json"
        if versions_file.exists():
            try:
                with open(versions_file, 'r') as f:
                    versions_data = json.load(f)
                
                for version_data in versions_data:
                    version = DataVersion(**version_data)
                    self.data_versions[version.version_id] = version
                
                logger.info(f"Loaded {len(self.data_versions)} versions")
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")
    
    async def _save_versions(self):
        """Save versions to storage"""
        
        versions_file = self.storage_path / "versions.json"
        try:
            versions_data = [asdict(v) for v in self.data_versions.values()]
            
            with open(versions_file, 'w') as f:
                json.dump(versions_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.data_versions)} versions")
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")
    
    # ===== PUBLIC API =====
    
    async def get_drift_reports(
        self,
        dataset_name: Optional[str] = None,
        severity: Optional[DriftSeverity] = None,
        limit: int = 100
    ) -> List[DataDriftReport]:
        """Get drift reports with optional filtering"""
        
        reports = list(self.drift_reports.values())
        
        if dataset_name:
            reports = [r for r in reports if r.dataset_name == dataset_name]
        
        if severity:
            reports = [r for r in reports if r.drift_severity == severity]
        
        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x.created_at, reverse=True)
        
        return reports[:limit]
    
    async def get_versioning_metrics(self) -> Dict[str, Any]:
        """Get versioning service metrics"""
        
        return {
            "service_metrics": self.versioning_metrics,
            "drift_config": self.drift_config,
            "active_datasets": len(self.active_versions),
            "total_versions": len(self.data_versions),
            "recent_drift_reports": len([
                r for r in self.drift_reports.values()
                if (datetime.utcnow() - r.created_at).total_seconds() < 86400
            ])
        }

# Global instance
data_versioning_service = DataVersioningService() 