"""
Predictive Maintenance Service

This service provides comprehensive predictive maintenance capabilities for the Q Platform:
- Proactive system monitoring and health assessment
- Anomaly detection and pattern recognition
- Failure prediction using machine learning
- Automated maintenance scheduling and recommendations
- Resource degradation tracking
- Performance trend analysis
- Alert and notification management
- Maintenance history and analytics
- Integration with system components for health monitoring
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import statistics
import math

# ML libraries for anomaly detection and prediction
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    sklearn_available = True
except ImportError:
    sklearn_available = False
    logging.warning("Scikit-learn not available - ML-based predictions will be limited")

try:
    import pandas as pd
    pandas_available = True
except ImportError:
    pandas_available = False
    logging.warning("Pandas not available - data analysis capabilities will be limited")

# System monitoring
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
    logging.warning("psutil not available - system monitoring will be limited")

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """System component types"""
    DATABASE = "database"
    SERVICE = "service"
    STORAGE = "storage"
    NETWORK = "network"
    COMPUTE = "compute"
    MEMORY = "memory"
    DISK = "disk"
    MODEL = "model"
    WORKFLOW = "workflow"
    API = "api"

class HealthStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"

class MaintenanceType(Enum):
    """Maintenance types"""
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    SCHEDULED = "scheduled"

class MaintenanceStatus(Enum):
    """Maintenance status"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AnomalyType(Enum):
    """Anomaly types"""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    BEHAVIOR = "behavior"
    PATTERN = "pattern"
    THRESHOLD = "threshold"

@dataclass
class SystemMetric:
    """System metric representation"""
    metric_id: str
    component_id: str
    component_type: ComponentType
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ComponentHealth:
    """Component health assessment"""
    component_id: str
    component_type: ComponentType
    name: str
    status: HealthStatus
    health_score: float
    last_check: datetime
    metrics: List[SystemMetric]
    issues: List[str] = None
    recommendations: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Anomaly:
    """Anomaly detection result"""
    anomaly_id: str
    component_id: str
    anomaly_type: AnomalyType
    description: str
    severity: AlertSeverity
    detected_at: datetime
    metric_values: Dict[str, float]
    confidence_score: float
    predicted_impact: str
    recommended_actions: List[str] = None
    
    def __post_init__(self):
        if self.recommended_actions is None:
            self.recommended_actions = []

@dataclass
class MaintenanceTask:
    """Maintenance task representation"""
    task_id: str
    component_id: str
    maintenance_type: MaintenanceType
    title: str
    description: str
    priority: int
    estimated_duration: timedelta
    scheduled_time: datetime
    status: MaintenanceStatus
    assigned_to: Optional[str] = None
    dependencies: List[str] = None
    resources_required: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.resources_required is None:
            self.resources_required = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MaintenanceWindow:
    """Maintenance window definition"""
    window_id: str
    name: str
    start_time: datetime
    end_time: datetime
    components: List[str]
    maintenance_type: MaintenanceType
    impact_assessment: str
    approval_required: bool = True
    created_by: Optional[str] = None
    
@dataclass
class PredictionModel:
    """Prediction model metadata"""
    model_id: str
    component_type: ComponentType
    target_metric: str
    model_type: str
    accuracy: float
    last_trained: datetime
    training_data_size: int
    prediction_horizon: int  # hours
    
@dataclass
class FailurePrediction:
    """Failure prediction result"""
    prediction_id: str
    component_id: str
    predicted_failure_time: datetime
    confidence: float
    failure_type: str
    contributing_factors: List[str]
    recommended_actions: List[str]
    created_at: datetime

class PredictiveMaintenanceService:
    """
    Comprehensive Predictive Maintenance Service
    """
    
    def __init__(self, storage_path: str = "predictive_maintenance"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Component management
        self.components: Dict[str, ComponentHealth] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Anomaly detection
        self.anomalies: Dict[str, Anomaly] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        
        # Maintenance management
        self.maintenance_tasks: Dict[str, MaintenanceTask] = {}
        self.maintenance_windows: Dict[str, MaintenanceWindow] = {}
        self.maintenance_history: List[Dict[str, Any]] = []
        
        # ML models for prediction
        self.prediction_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, PredictionModel] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Failure predictions
        self.failure_predictions: Dict[str, FailurePrediction] = {}
        
        # Configuration
        self.config = {
            "monitoring_interval": 60,  # seconds
            "anomaly_detection_window": 3600,  # seconds
            "health_check_interval": 300,  # seconds
            "prediction_horizon": 24,  # hours
            "min_training_samples": 100,
            "anomaly_threshold": 0.1,
            "health_score_threshold": 0.8,
            "enable_auto_scheduling": True,
            "enable_failure_prediction": True,
            "alert_cooldown": 1800,  # seconds
            "max_concurrent_maintenance": 3
        }
        
        # Performance metrics
        self.metrics = {
            "total_components": 0,
            "healthy_components": 0,
            "anomalies_detected": 0,
            "maintenance_tasks_completed": 0,
            "predictions_made": 0,
            "prediction_accuracy": 0.0,
            "average_response_time": 0.0,
            "uptime_percentage": 99.0
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Service integrations
        self.vault_client = VaultClient()
        
        # Alert management
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        logger.info("Predictive Maintenance Service initialized")
    
    async def initialize(self):
        """Initialize the predictive maintenance service"""
        logger.info("Initializing Predictive Maintenance Service")
        
        # Load existing data
        await self._load_maintenance_data()
        
        # Initialize ML models
        await self._initialize_ml_models()
        
        # Register system components
        await self._register_system_components()
        
        # Start monitoring
        await self._start_monitoring()
        
        logger.info("Predictive Maintenance Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the predictive maintenance service"""
        logger.info("Shutting down Predictive Maintenance Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Save data
        await self._save_maintenance_data()
        
        logger.info("Predictive Maintenance Service shutdown complete")
    
    # ===== COMPONENT MANAGEMENT =====
    
    async def register_component(self, component: ComponentHealth) -> bool:
        """Register a system component for monitoring"""
        
        try:
            self.components[component.component_id] = component
            self.metrics["total_components"] += 1
            
            # Initialize monitoring for component
            await self._initialize_component_monitoring(component.component_id)
            
            logger.info(f"Registered component: {component.component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering component: {e}")
            return False
    
    async def update_component_health(self, component_id: str, metrics: List[SystemMetric]) -> bool:
        """Update component health based on metrics"""
        
        try:
            if component_id not in self.components:
                logger.warning(f"Component {component_id} not registered")
                return False
            
            component = self.components[component_id]
            
            # Store metrics
            for metric in metrics:
                self.metrics_history[f"{component_id}_{metric.metric_name}"].append(metric)
            
            # Calculate health score
            health_score = await self._calculate_health_score(component_id, metrics)
            
            # Update component
            component.metrics = metrics
            component.health_score = health_score
            component.last_check = datetime.utcnow()
            component.status = await self._determine_health_status(health_score, metrics)
            
            # Update global metrics
            await self._update_global_metrics()
            
            # Check for anomalies
            await self._check_for_anomalies(component_id, metrics)
            
            # Generate predictions if enabled
            if self.config["enable_failure_prediction"]:
                await self._generate_failure_prediction(component_id)
            
            # Auto-schedule maintenance if needed
            if self.config["enable_auto_scheduling"]:
                await self._auto_schedule_maintenance(component_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating component health: {e}")
            return False
    
    async def _calculate_health_score(self, component_id: str, metrics: List[SystemMetric]) -> float:
        """Calculate health score for a component"""
        
        try:
            if not metrics:
                return 0.5  # Default neutral score
            
            component = self.components[component_id]
            scores = []
            
            # Define health score calculation based on component type
            if component.component_type == ComponentType.COMPUTE:
                cpu_metrics = [m for m in metrics if 'cpu' in m.metric_name.lower()]
                memory_metrics = [m for m in metrics if 'memory' in m.metric_name.lower()]
                
                # CPU health (lower usage is better up to a point)
                if cpu_metrics:
                    cpu_usage = cpu_metrics[0].value
                    cpu_score = 1.0 - min(cpu_usage / 100.0, 1.0)  # Assuming percentage
                    scores.append(cpu_score)
                
                # Memory health
                if memory_metrics:
                    memory_usage = memory_metrics[0].value
                    memory_score = 1.0 - min(memory_usage / 100.0, 1.0)
                    scores.append(memory_score)
            
            elif component.component_type == ComponentType.STORAGE:
                disk_metrics = [m for m in metrics if 'disk' in m.metric_name.lower()]
                io_metrics = [m for m in metrics if 'io' in m.metric_name.lower()]
                
                # Disk space health
                if disk_metrics:
                    disk_usage = disk_metrics[0].value
                    disk_score = 1.0 - min(disk_usage / 100.0, 1.0)
                    scores.append(disk_score)
                
                # I/O performance
                if io_metrics:
                    io_score = min(io_metrics[0].value / 1000.0, 1.0)  # Assuming higher is better
                    scores.append(io_score)
            
            elif component.component_type == ComponentType.SERVICE:
                response_time_metrics = [m for m in metrics if 'response_time' in m.metric_name.lower()]
                error_rate_metrics = [m for m in metrics if 'error' in m.metric_name.lower()]
                
                # Response time health (lower is better)
                if response_time_metrics:
                    response_time = response_time_metrics[0].value
                    response_score = max(0.0, 1.0 - (response_time / 5000.0))  # 5 second threshold
                    scores.append(response_score)
                
                # Error rate health (lower is better)
                if error_rate_metrics:
                    error_rate = error_rate_metrics[0].value
                    error_score = max(0.0, 1.0 - (error_rate / 10.0))  # 10% threshold
                    scores.append(error_score)
            
            # Calculate overall score
            if scores:
                return statistics.mean(scores)
            else:
                # Fallback scoring based on metric values
                normalized_scores = []
                for metric in metrics:
                    # Simple normalization (assuming lower values are better for most metrics)
                    if 'usage' in metric.metric_name.lower() or 'error' in metric.metric_name.lower():
                        normalized_scores.append(max(0.0, 1.0 - min(metric.value / 100.0, 1.0)))
                    else:
                        normalized_scores.append(min(metric.value / 100.0, 1.0))
                
                return statistics.mean(normalized_scores) if normalized_scores else 0.5
                
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 0.5
    
    async def _determine_health_status(self, health_score: float, metrics: List[SystemMetric]) -> HealthStatus:
        """Determine health status based on score and metrics"""
        
        try:
            # Check for critical conditions first
            for metric in metrics:
                if 'error' in metric.metric_name.lower() and metric.value > 50:
                    return HealthStatus.CRITICAL
                if 'usage' in metric.metric_name.lower() and metric.value > 95:
                    return HealthStatus.CRITICAL
            
            # Status based on health score
            if health_score >= 0.9:
                return HealthStatus.HEALTHY
            elif health_score >= 0.7:
                return HealthStatus.WARNING
            elif health_score >= 0.5:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.CRITICAL
                
        except Exception as e:
            logger.error(f"Error determining health status: {e}")
            return HealthStatus.UNKNOWN
    
    # ===== ANOMALY DETECTION =====
    
    async def _check_for_anomalies(self, component_id: str, current_metrics: List[SystemMetric]):
        """Check for anomalies in component metrics"""
        
        try:
            for metric in current_metrics:
                metric_key = f"{component_id}_{metric.metric_name}"
                
                # Get historical data
                history = list(self.metrics_history[metric_key])
                
                if len(history) < 10:  # Need sufficient history
                    continue
                
                # Extract values
                values = [m.value for m in history[-100:]]  # Last 100 values
                current_value = metric.value
                
                # Statistical anomaly detection
                anomaly_detected = await self._detect_statistical_anomaly(values, current_value)
                
                if anomaly_detected:
                    # Create anomaly record
                    anomaly = Anomaly(
                        anomaly_id=f"anom_{uuid.uuid4().hex[:12]}",
                        component_id=component_id,
                        anomaly_type=AnomalyType.THRESHOLD,
                        description=f"Anomaly detected in {metric.metric_name}",
                        severity=await self._determine_anomaly_severity(current_value, values),
                        detected_at=datetime.utcnow(),
                        metric_values={metric.metric_name: current_value},
                        confidence_score=0.8,  # Simplified
                        predicted_impact="Performance degradation possible",
                        recommended_actions=[f"Investigate {metric.metric_name} spike"]
                    )
                    
                    # Store anomaly
                    self.anomalies[anomaly.anomaly_id] = anomaly
                    self.metrics["anomalies_detected"] += 1
                    
                    # Generate alert
                    await self._generate_alert(anomaly)
                    
                    logger.warning(f"Anomaly detected: {anomaly.anomaly_id}")
            
            # ML-based anomaly detection if available
            if sklearn_available:
                await self._ml_anomaly_detection(component_id, current_metrics)
                
        except Exception as e:
            logger.error(f"Error checking for anomalies: {e}")
    
    async def _detect_statistical_anomaly(self, historical_values: List[float], current_value: float) -> bool:
        """Detect statistical anomalies using threshold-based method"""
        
        try:
            if len(historical_values) < 5:
                return False
            
            # Calculate statistics
            mean_val = statistics.mean(historical_values)
            std_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
            
            if std_val == 0:
                return abs(current_value - mean_val) > 0.1 * abs(mean_val)
            
            # Z-score based detection
            z_score = abs(current_value - mean_val) / std_val
            
            return z_score > 3.0  # 3-sigma rule
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
            return False
    
    async def _ml_anomaly_detection(self, component_id: str, current_metrics: List[SystemMetric]):
        """ML-based anomaly detection using Isolation Forest"""
        
        try:
            # Prepare feature vector
            features = []
            for metric in current_metrics:
                features.append(metric.value)
            
            if not features:
                return
            
            # Get or create anomaly detector for component
            detector_key = f"{component_id}_detector"
            if detector_key not in self.anomaly_detectors:
                # Initialize detector
                self.anomaly_detectors[detector_key] = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                
                # Train on historical data
                await self._train_anomaly_detector(component_id, detector_key)
            
            detector = self.anomaly_detectors[detector_key]
            
            # Predict anomaly
            prediction = detector.predict([features])
            anomaly_score = detector.decision_function([features])[0]
            
            if prediction[0] == -1:  # Anomaly detected
                # Create anomaly record
                anomaly = Anomaly(
                    anomaly_id=f"ml_anom_{uuid.uuid4().hex[:12]}",
                    component_id=component_id,
                    anomaly_type=AnomalyType.PATTERN,
                    description="ML-detected behavioral anomaly",
                    severity=AlertSeverity.WARNING,
                    detected_at=datetime.utcnow(),
                    metric_values={m.metric_name: m.value for m in current_metrics},
                    confidence_score=abs(anomaly_score),
                    predicted_impact="Potential performance or reliability issue",
                    recommended_actions=["Investigate component behavior", "Check recent changes"]
                )
                
                self.anomalies[anomaly.anomaly_id] = anomaly
                await self._generate_alert(anomaly)
                
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {e}")
    
    async def _train_anomaly_detector(self, component_id: str, detector_key: str):
        """Train anomaly detector on historical data"""
        
        try:
            # Collect historical metrics
            training_data = []
            
            # Get recent history for each metric type
            component = self.components[component_id]
            for metric in component.metrics:
                metric_key = f"{component_id}_{metric.metric_name}"
                history = list(self.metrics_history[metric_key])
                
                if len(history) >= 50:  # Need sufficient data
                    recent_history = history[-100:]  # Last 100 values
                    for hist_metric in recent_history:
                        training_data.append([hist_metric.value])
            
            if len(training_data) >= 50:
                detector = self.anomaly_detectors[detector_key]
                detector.fit(training_data)
                logger.info(f"Trained anomaly detector for {component_id}")
                
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
    
    async def _determine_anomaly_severity(self, current_value: float, historical_values: List[float]) -> AlertSeverity:
        """Determine severity of detected anomaly"""
        
        try:
            if not historical_values:
                return AlertSeverity.WARNING
            
            mean_val = statistics.mean(historical_values)
            std_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
            
            if std_val == 0:
                deviation_ratio = abs(current_value - mean_val) / max(abs(mean_val), 1.0)
            else:
                z_score = abs(current_value - mean_val) / std_val
                deviation_ratio = z_score / 3.0  # Normalize by 3-sigma
            
            if deviation_ratio > 2.0:
                return AlertSeverity.CRITICAL
            elif deviation_ratio > 1.5:
                return AlertSeverity.ERROR
            elif deviation_ratio > 1.0:
                return AlertSeverity.WARNING
            else:
                return AlertSeverity.INFO
                
        except Exception as e:
            logger.error(f"Error determining anomaly severity: {e}")
            return AlertSeverity.WARNING
    
    # ===== FAILURE PREDICTION =====
    
    async def _generate_failure_prediction(self, component_id: str):
        """Generate failure prediction for component"""
        
        try:
            if not sklearn_available:
                return
            
            component = self.components[component_id]
            
            # Get prediction model for component type
            model_key = f"{component.component_type.value}_prediction"
            
            if model_key not in self.prediction_models:
                # Train prediction model if enough data
                await self._train_prediction_model(component.component_type)
            
            if model_key not in self.prediction_models:
                return  # Not enough data to train
            
            model = self.prediction_models[model_key]
            scaler = self.scalers.get(model_key)
            
            # Prepare features from recent metrics
            features = await self._prepare_prediction_features(component_id)
            
            if not features:
                return
            
            # Make prediction
            if scaler:
                features_scaled = scaler.transform([features])
            else:
                features_scaled = [features]
            
            # Predict failure probability or time
            prediction = model.predict(features_scaled)[0]
            
            # Convert prediction to failure prediction
            if prediction > 0.7:  # High failure probability
                predicted_failure_time = datetime.utcnow() + timedelta(hours=self.config["prediction_horizon"])
                
                failure_prediction = FailurePrediction(
                    prediction_id=f"pred_{uuid.uuid4().hex[:12]}",
                    component_id=component_id,
                    predicted_failure_time=predicted_failure_time,
                    confidence=prediction,
                    failure_type="Performance degradation",
                    contributing_factors=await self._identify_contributing_factors(component_id),
                    recommended_actions=await self._get_failure_prevention_actions(component_id),
                    created_at=datetime.utcnow()
                )
                
                self.failure_predictions[failure_prediction.prediction_id] = failure_prediction
                self.metrics["predictions_made"] += 1
                
                # Schedule preventive maintenance
                await self._schedule_preventive_maintenance(component_id, failure_prediction)
                
                logger.warning(f"Failure predicted for {component_id}: {failure_prediction.prediction_id}")
                
        except Exception as e:
            logger.error(f"Error generating failure prediction: {e}")
    
    async def _train_prediction_model(self, component_type: ComponentType):
        """Train failure prediction model for component type"""
        
        try:
            if not sklearn_available:
                return
            
            # Collect training data
            training_data = await self._collect_prediction_training_data(component_type)
            
            if len(training_data) < self.config["min_training_samples"]:
                logger.warning(f"Not enough training data for {component_type.value} prediction model")
                return
            
            # Prepare features and targets
            X = []
            y = []
            
            for data_point in training_data:
                X.append(data_point["features"])
                y.append(data_point["target"])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = 1.0 - mean_absolute_error(y_test, y_pred)  # Simplified accuracy
            
            # Store model
            model_key = f"{component_type.value}_prediction"
            self.prediction_models[model_key] = model
            self.scalers[model_key] = scaler
            
            # Store metadata
            self.model_metadata[model_key] = PredictionModel(
                model_id=model_key,
                component_type=component_type,
                target_metric="failure_probability",
                model_type="RandomForest",
                accuracy=accuracy,
                last_trained=datetime.utcnow(),
                training_data_size=len(training_data),
                prediction_horizon=self.config["prediction_horizon"]
            )
            
            logger.info(f"Trained prediction model for {component_type.value} with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error training prediction model: {e}")
    
    async def _collect_prediction_training_data(self, component_type: ComponentType) -> List[Dict[str, Any]]:
        """Collect training data for prediction model"""
        
        try:
            training_data = []
            
            # Use maintenance history as training data
            for maintenance_record in self.maintenance_history:
                if maintenance_record.get("component_type") == component_type.value:
                    # Extract features from metrics before maintenance
                    features = maintenance_record.get("pre_maintenance_features", [])
                    
                    # Target: time to failure (simplified)
                    time_to_maintenance = maintenance_record.get("time_to_maintenance", 24)
                    target = max(0.0, 1.0 - (time_to_maintenance / 168.0))  # Normalize to 0-1 over week
                    
                    if features:
                        training_data.append({
                            "features": features,
                            "target": target
                        })
            
            # Also use anomaly data
            for anomaly in self.anomalies.values():
                if anomaly.component_id in self.components:
                    component = self.components[anomaly.component_id]
                    if component.component_type == component_type:
                        features = list(anomaly.metric_values.values())
                        
                        # Target based on anomaly severity
                        severity_map = {
                            AlertSeverity.INFO: 0.2,
                            AlertSeverity.WARNING: 0.5,
                            AlertSeverity.ERROR: 0.8,
                            AlertSeverity.CRITICAL: 1.0
                        }
                        target = severity_map.get(anomaly.severity, 0.5)
                        
                        if features:
                            training_data.append({
                                "features": features,
                                "target": target
                            })
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error collecting prediction training data: {e}")
            return []
    
    async def _prepare_prediction_features(self, component_id: str) -> List[float]:
        """Prepare features for failure prediction"""
        
        try:
            component = self.components[component_id]
            features = []
            
            # Current metric values
            for metric in component.metrics:
                features.append(metric.value)
            
            # Health score
            features.append(component.health_score)
            
            # Trend features (if available)
            for metric in component.metrics:
                metric_key = f"{component_id}_{metric.metric_name}"
                history = list(self.metrics_history[metric_key])
                
                if len(history) >= 5:
                    recent_values = [m.value for m in history[-5:]]
                    
                    # Trend slope
                    if len(recent_values) > 1:
                        trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                        features.append(trend)
                    
                    # Variance
                    if len(recent_values) > 1:
                        variance = statistics.variance(recent_values)
                        features.append(variance)
            
            # Pad or truncate to fixed size
            target_size = 20
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {e}")
            return []
    
    async def _identify_contributing_factors(self, component_id: str) -> List[str]:
        """Identify factors contributing to potential failure"""
        
        try:
            factors = []
            component = self.components[component_id]
            
            # Check current metrics
            for metric in component.metrics:
                if 'usage' in metric.metric_name.lower() and metric.value > 80:
                    factors.append(f"High {metric.metric_name}: {metric.value}%")
                elif 'error' in metric.metric_name.lower() and metric.value > 5:
                    factors.append(f"Elevated {metric.metric_name}: {metric.value}%")
                elif 'response_time' in metric.metric_name.lower() and metric.value > 1000:
                    factors.append(f"Slow {metric.metric_name}: {metric.value}ms")
            
            # Check recent anomalies
            recent_anomalies = [
                a for a in self.anomalies.values() 
                if a.component_id == component_id and 
                (datetime.utcnow() - a.detected_at).total_seconds() < 3600
            ]
            
            if recent_anomalies:
                factors.append(f"Recent anomalies detected: {len(recent_anomalies)}")
            
            return factors
            
        except Exception as e:
            logger.error(f"Error identifying contributing factors: {e}")
            return []
    
    async def _get_failure_prevention_actions(self, component_id: str) -> List[str]:
        """Get recommended actions to prevent failure"""
        
        try:
            actions = []
            component = self.components[component_id]
            
            # Generic actions based on component type
            if component.component_type == ComponentType.COMPUTE:
                actions.extend([
                    "Monitor CPU and memory usage closely",
                    "Consider scaling resources",
                    "Check for resource leaks"
                ])
            elif component.component_type == ComponentType.STORAGE:
                actions.extend([
                    "Monitor disk space usage",
                    "Check I/O performance",
                    "Consider storage cleanup"
                ])
            elif component.component_type == ComponentType.SERVICE:
                actions.extend([
                    "Monitor response times",
                    "Check error logs",
                    "Verify service dependencies"
                ])
            
            # Specific actions based on metrics
            for metric in component.metrics:
                if 'memory' in metric.metric_name.lower() and metric.value > 85:
                    actions.append("Restart service to free memory")
                elif 'disk' in metric.metric_name.lower() and metric.value > 85:
                    actions.append("Clean up disk space")
            
            return actions
            
        except Exception as e:
            logger.error(f"Error getting failure prevention actions: {e}")
            return []
    
    # ===== MAINTENANCE SCHEDULING =====
    
    async def schedule_maintenance(self, task: MaintenanceTask) -> bool:
        """Schedule a maintenance task"""
        
        try:
            # Validate task
            if task.task_id in self.maintenance_tasks:
                logger.warning(f"Maintenance task {task.task_id} already exists")
                return False
            
            # Check for conflicts
            conflicts = await self._check_maintenance_conflicts(task)
            if conflicts:
                logger.warning(f"Maintenance conflicts detected for {task.task_id}")
                return False
            
            # Store task
            self.maintenance_tasks[task.task_id] = task
            
            # Schedule execution if needed
            if task.status == MaintenanceStatus.SCHEDULED:
                await self._schedule_task_execution(task)
            
            logger.info(f"Scheduled maintenance task: {task.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error scheduling maintenance: {e}")
            return False
    
    async def _auto_schedule_maintenance(self, component_id: str):
        """Automatically schedule maintenance based on component health"""
        
        try:
            component = self.components[component_id]
            
            # Check if maintenance is needed
            if component.status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                # Create maintenance task
                task = MaintenanceTask(
                    task_id=f"auto_{component_id}_{datetime.utcnow().timestamp()}",
                    component_id=component_id,
                    maintenance_type=MaintenanceType.PREDICTIVE,
                    title=f"Automatic maintenance for {component.name}",
                    description=f"Health score: {component.health_score:.2f}, Status: {component.status.value}",
                    priority=1 if component.status == HealthStatus.CRITICAL else 2,
                    estimated_duration=timedelta(hours=1),
                    scheduled_time=datetime.utcnow() + timedelta(hours=2),
                    status=MaintenanceStatus.SCHEDULED
                )
                
                await self.schedule_maintenance(task)
                
        except Exception as e:
            logger.error(f"Error auto-scheduling maintenance: {e}")
    
    async def _schedule_preventive_maintenance(self, component_id: str, prediction: FailurePrediction):
        """Schedule preventive maintenance based on failure prediction"""
        
        try:
            # Schedule maintenance before predicted failure
            maintenance_time = prediction.predicted_failure_time - timedelta(hours=4)
            
            task = MaintenanceTask(
                task_id=f"prev_{component_id}_{prediction.prediction_id}",
                component_id=component_id,
                maintenance_type=MaintenanceType.PREVENTIVE,
                title=f"Preventive maintenance (Prediction: {prediction.prediction_id})",
                description=f"Prevent predicted failure: {prediction.failure_type}",
                priority=1,
                estimated_duration=timedelta(hours=2),
                scheduled_time=maintenance_time,
                status=MaintenanceStatus.SCHEDULED,
                metadata={"prediction_id": prediction.prediction_id}
            )
            
            await self.schedule_maintenance(task)
            
        except Exception as e:
            logger.error(f"Error scheduling preventive maintenance: {e}")
    
    async def _check_maintenance_conflicts(self, task: MaintenanceTask) -> List[str]:
        """Check for maintenance scheduling conflicts"""
        
        try:
            conflicts = []
            
            # Check for overlapping maintenance on same component
            for existing_task in self.maintenance_tasks.values():
                if (existing_task.component_id == task.component_id and 
                    existing_task.status == MaintenanceStatus.SCHEDULED):
                    
                    # Check time overlap
                    existing_end = existing_task.scheduled_time + existing_task.estimated_duration
                    new_end = task.scheduled_time + task.estimated_duration
                    
                    if (task.scheduled_time < existing_end and 
                        new_end > existing_task.scheduled_time):
                        conflicts.append(f"Overlaps with task {existing_task.task_id}")
            
            # Check resource availability
            if task.resources_required:
                # Simplified resource check
                concurrent_tasks = len([
                    t for t in self.maintenance_tasks.values()
                    if t.status == MaintenanceStatus.SCHEDULED and
                    abs((t.scheduled_time - task.scheduled_time).total_seconds()) < 3600
                ])
                
                if concurrent_tasks >= self.config["max_concurrent_maintenance"]:
                    conflicts.append("Too many concurrent maintenance tasks")
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error checking maintenance conflicts: {e}")
            return []
    
    # ===== ALERT MANAGEMENT =====
    
    async def _generate_alert(self, anomaly: Anomaly):
        """Generate alert for anomaly"""
        
        try:
            alert_key = f"{anomaly.component_id}_{anomaly.anomaly_type.value}"
            
            # Check alert cooldown
            if alert_key in self.active_alerts:
                last_alert = self.active_alerts[alert_key]["timestamp"]
                if (datetime.utcnow() - last_alert).total_seconds() < self.config["alert_cooldown"]:
                    return  # Skip duplicate alert
            
            # Create alert
            alert = {
                "alert_id": f"alert_{uuid.uuid4().hex[:12]}",
                "anomaly_id": anomaly.anomaly_id,
                "component_id": anomaly.component_id,
                "severity": anomaly.severity.value,
                "title": f"Anomaly detected: {anomaly.description}",
                "description": f"Component: {anomaly.component_id}, Type: {anomaly.anomaly_type.value}",
                "timestamp": datetime.utcnow(),
                "acknowledged": False,
                "resolved": False
            }
            
            # Store alert
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            # Publish alert event
            await self._publish_alert_event(alert)
            
            logger.warning(f"Alert generated: {alert['alert_id']}")
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
    
    async def _publish_alert_event(self, alert: Dict[str, Any]):
        """Publish alert event to message queue"""
        
        try:
            await shared_pulsar_client.publish("q.maintenance.alerts", alert)
            
        except Exception as e:
            logger.warning(f"Error publishing alert event: {e}")
    
    # ===== SYSTEM MONITORING =====
    
    async def _register_system_components(self):
        """Register standard system components for monitoring"""
        
        try:
            # CPU component
            cpu_component = ComponentHealth(
                component_id="system_cpu",
                component_type=ComponentType.COMPUTE,
                name="System CPU",
                status=HealthStatus.HEALTHY,
                health_score=1.0,
                last_check=datetime.utcnow(),
                metrics=[]
            )
            await self.register_component(cpu_component)
            
            # Memory component
            memory_component = ComponentHealth(
                component_id="system_memory",
                component_type=ComponentType.MEMORY,
                name="System Memory",
                status=HealthStatus.HEALTHY,
                health_score=1.0,
                last_check=datetime.utcnow(),
                metrics=[]
            )
            await self.register_component(memory_component)
            
            # Disk component
            disk_component = ComponentHealth(
                component_id="system_disk",
                component_type=ComponentType.DISK,
                name="System Disk",
                status=HealthStatus.HEALTHY,
                health_score=1.0,
                last_check=datetime.utcnow(),
                metrics=[]
            )
            await self.register_component(disk_component)
            
            logger.info("System components registered")
            
        except Exception as e:
            logger.error(f"Error registering system components: {e}")
    
    async def _start_monitoring(self):
        """Start background monitoring tasks"""
        
        try:
            # System monitoring task
            task = asyncio.create_task(self._system_monitoring_loop())
            self.background_tasks.add(task)
            
            # Health check task
            task = asyncio.create_task(self._health_check_loop())
            self.background_tasks.add(task)
            
            # Maintenance execution task
            task = asyncio.create_task(self._maintenance_execution_loop())
            self.background_tasks.add(task)
            
            logger.info("Background monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
    
    async def _system_monitoring_loop(self):
        """Background system monitoring loop"""
        
        while True:
            try:
                await asyncio.sleep(self.config["monitoring_interval"])
                
                # Collect system metrics
                if psutil_available:
                    await self._collect_system_metrics()
                
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system metrics using psutil"""
        
        try:
            current_time = datetime.utcnow()
            
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_metric = SystemMetric(
                metric_id=f"cpu_{current_time.timestamp()}",
                component_id="system_cpu",
                component_type=ComponentType.COMPUTE,
                metric_name="cpu_usage",
                value=cpu_usage,
                unit="percent",
                timestamp=current_time
            )
            
            await self.update_component_health("system_cpu", [cpu_metric])
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_metric = SystemMetric(
                metric_id=f"memory_{current_time.timestamp()}",
                component_id="system_memory",
                component_type=ComponentType.MEMORY,
                metric_name="memory_usage",
                value=memory.percent,
                unit="percent",
                timestamp=current_time
            )
            
            await self.update_component_health("system_memory", [memory_metric])
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_metric = SystemMetric(
                metric_id=f"disk_{current_time.timestamp()}",
                component_id="system_disk",
                component_type=ComponentType.DISK,
                metric_name="disk_usage",
                value=(disk.used / disk.total) * 100,
                unit="percent",
                timestamp=current_time
            )
            
            await self.update_component_health("system_disk", [disk_metric])
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        
        while True:
            try:
                await asyncio.sleep(self.config["health_check_interval"])
                
                # Update global health metrics
                await self._update_global_metrics()
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _maintenance_execution_loop(self):
        """Background maintenance execution loop"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for scheduled maintenance tasks
                now = datetime.utcnow()
                
                for task in self.maintenance_tasks.values():
                    if (task.status == MaintenanceStatus.SCHEDULED and 
                        task.scheduled_time <= now):
                        
                        # Execute maintenance task
                        await self._execute_maintenance_task(task)
                
            except Exception as e:
                logger.error(f"Error in maintenance execution loop: {e}")
    
    async def _execute_maintenance_task(self, task: MaintenanceTask):
        """Execute a maintenance task"""
        
        try:
            # Update task status
            task.status = MaintenanceStatus.IN_PROGRESS
            
            # Log maintenance start
            logger.info(f"Starting maintenance task: {task.task_id}")
            
            # Perform maintenance actions based on type
            success = await self._perform_maintenance_actions(task)
            
            # Update task status
            if success:
                task.status = MaintenanceStatus.COMPLETED
                self.metrics["maintenance_tasks_completed"] += 1
                
                # Record maintenance history
                maintenance_record = {
                    "task_id": task.task_id,
                    "component_id": task.component_id,
                    "component_type": self.components[task.component_id].component_type.value,
                    "maintenance_type": task.maintenance_type.value,
                    "completed_at": datetime.utcnow().isoformat(),
                    "duration": (datetime.utcnow() - task.scheduled_time).total_seconds(),
                    "success": True
                }
                self.maintenance_history.append(maintenance_record)
                
                logger.info(f"Completed maintenance task: {task.task_id}")
            else:
                task.status = MaintenanceStatus.FAILED
                logger.error(f"Failed maintenance task: {task.task_id}")
            
        except Exception as e:
            logger.error(f"Error executing maintenance task: {e}")
            task.status = MaintenanceStatus.FAILED
    
    async def _perform_maintenance_actions(self, task: MaintenanceTask) -> bool:
        """Perform actual maintenance actions"""
        
        try:
            # Simulate maintenance actions
            # In a real implementation, this would perform actual maintenance
            
            component = self.components.get(task.component_id)
            if not component:
                return False
            
            # Simulate maintenance delay
            await asyncio.sleep(5)  # 5 seconds for demo
            
            # Reset component health score
            component.health_score = min(1.0, component.health_score + 0.1)
            component.status = HealthStatus.HEALTHY
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing maintenance actions: {e}")
            return False
    
    # ===== UTILITY METHODS =====
    
    async def _update_global_metrics(self):
        """Update global health metrics"""
        
        try:
            total_components = len(self.components)
            healthy_components = sum(
                1 for c in self.components.values() 
                if c.status == HealthStatus.HEALTHY
            )
            
            self.metrics["total_components"] = total_components
            self.metrics["healthy_components"] = healthy_components
            
            if total_components > 0:
                self.metrics["uptime_percentage"] = (healthy_components / total_components) * 100
            
        except Exception as e:
            logger.error(f"Error updating global metrics: {e}")
    
    async def _initialize_component_monitoring(self, component_id: str):
        """Initialize monitoring for a component"""
        
        try:
            # Set up metric collection for component
            # This would depend on the specific component type
            pass
            
        except Exception as e:
            logger.error(f"Error initializing component monitoring: {e}")
    
    async def _schedule_task_execution(self, task: MaintenanceTask):
        """Schedule task execution"""
        
        try:
            # In a real implementation, this would integrate with a job scheduler
            # For now, we rely on the background loop to check scheduled tasks
            pass
            
        except Exception as e:
            logger.error(f"Error scheduling task execution: {e}")
    
    async def _initialize_ml_models(self):
        """Initialize ML models for prediction and anomaly detection"""
        
        try:
            if not sklearn_available:
                return
            
            # Initialize models for different component types
            for component_type in ComponentType:
                if await self._has_training_data_for_type(component_type):
                    await self._train_prediction_model(component_type)
            
            logger.info("ML models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    async def _has_training_data_for_type(self, component_type: ComponentType) -> bool:
        """Check if we have training data for component type"""
        
        try:
            # Check maintenance history
            relevant_records = [
                r for r in self.maintenance_history 
                if r.get("component_type") == component_type.value
            ]
            
            return len(relevant_records) >= self.config["min_training_samples"]
            
        except Exception as e:
            logger.error(f"Error checking training data: {e}")
            return False
    
    # ===== PERSISTENCE =====
    
    async def _load_maintenance_data(self):
        """Load maintenance data from storage"""
        
        try:
            # Load components
            components_file = self.storage_path / "components.json"
            if components_file.exists():
                with open(components_file, 'r') as f:
                    components_data = json.load(f)
                    for comp_data in components_data:
                        component = ComponentHealth(**comp_data)
                        self.components[component.component_id] = component
            
            # Load maintenance tasks
            tasks_file = self.storage_path / "maintenance_tasks.json"
            if tasks_file.exists():
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                    for task_data in tasks_data:
                        task = MaintenanceTask(**task_data)
                        self.maintenance_tasks[task.task_id] = task
            
            # Load maintenance history
            history_file = self.storage_path / "maintenance_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.maintenance_history = json.load(f)
            
            logger.info("Maintenance data loaded from storage")
            
        except Exception as e:
            logger.warning(f"Error loading maintenance data: {e}")
    
    async def _save_maintenance_data(self):
        """Save maintenance data to storage"""
        
        try:
            # Save components
            components_data = []
            for component in self.components.values():
                components_data.append(asdict(component))
            
            components_file = self.storage_path / "components.json"
            with open(components_file, 'w') as f:
                json.dump(components_data, f, indent=2, default=str)
            
            # Save maintenance tasks
            tasks_data = []
            for task in self.maintenance_tasks.values():
                tasks_data.append(asdict(task))
            
            tasks_file = self.storage_path / "maintenance_tasks.json"
            with open(tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2, default=str)
            
            # Save maintenance history
            history_file = self.storage_path / "maintenance_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.maintenance_history, f, indent=2, default=str)
            
            logger.info("Maintenance data saved to storage")
            
        except Exception as e:
            logger.warning(f"Error saving maintenance data: {e}")
    
    # ===== API METHODS =====
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        
        healthy_count = sum(1 for c in self.components.values() if c.status == HealthStatus.HEALTHY)
        total_count = len(self.components)
        
        return {
            "overall_health": "healthy" if healthy_count / max(total_count, 1) > 0.8 else "degraded",
            "healthy_components": healthy_count,
            "total_components": total_count,
            "health_percentage": (healthy_count / max(total_count, 1)) * 100,
            "active_anomalies": len([a for a in self.anomalies.values() if (datetime.utcnow() - a.detected_at).total_seconds() < 3600]),
            "scheduled_maintenance": len([t for t in self.maintenance_tasks.values() if t.status == MaintenanceStatus.SCHEDULED]),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def get_component_details(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a component"""
        
        if component_id not in self.components:
            return None
        
        component = self.components[component_id]
        
        # Get recent anomalies
        recent_anomalies = [
            asdict(a) for a in self.anomalies.values()
            if a.component_id == component_id and
            (datetime.utcnow() - a.detected_at).total_seconds() < 86400
        ]
        
        # Get maintenance history
        maintenance_history = [
            record for record in self.maintenance_history
            if record.get("component_id") == component_id
        ]
        
        return {
            "component": asdict(component),
            "recent_anomalies": recent_anomalies,
            "maintenance_history": maintenance_history[-10:],  # Last 10 records
            "failure_predictions": [
                asdict(p) for p in self.failure_predictions.values()
                if p.component_id == component_id
            ]
        }
    
    async def get_maintenance_schedule(self) -> List[Dict[str, Any]]:
        """Get maintenance schedule"""
        
        scheduled_tasks = [
            asdict(task) for task in self.maintenance_tasks.values()
            if task.status == MaintenanceStatus.SCHEDULED
        ]
        
        # Sort by scheduled time
        scheduled_tasks.sort(key=lambda x: x["scheduled_time"])
        
        return scheduled_tasks
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        return {
            "metrics": self.metrics,
            "component_count_by_type": {
                comp_type.value: len([c for c in self.components.values() if c.component_type == comp_type])
                for comp_type in ComponentType
            },
            "health_distribution": {
                status.value: len([c for c in self.components.values() if c.status == status])
                for status in HealthStatus
            },
            "recent_anomalies": len([a for a in self.anomalies.values() if (datetime.utcnow() - a.detected_at).total_seconds() < 3600]),
            "predictions_accuracy": self.metrics["prediction_accuracy"],
            "timestamp": datetime.utcnow().isoformat()
        }

# Global service instance
predictive_maintenance_service = PredictiveMaintenanceService() 