"""
Automated Incident Detection Service

This service provides intelligent incident detection and monitoring:
- Pattern-based anomaly detection
- Multi-dimensional system health monitoring
- Proactive issue identification
- Performance degradation detection
- Failure prediction and early warning
- Automated alerting and escalation
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import asdict, dataclass
from enum import Enum
import uuid
import statistics
import numpy as np
from collections import defaultdict, deque

# Q Platform imports
from shared.q_analytics_schemas.models import PerformanceMetrics
from shared.q_workflow_schemas.models import WorkflowStatus
from app.services.pulsar_service import PulsarService
from app.services.ignite_service import IgniteService
from app.services.knowledge_graph_service import KnowledgeGraphService

logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    """Incident severity levels"""
    CRITICAL = "critical"       # System down or major functionality impaired
    HIGH = "high"               # Significant impact on operations
    MEDIUM = "medium"           # Moderate impact, some functionality affected
    LOW = "low"                 # Minor impact, no significant disruption
    INFO = "info"               # Informational, no impact

class IncidentType(Enum):
    """Types of incidents"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_FAILURE = "system_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONNECTIVITY_ISSUE = "connectivity_issue"
    DATA_ANOMALY = "data_anomaly"
    WORKFLOW_FAILURE = "workflow_failure"
    AGENT_MALFUNCTION = "agent_malfunction"
    SECURITY_BREACH = "security_breach"

class DetectionMethod(Enum):
    """Incident detection methods"""
    THRESHOLD_BASED = "threshold_based"
    PATTERN_RECOGNITION = "pattern_recognition"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    PREDICTIVE_MODEL = "predictive_model"

class IncidentStatus(Enum):
    """Incident status"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    CLOSED = "closed"

@dataclass
class HealthMetric:
    """Health metric for monitoring"""
    metric_id: str
    metric_name: str
    component: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]
    thresholds: Dict[str, float]

@dataclass
class Incident:
    """Incident record"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    incident_type: IncidentType
    detection_method: DetectionMethod
    affected_components: List[str]
    root_cause: Optional[str]
    
    # Status tracking
    status: IncidentStatus
    detected_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    
    # Context
    metrics: List[HealthMetric]
    context: Dict[str, Any]
    related_incidents: List[str]
    
    # Response
    responders: List[str]
    remediation_actions: List[str]
    resolution_notes: str

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    anomaly_id: str
    metric_name: str
    component: str
    anomaly_score: float
    confidence: float
    baseline_value: float
    current_value: float
    deviation_percentage: float
    detected_at: datetime
    context: Dict[str, Any]

@dataclass
class ThresholdRule:
    """Threshold-based detection rule"""
    rule_id: str
    metric_name: str
    component: str
    threshold_type: str  # greater_than, less_than, range
    threshold_value: float
    severity: IncidentSeverity
    enabled: bool
    created_at: datetime

class AutomatedIncidentDetection:
    """
    Service for automated incident detection and monitoring
    """
    
    def __init__(self):
        self.pulsar_service = PulsarService()
        self.ignite_service = IgniteService()
        self.knowledge_graph = KnowledgeGraphService()
        
        # Incident management
        self.active_incidents: Dict[str, Incident] = {}
        self.incident_history: List[Incident] = []
        self.threshold_rules: Dict[str, ThresholdRule] = {}
        
        # Monitoring state
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        
        # Detection configuration
        self.detection_config = {
            "anomaly_threshold": 0.8,
            "baseline_window_hours": 24,
            "metric_collection_interval": 30,  # seconds
            "incident_correlation_window": 300,  # 5 minutes
            "auto_resolve_threshold": 0.95,
            "escalation_time_minutes": 15
        }
        
        # Performance tracking
        self.detection_metrics = {
            "incidents_detected": 0,
            "false_positives": 0,
            "detection_accuracy": 0.0,
            "average_detection_time": 0.0,
            "incidents_auto_resolved": 0
        }
        
        # System health state
        self.system_health = {
            "overall_status": "healthy",
            "component_health": {},
            "active_incident_count": 0,
            "last_health_check": datetime.utcnow()
        }
    
    async def initialize(self):
        """Initialize the incident detection service"""
        logger.info("Initializing Automated Incident Detection Service")
        
        # Load threshold rules
        await self._load_threshold_rules()
        
        # Initialize anomaly detectors
        await self._initialize_anomaly_detectors()
        
        # Load baseline metrics
        await self._load_baseline_metrics()
        
        # Start monitoring tasks
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._anomaly_detection_loop())
        asyncio.create_task(self._incident_correlation_loop())
        asyncio.create_task(self._auto_resolution_loop())
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        logger.info("Automated Incident Detection Service initialized successfully")
    
    # ===== HEALTH MONITORING =====
    
    async def report_health_metric(
        self,
        metric_name: str,
        component: str,
        value: float,
        unit: str = "",
        tags: Dict[str, str] = None,
        thresholds: Dict[str, float] = None
    ):
        """
        Report a health metric for monitoring
        
        Args:
            metric_name: Name of the metric
            component: Component being monitored
            value: Metric value
            unit: Unit of measurement
            tags: Additional tags
            thresholds: Threshold values
        """
        metric = HealthMetric(
            metric_id=f"metric_{uuid.uuid4().hex[:12]}",
            metric_name=metric_name,
            component=component,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            thresholds=thresholds or {}
        )
        
        # Store metric
        metric_key = f"{component}:{metric_name}"
        self.metric_history[metric_key].append(metric)
        
        # Update system health
        self.system_health["component_health"][component] = {
            "last_metric_time": metric.timestamp,
            "last_metric_value": value,
            "status": "healthy"  # Will be updated by detection logic
        }
        
        # Check thresholds
        await self._check_threshold_violations(metric)
        
        # Detect anomalies
        await self._detect_metric_anomalies(metric)
    
    async def _check_threshold_violations(self, metric: HealthMetric):
        """Check if metric violates threshold rules"""
        metric_key = f"{metric.component}:{metric.metric_name}"
        
        # Check against configured rules
        for rule in self.threshold_rules.values():
            if (rule.metric_name == metric.metric_name and 
                rule.component == metric.component and 
                rule.enabled):
                
                violation = False
                
                if rule.threshold_type == "greater_than" and metric.value > rule.threshold_value:
                    violation = True
                elif rule.threshold_type == "less_than" and metric.value < rule.threshold_value:
                    violation = True
                elif rule.threshold_type == "range":
                    # Assuming threshold_value is the center, need range definition
                    pass
                
                if violation:
                    await self._create_threshold_incident(metric, rule)
    
    async def _detect_metric_anomalies(self, metric: HealthMetric):
        """Detect anomalies in metric values"""
        metric_key = f"{metric.component}:{metric.metric_name}"
        
        # Get historical data
        history = list(self.metric_history[metric_key])
        
        if len(history) < 10:  # Need minimum history
            return
        
        # Calculate baseline statistics
        recent_values = [m.value for m in history[-100:]]  # Last 100 values
        baseline_mean = statistics.mean(recent_values)
        baseline_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        
        # Detect anomaly using statistical method
        if baseline_std > 0:
            z_score = abs(metric.value - baseline_mean) / baseline_std
            
            if z_score > 3:  # 3 standard deviations
                anomaly = AnomalyDetection(
                    anomaly_id=f"anomaly_{uuid.uuid4().hex[:12]}",
                    metric_name=metric.metric_name,
                    component=metric.component,
                    anomaly_score=z_score,
                    confidence=min(1.0, z_score / 5),  # Normalize confidence
                    baseline_value=baseline_mean,
                    current_value=metric.value,
                    deviation_percentage=((metric.value - baseline_mean) / baseline_mean) * 100,
                    detected_at=datetime.utcnow(),
                    context={"z_score": z_score, "baseline_std": baseline_std}
                )
                
                await self._create_anomaly_incident(metric, anomaly)
    
    # ===== INCIDENT MANAGEMENT =====
    
    async def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        incident_type: IncidentType,
        affected_components: List[str],
        context: Dict[str, Any] = None
    ) -> str:
        """
        Create a new incident
        
        Args:
            title: Incident title
            description: Detailed description
            severity: Incident severity
            incident_type: Type of incident
            affected_components: List of affected components
            context: Additional context
            
        Returns:
            Incident ID
        """
        incident = Incident(
            incident_id=f"incident_{uuid.uuid4().hex[:12]}",
            title=title,
            description=description,
            severity=severity,
            incident_type=incident_type,
            detection_method=DetectionMethod.THRESHOLD_BASED,
            affected_components=affected_components,
            root_cause=None,
            status=IncidentStatus.DETECTED,
            detected_at=datetime.utcnow(),
            acknowledged_at=None,
            resolved_at=None,
            metrics=[],
            context=context or {},
            related_incidents=[],
            responders=[],
            remediation_actions=[],
            resolution_notes=""
        )
        
        # Store incident
        self.active_incidents[incident.incident_id] = incident
        
        # Update system health
        self.system_health["active_incident_count"] = len(self.active_incidents)
        await self._update_overall_health_status()
        
        # Publish incident
        await self.pulsar_service.publish(
            "q.incidents.detected",
            {
                "incident_id": incident.incident_id,
                "title": title,
                "severity": severity.value,
                "incident_type": incident_type.value,
                "affected_components": affected_components,
                "detected_at": incident.detected_at.isoformat(),
                "context": context or {}
            }
        )
        
        # Update metrics
        self.detection_metrics["incidents_detected"] += 1
        
        logger.warning(f"Incident detected: {incident.incident_id} - {title}")
        return incident.incident_id
    
    async def resolve_incident(
        self,
        incident_id: str,
        resolution_notes: str = "",
        auto_resolved: bool = False
    ) -> bool:
        """
        Resolve an incident
        
        Args:
            incident_id: ID of incident to resolve
            resolution_notes: Notes about resolution
            auto_resolved: Whether incident was auto-resolved
            
        Returns:
            True if successfully resolved
        """
        if incident_id not in self.active_incidents:
            return False
        
        incident = self.active_incidents[incident_id]
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.utcnow()
        incident.resolution_notes = resolution_notes
        
        # Move to history
        self.incident_history.append(incident)
        del self.active_incidents[incident_id]
        
        # Update system health
        self.system_health["active_incident_count"] = len(self.active_incidents)
        await self._update_overall_health_status()
        
        # Publish resolution
        await self.pulsar_service.publish(
            "q.incidents.resolved",
            {
                "incident_id": incident_id,
                "resolution_time": incident.resolved_at.isoformat(),
                "duration_minutes": (incident.resolved_at - incident.detected_at).total_seconds() / 60,
                "auto_resolved": auto_resolved,
                "resolution_notes": resolution_notes
            }
        )
        
        # Update metrics
        if auto_resolved:
            self.detection_metrics["incidents_auto_resolved"] += 1
        
        logger.info(f"Incident resolved: {incident_id}")
        return True
    
    async def _create_threshold_incident(self, metric: HealthMetric, rule: ThresholdRule):
        """Create incident from threshold violation"""
        title = f"Threshold violation: {metric.metric_name} on {metric.component}"
        description = f"Metric {metric.metric_name} value {metric.value} {rule.threshold_type} threshold {rule.threshold_value}"
        
        await self.create_incident(
            title=title,
            description=description,
            severity=rule.severity,
            incident_type=IncidentType.PERFORMANCE_DEGRADATION,
            affected_components=[metric.component],
            context={
                "metric_name": metric.metric_name,
                "metric_value": metric.value,
                "threshold_value": rule.threshold_value,
                "threshold_type": rule.threshold_type,
                "rule_id": rule.rule_id
            }
        )
    
    async def _create_anomaly_incident(self, metric: HealthMetric, anomaly: AnomalyDetection):
        """Create incident from anomaly detection"""
        title = f"Anomaly detected: {metric.metric_name} on {metric.component}"
        description = f"Anomalous value {metric.value} detected (baseline: {anomaly.baseline_value:.2f}, deviation: {anomaly.deviation_percentage:.1f}%)"
        
        # Determine severity based on anomaly score
        if anomaly.anomaly_score > 5:
            severity = IncidentSeverity.HIGH
        elif anomaly.anomaly_score > 4:
            severity = IncidentSeverity.MEDIUM
        else:
            severity = IncidentSeverity.LOW
        
        await self.create_incident(
            title=title,
            description=description,
            severity=severity,
            incident_type=IncidentType.DATA_ANOMALY,
            affected_components=[metric.component],
            context={
                "anomaly_id": anomaly.anomaly_id,
                "anomaly_score": anomaly.anomaly_score,
                "confidence": anomaly.confidence,
                "baseline_value": anomaly.baseline_value,
                "current_value": anomaly.current_value,
                "deviation_percentage": anomaly.deviation_percentage
            }
        )
    
    # ===== PATTERN RECOGNITION =====
    
    async def analyze_incident_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in incident history"""
        if len(self.incident_history) < 5:
            return {"patterns": [], "insights": []}
        
        patterns = []
        insights = []
        
        # Analyze incident frequency by component
        component_incidents = defaultdict(int)
        for incident in self.incident_history:
            for component in incident.affected_components:
                component_incidents[component] += 1
        
        # Find components with high incident rates
        for component, count in component_incidents.items():
            if count > len(self.incident_history) * 0.3:  # More than 30% of incidents
                patterns.append({
                    "type": "high_incident_component",
                    "component": component,
                    "incident_count": count,
                    "percentage": (count / len(self.incident_history)) * 100
                })
        
        # Analyze incident types
        type_distribution = defaultdict(int)
        for incident in self.incident_history:
            type_distribution[incident.incident_type.value] += 1
        
        most_common_type = max(type_distribution.items(), key=lambda x: x[1])
        if most_common_type[1] > 2:
            insights.append({
                "type": "common_incident_type",
                "incident_type": most_common_type[0],
                "count": most_common_type[1],
                "recommendation": f"Focus on preventing {most_common_type[0]} incidents"
            })
        
        # Analyze resolution times
        resolution_times = []
        for incident in self.incident_history:
            if incident.resolved_at:
                resolution_time = (incident.resolved_at - incident.detected_at).total_seconds() / 60
                resolution_times.append(resolution_time)
        
        if resolution_times:
            avg_resolution_time = statistics.mean(resolution_times)
            insights.append({
                "type": "resolution_time_analysis",
                "average_resolution_minutes": avg_resolution_time,
                "median_resolution_minutes": statistics.median(resolution_times),
                "recommendation": "Focus on faster resolution" if avg_resolution_time > 60 else "Good resolution times"
            })
        
        return {
            "patterns": patterns,
            "insights": insights,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    # ===== PREDICTIVE ANALYSIS =====
    
    async def predict_potential_incidents(self) -> List[Dict[str, Any]]:
        """Predict potential incidents based on trends"""
        predictions = []
        
        # Analyze metric trends
        for metric_key, history in self.metric_history.items():
            if len(history) < 20:  # Need sufficient history
                continue
            
            component, metric_name = metric_key.split(":", 1)
            recent_values = [m.value for m in list(history)[-20:]]
            
            # Simple trend analysis
            if len(recent_values) >= 3:
                # Calculate slope of recent trend
                x = list(range(len(recent_values)))
                slope = np.polyfit(x, recent_values, 1)[0]
                
                # Predict if trend continues
                current_value = recent_values[-1]
                predicted_value = current_value + slope * 10  # Predict 10 time units ahead
                
                # Check if prediction would violate thresholds
                for rule in self.threshold_rules.values():
                    if (rule.metric_name == metric_name and 
                        rule.component == component and 
                        rule.enabled):
                        
                        violation_risk = False
                        
                        if rule.threshold_type == "greater_than" and predicted_value > rule.threshold_value:
                            violation_risk = True
                        elif rule.threshold_type == "less_than" and predicted_value < rule.threshold_value:
                            violation_risk = True
                        
                        if violation_risk:
                            predictions.append({
                                "component": component,
                                "metric_name": metric_name,
                                "current_value": current_value,
                                "predicted_value": predicted_value,
                                "threshold_value": rule.threshold_value,
                                "threshold_type": rule.threshold_type,
                                "trend_slope": slope,
                                "risk_level": "high" if abs(slope) > 1 else "medium",
                                "estimated_time_to_violation": max(1, abs((rule.threshold_value - current_value) / slope)) if slope != 0 else float('inf')
                            })
        
        return predictions
    
    # ===== BACKGROUND TASKS =====
    
    async def _health_monitoring_loop(self):
        """Background task for continuous health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.detection_config["metric_collection_interval"])
                
                # Update system health
                self.system_health["last_health_check"] = datetime.utcnow()
                
                # Check for stale components
                await self._check_stale_components()
                
                # Update overall health status
                await self._update_overall_health_status()
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
    
    async def _anomaly_detection_loop(self):
        """Background task for anomaly detection"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Run advanced anomaly detection
                await self._run_advanced_anomaly_detection()
                
            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
    
    async def _incident_correlation_loop(self):
        """Background task for incident correlation"""
        while True:
            try:
                await asyncio.sleep(120)  # Run every 2 minutes
                
                # Correlate related incidents
                await self._correlate_incidents()
                
            except Exception as e:
                logger.error(f"Error in incident correlation loop: {e}")
    
    async def _auto_resolution_loop(self):
        """Background task for auto-resolution"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Check for auto-resolvable incidents
                await self._check_auto_resolution()
                
            except Exception as e:
                logger.error(f"Error in auto-resolution loop: {e}")
    
    # ===== HELPER METHODS =====
    
    async def _update_overall_health_status(self):
        """Update overall system health status"""
        if self.system_health["active_incident_count"] == 0:
            self.system_health["overall_status"] = "healthy"
        else:
            # Check severity of active incidents
            critical_incidents = sum(1 for incident in self.active_incidents.values() 
                                   if incident.severity == IncidentSeverity.CRITICAL)
            high_incidents = sum(1 for incident in self.active_incidents.values() 
                               if incident.severity == IncidentSeverity.HIGH)
            
            if critical_incidents > 0:
                self.system_health["overall_status"] = "critical"
            elif high_incidents > 0:
                self.system_health["overall_status"] = "degraded"
            else:
                self.system_health["overall_status"] = "warning"
    
    async def _check_stale_components(self):
        """Check for components that haven't reported metrics recently"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=10)
        
        for component, health_info in self.system_health["component_health"].items():
            last_metric_time = health_info.get("last_metric_time")
            if last_metric_time and last_metric_time < cutoff_time:
                # Component is stale
                await self.create_incident(
                    title=f"Component {component} not reporting metrics",
                    description=f"Component {component} has not reported metrics for over 10 minutes",
                    severity=IncidentSeverity.MEDIUM,
                    incident_type=IncidentType.CONNECTIVITY_ISSUE,
                    affected_components=[component],
                    context={"last_metric_time": last_metric_time.isoformat()}
                )
    
    async def _run_advanced_anomaly_detection(self):
        """Run advanced anomaly detection algorithms"""
        # Placeholder for more sophisticated anomaly detection
        pass
    
    async def _correlate_incidents(self):
        """Correlate related incidents"""
        # Group incidents by time and affected components
        recent_incidents = [
            incident for incident in self.active_incidents.values()
            if (datetime.utcnow() - incident.detected_at).total_seconds() < 
               self.detection_config["incident_correlation_window"]
        ]
        
        # Find incidents affecting same components
        component_incidents = defaultdict(list)
        for incident in recent_incidents:
            for component in incident.affected_components:
                component_incidents[component].append(incident)
        
        # Link related incidents
        for component, incidents in component_incidents.items():
            if len(incidents) > 1:
                for i, incident in enumerate(incidents):
                    related_ids = [inc.incident_id for inc in incidents if inc.incident_id != incident.incident_id]
                    incident.related_incidents.extend(related_ids)
    
    async def _check_auto_resolution(self):
        """Check for incidents that can be auto-resolved"""
        current_time = datetime.utcnow()
        
        for incident in list(self.active_incidents.values()):
            # Check if incident conditions have improved
            if await self._incident_conditions_improved(incident):
                await self.resolve_incident(
                    incident.incident_id,
                    "Auto-resolved: conditions have returned to normal",
                    auto_resolved=True
                )
    
    async def _incident_conditions_improved(self, incident: Incident) -> bool:
        """Check if incident conditions have improved"""
        # Simple check: see if related metrics are back to normal
        improvement_threshold = self.detection_config["auto_resolve_threshold"]
        
        # Check threshold-based incidents
        if "threshold_value" in incident.context:
            metric_name = incident.context.get("metric_name")
            component = incident.affected_components[0] if incident.affected_components else ""
            
            if metric_name and component:
                metric_key = f"{component}:{metric_name}"
                if metric_key in self.metric_history:
                    recent_metrics = list(self.metric_history[metric_key])[-5:]  # Last 5 metrics
                    if recent_metrics:
                        avg_recent = statistics.mean([m.value for m in recent_metrics])
                        threshold_value = incident.context["threshold_value"]
                        
                        # Check if average is within acceptable range
                        if abs(avg_recent - threshold_value) / threshold_value < (1 - improvement_threshold):
                            return True
        
        return False
    
    # ===== PLACEHOLDER METHODS =====
    
    async def _load_threshold_rules(self):
        """Load threshold rules from configuration"""
        # Create some default rules
        default_rules = [
            ThresholdRule(
                rule_id="cpu_high",
                metric_name="cpu_utilization",
                component="system",
                threshold_type="greater_than",
                threshold_value=80.0,
                severity=IncidentSeverity.HIGH,
                enabled=True,
                created_at=datetime.utcnow()
            ),
            ThresholdRule(
                rule_id="memory_high",
                metric_name="memory_utilization",
                component="system",
                threshold_type="greater_than",
                threshold_value=85.0,
                severity=IncidentSeverity.MEDIUM,
                enabled=True,
                created_at=datetime.utcnow()
            )
        ]
        
        for rule in default_rules:
            self.threshold_rules[rule.rule_id] = rule
    
    async def _initialize_anomaly_detectors(self):
        """Initialize anomaly detection algorithms"""
        pass
    
    async def _load_baseline_metrics(self):
        """Load baseline metrics for comparison"""
        pass
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for incident management"""
        topics = [
            "q.incidents.detected",
            "q.incidents.resolved",
            "q.health.metrics",
            "q.anomalies.detected"
        ]
        
        for topic in topics:
            await self.pulsar_service.ensure_topic(topic)

# Global service instance
automated_incident_detection = AutomatedIncidentDetection() 