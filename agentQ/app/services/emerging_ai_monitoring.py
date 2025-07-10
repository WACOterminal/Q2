"""
Emerging AI Monitoring and Observability Service

This service provides comprehensive monitoring for emerging AI systems:
- Real-time metrics collection
- Performance tracking and analysis
- Alerting and anomaly detection
- Dashboard data preparation
- System health monitoring
- Resource utilization tracking
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import json
import statistics
from collections import defaultdict, deque

# Q Platform imports
from app.services.pulsar_service import PulsarService
from app.services.ignite_service import IgniteService
from app.services.memory_service import MemoryService
from shared.q_memory_schemas.memory_models import AgentMemory, MemoryType

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    GAUGE = "gauge"           # Point-in-time value
    COUNTER = "counter"       # Cumulative value
    HISTOGRAM = "histogram"   # Distribution of values
    TIMING = "timing"         # Duration measurements

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ComponentType(Enum):
    """Emerging AI component types"""
    QUANTUM_OPTIMIZATION = "quantum_optimization"
    QUANTUM_ML = "quantum_ml"
    SPIKING_NETWORKS = "spiking_networks"
    NEUROMORPHIC_ENGINE = "neuromorphic_engine"
    ENERGY_COMPUTING = "energy_computing"
    HYBRID_WORKFLOWS = "hybrid_workflows"

@dataclass
class Metric:
    """Individual metric data point"""
    metric_id: str
    component: ComponentType
    metric_type: MetricType
    name: str
    value: float
    tags: Dict[str, str]
    timestamp: datetime
    unit: str = ""
    description: str = ""

@dataclass
class Alert:
    """Alert definition and state"""
    alert_id: str
    component: ComponentType
    severity: AlertSeverity
    condition: str
    threshold: float
    message: str
    is_active: bool
    triggered_at: Optional[datetime]
    resolved_at: Optional[datetime]
    occurrences: int

@dataclass
class PerformanceReport:
    """Performance analysis report"""
    report_id: str
    component: ComponentType
    time_period: timedelta
    metrics_analyzed: int
    
    # Performance summary
    avg_response_time: float
    avg_throughput: float
    avg_cpu_utilization: float
    avg_memory_utilization: float
    
    # Quantum-specific metrics
    avg_quantum_advantage: Optional[float]
    quantum_fidelity: Optional[float]
    
    # Neuromorphic-specific metrics
    avg_spike_rate: Optional[float]
    energy_efficiency: Optional[float]
    
    # Anomalies and trends
    anomalies_detected: int
    performance_trend: str  # "improving", "stable", "degrading"
    
    created_at: datetime

class EmergingAIMonitoring:
    """
    Monitoring and observability service for emerging AI systems
    """
    
    def __init__(self):
        self.pulsar_service = PulsarService()
        self.ignite_service = IgniteService()
        self.memory_service = MemoryService()
        
        # Metrics storage (in-memory circular buffers)
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, Alert] = {}
        self.performance_reports: List[PerformanceReport] = []
        
        # Metric collectors
        self.metric_collectors: Dict[ComponentType, Callable] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        
        # Configuration
        self.config = {
            "metrics_collection_interval": 10.0,  # seconds
            "performance_analysis_interval": 300.0,  # 5 minutes
            "alerting_enabled": True,
            "dashboard_update_interval": 30.0,  # seconds
            "metric_retention_hours": 24,
            "anomaly_detection_enabled": True
        }
        
        # System state tracking
        self.system_health = {
            "overall_status": "healthy",
            "component_statuses": {},
            "last_health_check": datetime.utcnow(),
            "active_alerts": 0,
            "performance_score": 1.0
        }
    
    async def initialize(self):
        """Initialize the monitoring service"""
        logger.info("Initializing Emerging AI Monitoring Service")
        
        # Setup metric collectors
        await self._setup_metric_collectors()
        
        # Setup alert rules
        await self._setup_alert_rules()
        
        # Start background tasks
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._performance_analysis_loop())
        asyncio.create_task(self._alerting_loop())
        asyncio.create_task(self._dashboard_update_loop())
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        logger.info("Emerging AI Monitoring Service initialized successfully")
    
    # ===== METRICS COLLECTION =====
    
    async def collect_metric(self, metric: Metric):
        """Collect a single metric"""
        # Store in buffer
        metric_key = f"{metric.component.value}:{metric.name}"
        self.metrics_buffer[metric_key].append(metric)
        
        # Publish to real-time stream
        await self.pulsar_service.publish(
            "q.monitoring.metrics.realtime",
            asdict(metric)
        )
        
        # Check for alert conditions
        if self.config["alerting_enabled"]:
            await self._check_alert_conditions(metric)
    
    async def collect_quantum_metrics(self):
        """Collect quantum computing metrics"""
        try:
            from app.services.quantum_optimization_service import quantum_optimization_service
            from app.services.quantum_ml_experiments import quantum_ml_experiments
            
            # Quantum optimization metrics
            opt_metrics = quantum_optimization_service.metrics
            
            await self.collect_metric(Metric(
                metric_id=f"quantum_opt_{uuid.uuid4().hex[:8]}",
                component=ComponentType.QUANTUM_OPTIMIZATION,
                metric_type=MetricType.COUNTER,
                name="problems_solved",
                value=float(opt_metrics.get("problems_solved", 0)),
                tags={"service": "quantum_optimization"},
                timestamp=datetime.utcnow(),
                unit="count",
                description="Total quantum optimization problems solved"
            ))
            
            await self.collect_metric(Metric(
                metric_id=f"quantum_adv_{uuid.uuid4().hex[:8]}",
                component=ComponentType.QUANTUM_OPTIMIZATION,
                metric_type=MetricType.GAUGE,
                name="average_quantum_advantage",
                value=float(opt_metrics.get("average_quantum_advantage", 1.0)),
                tags={"service": "quantum_optimization"},
                timestamp=datetime.utcnow(),
                unit="ratio",
                description="Average quantum advantage achieved"
            ))
            
            # Quantum ML metrics
            ml_metrics = quantum_ml_experiments.metrics
            
            await self.collect_metric(Metric(
                metric_id=f"quantum_ml_{uuid.uuid4().hex[:8]}",
                component=ComponentType.QUANTUM_ML,
                metric_type=MetricType.COUNTER,
                name="experiments_completed",
                value=float(ml_metrics.get("experiments_completed", 0)),
                tags={"service": "quantum_ml"},
                timestamp=datetime.utcnow(),
                unit="count",
                description="Total quantum ML experiments completed"
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect quantum metrics: {e}")
    
    async def collect_neuromorphic_metrics(self):
        """Collect neuromorphic computing metrics"""
        try:
            from app.services.spiking_neural_networks import spiking_neural_networks
            from app.services.neuromorphic_engine import neuromorphic_engine
            
            # Spiking networks metrics
            snn_metrics = spiking_neural_networks.metrics
            
            await self.collect_metric(Metric(
                metric_id=f"snn_spikes_{uuid.uuid4().hex[:8]}",
                component=ComponentType.SPIKING_NETWORKS,
                metric_type=MetricType.COUNTER,
                name="total_spikes_processed",
                value=float(snn_metrics.get("total_spikes_processed", 0)),
                tags={"service": "spiking_networks"},
                timestamp=datetime.utcnow(),
                unit="spikes",
                description="Total spikes processed by spiking networks"
            ))
            
            await self.collect_metric(Metric(
                metric_id=f"snn_energy_{uuid.uuid4().hex[:8]}",
                component=ComponentType.SPIKING_NETWORKS,
                metric_type=MetricType.GAUGE,
                name="total_energy_consumed",
                value=float(snn_metrics.get("total_energy_consumed", 0.0)),
                tags={"service": "spiking_networks"},
                timestamp=datetime.utcnow(),
                unit="picojoules",
                description="Total energy consumed by spiking networks"
            ))
            
            # Neuromorphic engine metrics
            neuro_metrics = neuromorphic_engine.metrics
            
            await self.collect_metric(Metric(
                metric_id=f"neuro_tasks_{uuid.uuid4().hex[:8]}",
                component=ComponentType.NEUROMORPHIC_ENGINE,
                metric_type=MetricType.COUNTER,
                name="tasks_completed",
                value=float(neuro_metrics.get("tasks_completed", 0)),
                tags={"service": "neuromorphic_engine"},
                timestamp=datetime.utcnow(),
                unit="count",
                description="Total neuromorphic tasks completed"
            ))
            
            await self.collect_metric(Metric(
                metric_id=f"neuro_accuracy_{uuid.uuid4().hex[:8]}",
                component=ComponentType.NEUROMORPHIC_ENGINE,
                metric_type=MetricType.GAUGE,
                name="average_accuracy",
                value=float(neuro_metrics.get("average_accuracy", 0.0)),
                tags={"service": "neuromorphic_engine"},
                timestamp=datetime.utcnow(),
                unit="ratio",
                description="Average accuracy of neuromorphic tasks"
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect neuromorphic metrics: {e}")
    
    async def collect_energy_metrics(self):
        """Collect energy efficiency metrics"""
        try:
            from app.services.energy_efficient_computing import energy_efficient_computing
            
            energy_metrics = energy_efficient_computing.metrics
            
            await self.collect_metric(Metric(
                metric_id=f"energy_consumed_{uuid.uuid4().hex[:8]}",
                component=ComponentType.ENERGY_COMPUTING,
                metric_type=MetricType.COUNTER,
                name="total_energy_consumed",
                value=float(energy_metrics.get("total_energy_consumed", 0.0)),
                tags={"service": "energy_computing"},
                timestamp=datetime.utcnow(),
                unit="joules",
                description="Total energy consumed by all devices"
            ))
            
            await self.collect_metric(Metric(
                metric_id=f"energy_saved_{uuid.uuid4().hex[:8]}",
                component=ComponentType.ENERGY_COMPUTING,
                metric_type=MetricType.COUNTER,
                name="total_energy_saved",
                value=float(energy_metrics.get("total_energy_saved", 0.0)),
                tags={"service": "energy_computing"},
                timestamp=datetime.utcnow(),
                unit="joules",
                description="Total energy saved through optimization"
            ))
            
            await self.collect_metric(Metric(
                metric_id=f"energy_efficiency_{uuid.uuid4().hex[:8]}",
                component=ComponentType.ENERGY_COMPUTING,
                metric_type=MetricType.GAUGE,
                name="average_efficiency",
                value=float(energy_metrics.get("average_efficiency", 0.0)),
                tags={"service": "energy_computing"},
                timestamp=datetime.utcnow(),
                unit="ops_per_joule",
                description="Average energy efficiency across devices"
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect energy metrics: {e}")
    
    # ===== PERFORMANCE ANALYSIS =====
    
    async def analyze_performance(self, component: ComponentType, time_period: timedelta) -> PerformanceReport:
        """Analyze performance for a specific component"""
        logger.info(f"Analyzing performance for {component.value}")
        
        # Get metrics for the time period
        end_time = datetime.utcnow()
        start_time = end_time - time_period
        
        component_metrics = []
        for metric_key, metrics in self.metrics_buffer.items():
            if metric_key.startswith(component.value):
                component_metrics.extend([
                    m for m in metrics 
                    if start_time <= m.timestamp <= end_time
                ])
        
        if not component_metrics:
            logger.warning(f"No metrics found for {component.value}")
            return self._create_empty_report(component, time_period)
        
        # Calculate performance statistics
        response_times = [m.value for m in component_metrics if "response_time" in m.name]
        throughput_values = [m.value for m in component_metrics if "throughput" in m.name]
        quantum_advantages = [m.value for m in component_metrics if "quantum_advantage" in m.name]
        spike_rates = [m.value for m in component_metrics if "spike_rate" in m.name]
        energy_efficiency = [m.value for m in component_metrics if "efficiency" in m.name]
        
        # Detect anomalies
        anomalies = await self._detect_anomalies(component_metrics)
        
        # Determine performance trend
        trend = await self._calculate_trend(component_metrics)
        
        report = PerformanceReport(
            report_id=f"perf_{component.value}_{uuid.uuid4().hex[:8]}",
            component=component,
            time_period=time_period,
            metrics_analyzed=len(component_metrics),
            avg_response_time=statistics.mean(response_times) if response_times else 0.0,
            avg_throughput=statistics.mean(throughput_values) if throughput_values else 0.0,
            avg_cpu_utilization=0.5,  # Mock value
            avg_memory_utilization=0.3,  # Mock value
            avg_quantum_advantage=statistics.mean(quantum_advantages) if quantum_advantages else None,
            quantum_fidelity=0.95 if component in [ComponentType.QUANTUM_OPTIMIZATION, ComponentType.QUANTUM_ML] else None,
            avg_spike_rate=statistics.mean(spike_rates) if spike_rates else None,
            energy_efficiency=statistics.mean(energy_efficiency) if energy_efficiency else None,
            anomalies_detected=len(anomalies),
            performance_trend=trend,
            created_at=datetime.utcnow()
        )
        
        self.performance_reports.append(report)
        
        # Store significant performance changes as memories
        if report.anomalies_detected > 0 or trend == "degrading":
            await self._store_performance_memory(report)
        
        return report
    
    async def _detect_anomalies(self, metrics: List[Metric]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics using simple statistical methods"""
        anomalies = []
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric.value)
        
        # Check for statistical anomalies
        for metric_name, values in metrics_by_name.items():
            if len(values) < 10:  # Need minimum data points
                continue
            
            mean_val = statistics.mean(values)
            stdev_val = statistics.stdev(values) if len(values) > 1 else 0
            
            # Values more than 2 standard deviations from mean are anomalies
            for i, value in enumerate(values):
                if abs(value - mean_val) > 2 * stdev_val:
                    anomalies.append({
                        "metric_name": metric_name,
                        "value": value,
                        "expected_range": (mean_val - 2*stdev_val, mean_val + 2*stdev_val),
                        "deviation": abs(value - mean_val) / stdev_val if stdev_val > 0 else 0
                    })
        
        return anomalies
    
    async def _calculate_trend(self, metrics: List[Metric]) -> str:
        """Calculate performance trend over time"""
        if len(metrics) < 5:
            return "stable"
        
        # Group by time windows and calculate averages
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        window_size = len(sorted_metrics) // 3
        
        if window_size < 2:
            return "stable"
        
        early_metrics = sorted_metrics[:window_size]
        late_metrics = sorted_metrics[-window_size:]
        
        early_avg = statistics.mean([m.value for m in early_metrics])
        late_avg = statistics.mean([m.value for m in late_metrics])
        
        # Calculate trend
        if late_avg > early_avg * 1.1:
            return "improving"
        elif late_avg < early_avg * 0.9:
            return "degrading"
        else:
            return "stable"
    
    def _create_empty_report(self, component: ComponentType, time_period: timedelta) -> PerformanceReport:
        """Create empty performance report when no metrics available"""
        return PerformanceReport(
            report_id=f"empty_{component.value}_{uuid.uuid4().hex[:8]}",
            component=component,
            time_period=time_period,
            metrics_analyzed=0,
            avg_response_time=0.0,
            avg_throughput=0.0,
            avg_cpu_utilization=0.0,
            avg_memory_utilization=0.0,
            avg_quantum_advantage=None,
            quantum_fidelity=None,
            avg_spike_rate=None,
            energy_efficiency=None,
            anomalies_detected=0,
            performance_trend="unknown",
            created_at=datetime.utcnow()
        )
    
    # ===== ALERTING =====
    
    async def _check_alert_conditions(self, metric: Metric):
        """Check if metric triggers any alert conditions"""
        for rule in self.alert_rules:
            if (rule["component"] == metric.component and 
                rule["metric_name"] == metric.name):
                
                condition_met = False
                
                if rule["condition"] == "greater_than":
                    condition_met = metric.value > rule["threshold"]
                elif rule["condition"] == "less_than":
                    condition_met = metric.value < rule["threshold"]
                elif rule["condition"] == "equals":
                    condition_met = abs(metric.value - rule["threshold"]) < 0.001
                
                alert_id = rule["alert_id"]
                
                if condition_met:
                    await self._trigger_alert(alert_id, metric)
                else:
                    await self._resolve_alert(alert_id)
    
    async def _trigger_alert(self, alert_id: str, triggering_metric: Metric):
        """Trigger an alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            if not alert.is_active:
                alert.is_active = True
                alert.triggered_at = datetime.utcnow()
                alert.resolved_at = None
            alert.occurrences += 1
        else:
            # Find alert rule
            rule = next((r for r in self.alert_rules if r["alert_id"] == alert_id), None)
            if rule:
                alert = Alert(
                    alert_id=alert_id,
                    component=rule["component"],
                    severity=AlertSeverity(rule["severity"]),
                    condition=rule["condition"],
                    threshold=rule["threshold"],
                    message=rule["message"],
                    is_active=True,
                    triggered_at=datetime.utcnow(),
                    resolved_at=None,
                    occurrences=1
                )
                self.alerts[alert_id] = alert
        
        # Publish alert
        await self.pulsar_service.publish(
            "q.monitoring.alerts.triggered",
            {
                "alert": asdict(self.alerts[alert_id]),
                "triggering_metric": asdict(triggering_metric),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.warning(f"Alert triggered: {alert_id} - {self.alerts[alert_id].message}")
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.alerts and self.alerts[alert_id].is_active:
            alert = self.alerts[alert_id]
            alert.is_active = False
            alert.resolved_at = datetime.utcnow()
            
            # Publish resolution
            await self.pulsar_service.publish(
                "q.monitoring.alerts.resolved",
                {
                    "alert": asdict(alert),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Alert resolved: {alert_id}")
    
    # ===== DASHBOARD DATA =====
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": self.system_health,
            "component_metrics": {},
            "active_alerts": [],
            "recent_performance": {},
            "trends": {}
        }
        
        # Get latest metrics for each component
        for component in ComponentType:
            component_data = await self._get_component_dashboard_data(component)
            dashboard_data["component_metrics"][component.value] = component_data
        
        # Get active alerts
        dashboard_data["active_alerts"] = [
            asdict(alert) for alert in self.alerts.values() if alert.is_active
        ]
        
        # Get recent performance reports
        recent_reports = sorted(self.performance_reports, key=lambda r: r.created_at, reverse=True)[:10]
        dashboard_data["recent_performance"] = [asdict(report) for report in recent_reports]
        
        # Calculate trends
        dashboard_data["trends"] = await self._calculate_dashboard_trends()
        
        return dashboard_data
    
    async def _get_component_dashboard_data(self, component: ComponentType) -> Dict[str, Any]:
        """Get dashboard data for a specific component"""
        component_metrics = {}
        
        # Get latest metrics for this component
        for metric_key, metrics in self.metrics_buffer.items():
            if metric_key.startswith(component.value) and metrics:
                latest_metric = metrics[-1]
                component_metrics[latest_metric.name] = {
                    "value": latest_metric.value,
                    "unit": latest_metric.unit,
                    "timestamp": latest_metric.timestamp.isoformat()
                }
        
        return component_metrics
    
    async def _calculate_dashboard_trends(self) -> Dict[str, Any]:
        """Calculate trends for dashboard display"""
        trends = {}
        
        # Calculate 24-hour trends for key metrics
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        for component in ComponentType:
            component_trends = {}
            
            for metric_key, metrics in self.metrics_buffer.items():
                if metric_key.startswith(component.value):
                    recent_metrics = [
                        m for m in metrics 
                        if start_time <= m.timestamp <= end_time
                    ]
                    
                    if len(recent_metrics) >= 2:
                        values = [m.value for m in recent_metrics]
                        
                        # Calculate percentage change
                        first_half = values[:len(values)//2]
                        second_half = values[len(values)//2:]
                        
                        if first_half and second_half:
                            first_avg = statistics.mean(first_half)
                            second_avg = statistics.mean(second_half)
                            
                            if first_avg != 0:
                                percent_change = ((second_avg - first_avg) / first_avg) * 100
                                component_trends[recent_metrics[0].name] = {
                                    "percent_change": percent_change,
                                    "direction": "up" if percent_change > 0 else "down"
                                }
            
            trends[component.value] = component_trends
        
        return trends
    
    # ===== BACKGROUND TASKS =====
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while True:
            try:
                await asyncio.sleep(self.config["metrics_collection_interval"])
                
                # Collect metrics from all components
                await self.collect_quantum_metrics()
                await self.collect_neuromorphic_metrics()
                await self.collect_energy_metrics()
                
                # Update system health
                await self._update_system_health()
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
    
    async def _performance_analysis_loop(self):
        """Background performance analysis loop"""
        while True:
            try:
                await asyncio.sleep(self.config["performance_analysis_interval"])
                
                # Analyze performance for each component
                analysis_period = timedelta(hours=1)
                
                for component in ComponentType:
                    await self.analyze_performance(component, analysis_period)
                
            except Exception as e:
                logger.error(f"Error in performance analysis loop: {e}")
    
    async def _alerting_loop(self):
        """Background alerting loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Update system health based on active alerts
                active_alerts = [alert for alert in self.alerts.values() if alert.is_active]
                self.system_health["active_alerts"] = len(active_alerts)
                
                # Determine overall system status
                critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
                error_alerts = [a for a in active_alerts if a.severity == AlertSeverity.ERROR]
                
                if critical_alerts:
                    self.system_health["overall_status"] = "critical"
                elif error_alerts:
                    self.system_health["overall_status"] = "degraded"
                elif active_alerts:
                    self.system_health["overall_status"] = "warning"
                else:
                    self.system_health["overall_status"] = "healthy"
                
            except Exception as e:
                logger.error(f"Error in alerting loop: {e}")
    
    async def _dashboard_update_loop(self):
        """Background dashboard update loop"""
        while True:
            try:
                await asyncio.sleep(self.config["dashboard_update_interval"])
                
                # Prepare and publish dashboard data
                dashboard_data = await self.get_dashboard_data()
                
                await self.pulsar_service.publish(
                    "q.monitoring.dashboard.update",
                    dashboard_data
                )
                
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
    
    # ===== HELPER METHODS =====
    
    async def _update_system_health(self):
        """Update overall system health status"""
        self.system_health["last_health_check"] = datetime.utcnow()
        
        # Calculate performance score based on recent metrics
        recent_reports = [r for r in self.performance_reports if 
                        (datetime.utcnow() - r.created_at).total_seconds() < 3600]  # Last hour
        
        if recent_reports:
            # Score based on anomalies and trends
            total_anomalies = sum(r.anomalies_detected for r in recent_reports)
            degrading_trends = sum(1 for r in recent_reports if r.performance_trend == "degrading")
            
            # Simple scoring algorithm
            base_score = 1.0
            anomaly_penalty = min(0.5, total_anomalies * 0.1)
            trend_penalty = min(0.3, degrading_trends * 0.1)
            
            self.system_health["performance_score"] = max(0.0, base_score - anomaly_penalty - trend_penalty)
        
        # Update component statuses
        for component in ComponentType:
            component_alerts = [a for a in self.alerts.values() 
                              if a.component == component and a.is_active]
            
            if component_alerts:
                max_severity = max(a.severity for a in component_alerts)
                self.system_health["component_statuses"][component.value] = max_severity.value
            else:
                self.system_health["component_statuses"][component.value] = "healthy"
    
    async def _store_performance_memory(self, report: PerformanceReport):
        """Store significant performance events as memories"""
        memory = AgentMemory(
            memory_id=f"perf_{report.report_id}",
            agent_id="emerging_ai_monitoring",
            memory_type=MemoryType.EXPERIENCE,
            content=f"Performance analysis for {report.component.value}: {report.performance_trend} trend, {report.anomalies_detected} anomalies",
            context=asdict(report),
            importance=0.8 if report.anomalies_detected > 0 else 0.5,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1
        )
        
        await self.memory_service.store_memory(memory)
    
    async def _setup_metric_collectors(self):
        """Setup metric collector functions"""
        self.metric_collectors = {
            ComponentType.QUANTUM_OPTIMIZATION: self.collect_quantum_metrics,
            ComponentType.QUANTUM_ML: self.collect_quantum_metrics,
            ComponentType.SPIKING_NETWORKS: self.collect_neuromorphic_metrics,
            ComponentType.NEUROMORPHIC_ENGINE: self.collect_neuromorphic_metrics,
            ComponentType.ENERGY_COMPUTING: self.collect_energy_metrics
        }
    
    async def _setup_alert_rules(self):
        """Setup default alert rules"""
        self.alert_rules = [
            {
                "alert_id": "quantum_advantage_low",
                "component": ComponentType.QUANTUM_OPTIMIZATION,
                "metric_name": "average_quantum_advantage",
                "condition": "less_than",
                "threshold": 0.8,
                "severity": "warning",
                "message": "Quantum advantage below expected threshold"
            },
            {
                "alert_id": "energy_efficiency_low",
                "component": ComponentType.ENERGY_COMPUTING,
                "metric_name": "average_efficiency",
                "condition": "less_than",
                "threshold": 500.0,
                "severity": "warning",
                "message": "Energy efficiency below optimal level"
            },
            {
                "alert_id": "neuromorphic_accuracy_low",
                "component": ComponentType.NEUROMORPHIC_ENGINE,
                "metric_name": "average_accuracy",
                "condition": "less_than",
                "threshold": 0.7,
                "severity": "error",
                "message": "Neuromorphic task accuracy below acceptable level"
            }
        ]
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for monitoring"""
        topics = [
            "q.monitoring.metrics.realtime",
            "q.monitoring.alerts.triggered",
            "q.monitoring.alerts.resolved",
            "q.monitoring.dashboard.update",
            "q.monitoring.performance.report"
        ]
        
        for topic in topics:
            await self.pulsar_service.ensure_topic(topic)

# Global service instance
emerging_ai_monitoring = EmergingAIMonitoring() 