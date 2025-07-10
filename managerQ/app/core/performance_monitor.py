import logging
import time
import asyncio
import threading
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import statistics
from datetime import datetime, timedelta

from .agent_registry import AgentRegistry, Agent, AgentMetrics
from .task_dispatcher import TaskDispatcher
from .failure_handler import FailureHandler
from .agent_communication import AgentCommunicationHub
from .coordination_protocols import CoordinationProtocolManager
from shared.pulsar_client import SharedPulsarClient

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class PerformanceLevel(str, Enum):
    """Performance levels for health scoring"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Represents a performance metric"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags,
            'labels': self.labels
        }

@dataclass
class Alert:
    """Represents a performance alert"""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceInsight:
    """Represents a performance insight or recommendation"""
    insight_id: str
    category: str
    title: str
    description: str
    severity: str
    recommendation: str
    impact: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    applicable_agents: List[str] = field(default_factory=list)
    metrics_involved: List[str] = field(default_factory=list)

class MetricsCollector:
    """Collects metrics from various system components"""
    
    def __init__(self, agent_registry: AgentRegistry, task_dispatcher: TaskDispatcher,
                 failure_handler: FailureHandler, communication_hub: AgentCommunicationHub,
                 coordination_manager: CoordinationProtocolManager):
        self.agent_registry = agent_registry
        self.task_dispatcher = task_dispatcher
        self.failure_handler = failure_handler
        self.communication_hub = communication_hub
        self.coordination_manager = coordination_manager
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=100000)
        self.current_metrics: Dict[str, Metric] = {}
        self.metric_aggregators: Dict[str, List[float]] = defaultdict(list)
        
        # Collection settings
        self.collection_interval = 30  # seconds
        self.retention_hours = 24
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start metrics collection"""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")
    
    async def stop(self):
        """Stop metrics collection"""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self._running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _collect_all_metrics(self):
        """Collect all metrics from system components"""
        timestamp = time.time()
        
        # Collect agent metrics
        await self._collect_agent_metrics(timestamp)
        
        # Collect task dispatcher metrics
        await self._collect_dispatcher_metrics(timestamp)
        
        # Collect failure handler metrics
        await self._collect_failure_metrics(timestamp)
        
        # Collect communication metrics
        await self._collect_communication_metrics(timestamp)
        
        # Collect coordination metrics
        await self._collect_coordination_metrics(timestamp)
        
        # Collect system metrics
        await self._collect_system_metrics(timestamp)
        
        # Clean up old metrics
        self._cleanup_old_metrics(timestamp)
    
    async def _collect_agent_metrics(self, timestamp: float):
        """Collect metrics from agent registry"""
        registry_stats = self.agent_registry.get_registry_stats()
        
        # Agent count metrics
        self._record_metric("agents_total", MetricType.GAUGE, registry_stats['total_agents'], timestamp)
        self._record_metric("agents_healthy", MetricType.GAUGE, registry_stats['healthy_agents'], timestamp)
        self._record_metric("agents_degraded", MetricType.GAUGE, registry_stats['degraded_agents'], timestamp)
        self._record_metric("agents_unhealthy", MetricType.GAUGE, registry_stats['unhealthy_agents'], timestamp)
        
        # Capacity metrics
        self._record_metric("agents_total_capacity", MetricType.GAUGE, registry_stats['total_capacity'], timestamp)
        self._record_metric("agents_current_load", MetricType.GAUGE, registry_stats['current_load'], timestamp)
        self._record_metric("agents_load_percentage", MetricType.GAUGE, registry_stats['load_percentage'], timestamp)
        
        # Success rate
        self._record_metric("agents_avg_success_rate", MetricType.GAUGE, registry_stats['average_success_rate'], timestamp)
        
        # Per-agent metrics
        for agent in self.agent_registry.get_all_agents():
            agent_tags = {'agent_id': agent.agent_id, 'personality': agent.personality}
            
            self._record_metric("agent_tasks_completed", MetricType.COUNTER, 
                              agent.metrics.total_tasks_completed, timestamp, agent_tags)
            self._record_metric("agent_tasks_failed", MetricType.COUNTER, 
                              agent.metrics.total_tasks_failed, timestamp, agent_tags)
            self._record_metric("agent_current_load", MetricType.GAUGE, 
                              agent.metrics.current_load, timestamp, agent_tags)
            self._record_metric("agent_cpu_usage", MetricType.GAUGE, 
                              agent.metrics.cpu_usage, timestamp, agent_tags)
            self._record_metric("agent_memory_usage", MetricType.GAUGE, 
                              agent.metrics.memory_usage, timestamp, agent_tags)
            self._record_metric("agent_success_rate", MetricType.GAUGE, 
                              agent.metrics.success_rate, timestamp, agent_tags)
            self._record_metric("agent_error_rate", MetricType.GAUGE, 
                              agent.metrics.error_rate, timestamp, agent_tags)
            self._record_metric("agent_response_time_p95", MetricType.GAUGE, 
                              agent.metrics.response_time_p95, timestamp, agent_tags)
    
    async def _collect_dispatcher_metrics(self, timestamp: float):
        """Collect metrics from task dispatcher"""
        dispatcher_stats = self.task_dispatcher.get_queue_stats()
        
        self._record_metric("dispatcher_queue_size", MetricType.GAUGE, 
                          dispatcher_stats['queue_size'], timestamp)
        self._record_metric("dispatcher_pending_tasks", MetricType.GAUGE, 
                          dispatcher_stats['pending_tasks'], timestamp)
        self._record_metric("dispatcher_tasks_dispatched", MetricType.COUNTER, 
                          dispatcher_stats['tasks_dispatched'], timestamp)
        self._record_metric("dispatcher_tasks_failed", MetricType.COUNTER, 
                          dispatcher_stats['tasks_failed'], timestamp)
        self._record_metric("dispatcher_avg_queue_time", MetricType.GAUGE, 
                          dispatcher_stats['average_queue_time'], timestamp)
    
    async def _collect_failure_metrics(self, timestamp: float):
        """Collect metrics from failure handler"""
        failure_stats = self.failure_handler.get_failure_stats()
        
        self._record_metric("failures_total_types", MetricType.GAUGE, 
                          failure_stats['total_failure_types'], timestamp)
        self._record_metric("failures_active", MetricType.GAUGE, 
                          failure_stats['active_failures'], timestamp)
        
        # Per-failure-type metrics
        for failure_type, count in failure_stats['failure_counts_by_type'].items():
            failure_tags = {'failure_type': failure_type}
            self._record_metric("failures_by_type", MetricType.COUNTER, count, timestamp, failure_tags)
    
    async def _collect_communication_metrics(self, timestamp: float):
        """Collect metrics from communication hub"""
        comm_stats = self.communication_hub.get_communication_stats()
        
        self._record_metric("communication_total_messages", MetricType.COUNTER, 
                          comm_stats['total_messages'], timestamp)
        self._record_metric("communication_active_conversations", MetricType.GAUGE, 
                          comm_stats['active_conversations'], timestamp)
        self._record_metric("communication_registered_groups", MetricType.GAUGE, 
                          comm_stats['registered_groups'], timestamp)
        self._record_metric("communication_pending_deliveries", MetricType.GAUGE, 
                          comm_stats['pending_deliveries'], timestamp)
        
        # Per-message-type metrics
        for msg_type, count in comm_stats['message_types_distribution'].items():
            msg_tags = {'message_type': msg_type}
            self._record_metric("communication_messages_by_type", MetricType.COUNTER, 
                              count, timestamp, msg_tags)
    
    async def _collect_coordination_metrics(self, timestamp: float):
        """Collect metrics from coordination manager"""
        coord_stats = self.coordination_manager.get_coordination_status()
        
        self._record_metric("coordination_active_coordinations", MetricType.GAUGE, 
                          coord_stats['active_coordinations'], timestamp)
        self._record_metric("coordination_active_locks", MetricType.GAUGE, 
                          coord_stats['active_locks'], timestamp)
        self._record_metric("coordination_cluster_size", MetricType.GAUGE, 
                          len(coord_stats['cluster_members']), timestamp)
        self._record_metric("coordination_current_term", MetricType.GAUGE, 
                          coord_stats['current_term'], timestamp)
        
        # Leadership metrics
        coord_tags = {'state': coord_stats['consensus_state']}
        self._record_metric("coordination_node_state", MetricType.GAUGE, 1, timestamp, coord_tags)
        
        if coord_stats['is_leader']:
            self._record_metric("coordination_is_leader", MetricType.GAUGE, 1, timestamp)
        else:
            self._record_metric("coordination_is_leader", MetricType.GAUGE, 0, timestamp)
    
    async def _collect_system_metrics(self, timestamp: float):
        """Collect system-level metrics"""
        # Calculate derived metrics
        total_agents = len(self.agent_registry.get_all_agents())
        if total_agents > 0:
            healthy_agents = sum(1 for a in self.agent_registry.get_all_agents() 
                               if a.status.value == "healthy")
            system_health = (healthy_agents / total_agents) * 100
            self._record_metric("system_health_percentage", MetricType.GAUGE, system_health, timestamp)
        
        # Throughput metrics
        recent_metrics = [m for m in self.metrics_history if timestamp - m.timestamp < 300]  # 5 minutes
        task_metrics = [m for m in recent_metrics if m.name == "dispatcher_tasks_dispatched"]
        
        if len(task_metrics) > 1:
            throughput = (task_metrics[-1].value - task_metrics[0].value) / 300  # tasks per second
            self._record_metric("system_throughput", MetricType.GAUGE, throughput, timestamp)
    
    def _record_metric(self, name: str, metric_type: MetricType, value: float, 
                      timestamp: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric"""
        metric = Metric(
            name=name,
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            tags=tags or {}
        )
        
        self.metrics_history.append(metric)
        self.current_metrics[name] = metric
        
        # Update aggregators for statistics
        self.metric_aggregators[name].append(value)
        if len(self.metric_aggregators[name]) > 1000:  # Keep last 1000 values
            self.metric_aggregators[name] = self.metric_aggregators[name][-1000:]
    
    def _cleanup_old_metrics(self, current_timestamp: float):
        """Clean up old metrics"""
        cutoff_time = current_timestamp - (self.retention_hours * 3600)
        
        # Clean up history
        while self.metrics_history and self.metrics_history[0].timestamp < cutoff_time:
            self.metrics_history.popleft()
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[Metric]:
        """Get historical data for a metric"""
        cutoff_time = time.time() - (hours * 3600)
        return [m for m in self.metrics_history 
                if m.name == metric_name and m.timestamp >= cutoff_time]
    
    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        values = self.metric_aggregators.get(metric_name, [])
        if not values:
            return {}
        
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'percentile_95': statistics.quantiles(values, n=20)[18] if len(values) > 1 else values[0],
            'count': len(values)
        }

class AlertManager:
    """Manages performance alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []
        
        # Alert settings
        self.check_interval = 60  # seconds
        self._alert_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Load default alert rules
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default alert rules"""
        self.alert_rules = [
            {
                'name': 'high_agent_error_rate',
                'condition': lambda metrics: any(
                    m.value > 25 for m in metrics if m.name == 'agent_error_rate'
                ),
                'severity': AlertSeverity.WARNING,
                'message': 'High error rate detected on one or more agents'
            },
            {
                'name': 'agent_offline',
                'condition': lambda metrics: any(
                    m.value == 0 for m in metrics if m.name == 'agents_healthy'
                ),
                'severity': AlertSeverity.CRITICAL,
                'message': 'No healthy agents available'
            },
            {
                'name': 'high_queue_size',
                'condition': lambda metrics: any(
                    m.value > 100 for m in metrics if m.name == 'dispatcher_queue_size'
                ),
                'severity': AlertSeverity.WARNING,
                'message': 'Task queue size is high'
            },
            {
                'name': 'low_system_health',
                'condition': lambda metrics: any(
                    m.value < 70 for m in metrics if m.name == 'system_health_percentage'
                ),
                'severity': AlertSeverity.ERROR,
                'message': 'System health is below acceptable threshold'
            },
            {
                'name': 'high_response_time',
                'condition': lambda metrics: any(
                    m.value > 5000 for m in metrics if m.name == 'agent_response_time_p95'
                ),
                'severity': AlertSeverity.WARNING,
                'message': 'High response times detected'
            }
        ]
    
    async def start(self):
        """Start alert monitoring"""
        if self._running:
            return
        
        self._running = True
        self._alert_task = asyncio.create_task(self._alert_check_loop())
        logger.info("Alert monitoring started")
    
    async def stop(self):
        """Stop alert monitoring"""
        self._running = False
        if self._alert_task:
            self._alert_task.cancel()
        logger.info("Alert monitoring stopped")
    
    async def _alert_check_loop(self):
        """Main alert checking loop"""
        while self._running:
            try:
                await self._check_alerts()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in alert check loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _check_alerts(self):
        """Check all alert rules against current metrics"""
        recent_metrics = list(self.metrics_collector.metrics_history)[-100:]  # Last 100 metrics
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](recent_metrics):
                    await self._trigger_alert(rule)
                else:
                    await self._resolve_alert(rule['name'])
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    async def _trigger_alert(self, rule: Dict[str, Any]):
        """Trigger an alert"""
        alert_name = rule['name']
        
        if alert_name in self.alerts and not self.alerts[alert_name].resolved:
            return  # Alert already active
        
        alert = Alert(
            alert_id=f"alert_{alert_name}_{int(time.time())}",
            name=alert_name,
            severity=rule['severity'],
            message=rule['message'],
            timestamp=time.time()
        )
        
        self.alerts[alert_name] = alert
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
    
    async def _resolve_alert(self, alert_name: str):
        """Resolve an alert"""
        if alert_name in self.alerts and not self.alerts[alert_name].resolved:
            alert = self.alerts[alert_name]
            alert.resolved = True
            alert.resolved_at = time.time()
            
            logger.info(f"Alert resolved: {alert_name}")
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history"""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alerts.values() if alert.timestamp >= cutoff_time]

class PerformanceAnalyzer:
    """Analyzes performance data and provides insights"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.insights: List[PerformanceInsight] = []
        
        # Analysis settings
        self.analysis_interval = 300  # 5 minutes
        self._analysis_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start performance analysis"""
        if self._running:
            return
        
        self._running = True
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        logger.info("Performance analysis started")
    
    async def stop(self):
        """Stop performance analysis"""
        self._running = False
        if self._analysis_task:
            self._analysis_task.cancel()
        logger.info("Performance analysis stopped")
    
    async def _analysis_loop(self):
        """Main analysis loop"""
        while self._running:
            try:
                await self._analyze_performance()
                await asyncio.sleep(self.analysis_interval)
            except Exception as e:
                logger.error(f"Error in performance analysis loop: {e}", exc_info=True)
                await asyncio.sleep(30)
    
    async def _analyze_performance(self):
        """Analyze current performance and generate insights"""
        self.insights.clear()
        
        # Analyze agent performance
        await self._analyze_agent_performance()
        
        # Analyze system bottlenecks
        await self._analyze_bottlenecks()
        
        # Analyze resource utilization
        await self._analyze_resource_utilization()
        
        # Analyze load balancing
        await self._analyze_load_balancing()
    
    async def _analyze_agent_performance(self):
        """Analyze individual agent performance"""
        for agent in self.metrics_collector.agent_registry.get_all_agents():
            # Check for underperforming agents
            if agent.metrics.success_rate < 80:
                self.insights.append(PerformanceInsight(
                    insight_id=f"agent_performance_{agent.agent_id}",
                    category="agent_performance",
                    title=f"Low success rate for agent {agent.agent_id}",
                    description=f"Agent {agent.agent_id} has a success rate of {agent.metrics.success_rate:.1f}%",
                    severity="warning",
                    recommendation="Investigate agent logs and consider restarting or updating the agent",
                    impact="Reduced system reliability and task completion rates",
                    confidence=0.9,
                    applicable_agents=[agent.agent_id]
                ))
            
            # Check for overloaded agents
            if agent.metrics.load_factor > 0.9:
                self.insights.append(PerformanceInsight(
                    insight_id=f"agent_overload_{agent.agent_id}",
                    category="resource_utilization",
                    title=f"Agent {agent.agent_id} is overloaded",
                    description=f"Agent {agent.agent_id} is running at {agent.metrics.load_factor*100:.1f}% capacity",
                    severity="warning",
                    recommendation="Scale up agents or reduce task load",
                    impact="Increased response times and potential task failures",
                    confidence=0.95,
                    applicable_agents=[agent.agent_id]
                ))
    
    async def _analyze_bottlenecks(self):
        """Analyze system bottlenecks"""
        # Check queue sizes
        queue_stats = self.metrics_collector.task_dispatcher.get_queue_stats()
        if queue_stats['queue_size'] > 50:
            self.insights.append(PerformanceInsight(
                insight_id="queue_bottleneck",
                category="bottleneck",
                title="Task queue bottleneck detected",
                description=f"Queue size is {queue_stats['queue_size']} tasks",
                severity="warning",
                recommendation="Scale up agents or optimize task processing",
                impact="Increased task latency and potential SLA violations",
                confidence=0.8
            ))
        
        # Check average queue time
        if queue_stats['average_queue_time'] > 30:
            self.insights.append(PerformanceInsight(
                insight_id="queue_latency",
                category="latency",
                title="High task queue latency",
                description=f"Average queue time is {queue_stats['average_queue_time']:.1f} seconds",
                severity="warning",
                recommendation="Optimize task routing and agent allocation",
                impact="Poor user experience and system responsiveness",
                confidence=0.85
            ))
    
    async def _analyze_resource_utilization(self):
        """Analyze resource utilization patterns"""
        registry_stats = self.metrics_collector.agent_registry.get_registry_stats()
        
        # Check overall utilization
        if registry_stats['load_percentage'] > 85:
            self.insights.append(PerformanceInsight(
                insight_id="high_utilization",
                category="resource_utilization",
                title="High system utilization",
                description=f"System is running at {registry_stats['load_percentage']:.1f}% capacity",
                severity="warning",
                recommendation="Scale up the system or optimize resource allocation",
                impact="Risk of performance degradation and failures",
                confidence=0.9
            ))
        elif registry_stats['load_percentage'] < 20:
            self.insights.append(PerformanceInsight(
                insight_id="low_utilization",
                category="resource_utilization",
                title="Low system utilization",
                description=f"System is running at only {registry_stats['load_percentage']:.1f}% capacity",
                severity="info",
                recommendation="Consider scaling down to reduce costs",
                impact="Resource waste and unnecessary costs",
                confidence=0.7
            ))
    
    async def _analyze_load_balancing(self):
        """Analyze load balancing effectiveness"""
        agents = self.metrics_collector.agent_registry.get_all_agents()
        if len(agents) < 2:
            return
        
        # Calculate load variance
        loads = [agent.metrics.current_load for agent in agents]
        if loads:
            avg_load = sum(loads) / len(loads)
            variance = sum((load - avg_load) ** 2 for load in loads) / len(loads)
            
            if variance > 10:  # High variance indicates poor load balancing
                self.insights.append(PerformanceInsight(
                    insight_id="load_imbalance",
                    category="load_balancing",
                    title="Poor load balancing detected",
                    description=f"Load variance is {variance:.1f}, indicating uneven distribution",
                    severity="warning",
                    recommendation="Review load balancing strategy and agent allocation",
                    impact="Suboptimal resource utilization and performance",
                    confidence=0.8
                ))
    
    def get_insights(self, category: Optional[str] = None) -> List[PerformanceInsight]:
        """Get performance insights"""
        if category:
            return [insight for insight in self.insights if insight.category == category]
        return self.insights.copy()
    
    def get_health_score(self) -> Tuple[float, PerformanceLevel]:
        """Calculate overall system health score"""
        registry_stats = self.metrics_collector.agent_registry.get_registry_stats()
        
        # Calculate component scores
        agent_health = registry_stats['healthy_agents'] / max(registry_stats['total_agents'], 1)
        success_rate = registry_stats['average_success_rate'] / 100
        
        # Load balance score (inverted variance)
        agents = self.metrics_collector.agent_registry.get_all_agents()
        if len(agents) > 1:
            loads = [agent.metrics.current_load for agent in agents]
            avg_load = sum(loads) / len(loads)
            variance = sum((load - avg_load) ** 2 for load in loads) / len(loads)
            load_balance_score = max(0, 1 - (variance / 20))
        else:
            load_balance_score = 1.0
        
        # Alert penalty
        active_alerts = len([i for i in self.insights if i.severity in ['error', 'critical']])
        alert_penalty = max(0, 1 - (active_alerts * 0.1))
        
        # Overall score
        health_score = (agent_health * 0.3 + success_rate * 0.3 + 
                       load_balance_score * 0.2 + alert_penalty * 0.2)
        
        # Determine performance level
        if health_score >= 0.9:
            level = PerformanceLevel.EXCELLENT
        elif health_score >= 0.8:
            level = PerformanceLevel.GOOD
        elif health_score >= 0.6:
            level = PerformanceLevel.FAIR
        elif health_score >= 0.4:
            level = PerformanceLevel.POOR
        else:
            level = PerformanceLevel.CRITICAL
        
        return health_score, level

class PerformanceMonitor:
    """Main performance monitoring coordinator"""
    
    def __init__(self, agent_registry: AgentRegistry, task_dispatcher: TaskDispatcher,
                 failure_handler: FailureHandler, communication_hub: AgentCommunicationHub,
                 coordination_manager: CoordinationProtocolManager):
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            agent_registry, task_dispatcher, failure_handler, 
            communication_hub, coordination_manager
        )
        self.alert_manager = AlertManager(self.metrics_collector)
        self.performance_analyzer = PerformanceAnalyzer(self.metrics_collector)
        
        self._running = False
    
    async def start(self):
        """Start the performance monitoring system"""
        if self._running:
            return
        
        self._running = True
        
        # Start all components
        await self.metrics_collector.start()
        await self.alert_manager.start()
        await self.performance_analyzer.start()
        
        logger.info("Performance monitoring system started")
    
    async def stop(self):
        """Stop the performance monitoring system"""
        self._running = False
        
        # Stop all components
        await self.metrics_collector.stop()
        await self.alert_manager.stop()
        await self.performance_analyzer.stop()
        
        logger.info("Performance monitoring system stopped")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        health_score, health_level = self.performance_analyzer.get_health_score()
        
        return {
            'health_score': health_score,
            'health_level': health_level.value,
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'insights': len(self.performance_analyzer.get_insights()),
            'current_metrics': {
                name: metric.to_dict() 
                for name, metric in self.metrics_collector.current_metrics.items()
            },
            'recent_alerts': [
                {
                    'name': alert.name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                }
                for alert in self.alert_manager.get_alert_history(1)  # Last hour
            ],
            'top_insights': [
                {
                    'title': insight.title,
                    'category': insight.category,
                    'severity': insight.severity,
                    'recommendation': insight.recommendation,
                    'confidence': insight.confidence
                }
                for insight in self.performance_analyzer.get_insights()[:5]
            ]
        }
    
    def get_metric_data(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get detailed metric data"""
        history = self.metrics_collector.get_metric_history(metric_name, hours)
        statistics = self.metrics_collector.get_metric_statistics(metric_name)
        
        return {
            'metric_name': metric_name,
            'history': [metric.to_dict() for metric in history],
            'statistics': statistics,
            'current_value': self.metrics_collector.current_metrics.get(metric_name, {}).get('value', 0)
        }

# Singleton instance
performance_monitor: Optional[PerformanceMonitor] = None 