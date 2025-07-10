import logging
import time
import asyncio
import threading
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import numpy as np
from datetime import datetime, timedelta
import statistics

from kubernetes import client, config

from .agent_registry import AgentRegistry
from .task_dispatcher import TaskDispatcher
from .performance_monitor import PerformanceMonitor, MetricsCollector
from .autoscaler import AutoScaler

logger = logging.getLogger(__name__)

# Initialize Kubernetes client
try:
    config.load_incluster_config()
    k8s_apps_v1 = client.AppsV1Api()
    k8s_metrics_v1 = client.CustomObjectsApi()
    logger.info("PredictiveAutoScaler: Loaded in-cluster Kubernetes config.")
except config.ConfigException:
    try:
        config.load_kube_config()
        k8s_apps_v1 = client.AppsV1Api()
        k8s_metrics_v1 = client.CustomObjectsApi()
        logger.info("PredictiveAutoScaler: Loaded local kubeconfig.")
    except config.ConfigException:
        logger.warning("Could not configure Kubernetes client for PredictiveAutoScaler.")
        k8s_apps_v1 = None
        k8s_metrics_v1 = None

class ScalingStrategy(str, Enum):
    """Different scaling strategies"""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"

class ScalingTrigger(str, Enum):
    """Reasons for scaling decisions"""
    QUEUE_DEPTH = "queue_depth"
    PREDICTION = "prediction"
    RESOURCE_PRESSURE = "resource_pressure"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    MANUAL = "manual"

@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    cpu_usage: float
    memory_usage: float
    network_io: float
    disk_io: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class ScalingDecision:
    """Represents a scaling decision"""
    deployment_name: str
    current_replicas: int
    target_replicas: int
    trigger: ScalingTrigger
    confidence: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)
    predicted_impact: Dict[str, float] = field(default_factory=dict)
    cost_impact: float = 0.0

@dataclass
class PredictionData:
    """Prediction data for workload forecasting"""
    timestamp: float
    predicted_load: float
    confidence: float
    time_horizon: int  # minutes
    factors: Dict[str, float] = field(default_factory=dict)

class WorkloadPredictor:
    """Predicts future workload using simple machine learning"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.load_history: deque = deque(maxlen=history_size)
        self.time_series: deque = deque(maxlen=history_size)
        self.seasonal_patterns: Dict[str, List[float]] = {}
        
        # Simple linear regression coefficients
        self.trend_coefficient = 0.0
        self.intercept = 0.0
        self.last_update = 0.0
        
    def add_data_point(self, timestamp: float, load: float):
        """Add a new data point to the predictor"""
        self.load_history.append(load)
        self.time_series.append(timestamp)
        
        # Update model periodically
        if time.time() - self.last_update > 300:  # 5 minutes
            self._update_model()
            self.last_update = time.time()
    
    def _update_model(self):
        """Update the prediction model with latest data"""
        if len(self.load_history) < 10:
            return
        
        # Simple linear regression for trend
        x = np.array(range(len(self.load_history)))
        y = np.array(self.load_history)
        
        if len(x) > 1:
            # Calculate trend
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator != 0:
                self.trend_coefficient = numerator / denominator
                self.intercept = y_mean - self.trend_coefficient * x_mean
        
        # Update seasonal patterns
        self._update_seasonal_patterns()
    
    def _update_seasonal_patterns(self):
        """Update seasonal patterns (hourly, daily)"""
        if len(self.time_series) < 24:
            return
        
        # Group by hour of day
        hourly_loads = defaultdict(list)
        for i, timestamp in enumerate(self.time_series):
            hour = datetime.fromtimestamp(timestamp).hour
            hourly_loads[hour].append(self.load_history[i])
        
        # Calculate average load for each hour
        self.seasonal_patterns['hourly'] = []
        for hour in range(24):
            if hour in hourly_loads:
                avg_load = statistics.mean(hourly_loads[hour])
                self.seasonal_patterns['hourly'].append(avg_load)
            else:
                self.seasonal_patterns['hourly'].append(0.0)
    
    def predict(self, time_horizon: int = 15) -> PredictionData:
        """Predict load for the next time_horizon minutes"""
        if len(self.load_history) < 5:
            # Not enough data for prediction
            current_load = self.load_history[-1] if self.load_history else 0
            return PredictionData(
                timestamp=time.time(),
                predicted_load=current_load,
                confidence=0.1,
                time_horizon=time_horizon
            )
        
        # Base prediction using trend
        future_point = len(self.load_history) + (time_horizon / 5)  # Assuming 5-minute intervals
        trend_prediction = self.trend_coefficient * future_point + self.intercept
        
        # Apply seasonal adjustment
        future_time = time.time() + (time_horizon * 60)
        future_hour = datetime.fromtimestamp(future_time).hour
        
        seasonal_adjustment = 1.0
        if 'hourly' in self.seasonal_patterns and len(self.seasonal_patterns['hourly']) > future_hour:
            current_hour = datetime.now().hour
            if current_hour < len(self.seasonal_patterns['hourly']):
                current_seasonal = self.seasonal_patterns['hourly'][current_hour]
                future_seasonal = self.seasonal_patterns['hourly'][future_hour]
                
                if current_seasonal > 0:
                    seasonal_adjustment = future_seasonal / current_seasonal
        
        predicted_load = max(0, trend_prediction * seasonal_adjustment)
        
        # Calculate confidence based on recent variance
        recent_loads = list(self.load_history)[-10:]
        variance = statistics.variance(recent_loads) if len(recent_loads) > 1 else 0
        confidence = max(0.1, 1.0 - (variance / (statistics.mean(recent_loads) + 1)))
        
        return PredictionData(
            timestamp=time.time(),
            predicted_load=predicted_load,
            confidence=min(0.95, confidence),
            time_horizon=time_horizon,
            factors={
                'trend': trend_prediction,
                'seasonal_adjustment': seasonal_adjustment,
                'variance': variance
            }
        )

class ResourceAnalyzer:
    """Analyzes resource utilization and capacity"""
    
    def __init__(self):
        self.resource_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.resource_limits: Dict[str, Dict[str, float]] = {}
        
    def add_resource_metrics(self, deployment_name: str, metrics: ResourceMetrics):
        """Add resource metrics for a deployment"""
        self.resource_history[deployment_name].append(metrics)
    
    def analyze_resource_pressure(self, deployment_name: str) -> Dict[str, float]:
        """Analyze resource pressure for a deployment"""
        if deployment_name not in self.resource_history:
            return {}
        
        recent_metrics = list(self.resource_history[deployment_name])[-5:]
        if not recent_metrics:
            return {}
        
        # Calculate average utilization
        avg_cpu = statistics.mean(m.cpu_usage for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_usage for m in recent_metrics)
        avg_network = statistics.mean(m.network_io for m in recent_metrics)
        avg_disk = statistics.mean(m.disk_io for m in recent_metrics)
        
        return {
            'cpu_pressure': avg_cpu / 100.0,
            'memory_pressure': avg_memory / 100.0,
            'network_pressure': min(1.0, avg_network / 1000.0),  # Normalize to 0-1
            'disk_pressure': min(1.0, avg_disk / 1000.0),
            'overall_pressure': max(avg_cpu / 100.0, avg_memory / 100.0)
        }
    
    def get_scaling_recommendation(self, deployment_name: str, current_replicas: int) -> Dict[str, Any]:
        """Get resource-based scaling recommendation"""
        pressure = self.analyze_resource_pressure(deployment_name)
        
        if not pressure:
            return {'action': 'none', 'reason': 'no_data'}
        
        overall_pressure = pressure.get('overall_pressure', 0)
        
        if overall_pressure > 0.8:  # High pressure
            return {
                'action': 'scale_up',
                'reason': 'high_resource_pressure',
                'recommended_replicas': min(current_replicas + 2, 10),
                'pressure_score': overall_pressure
            }
        elif overall_pressure < 0.3 and current_replicas > 1:  # Low pressure
            return {
                'action': 'scale_down',
                'reason': 'low_resource_utilization',
                'recommended_replicas': max(current_replicas - 1, 1),
                'pressure_score': overall_pressure
            }
        else:
            return {
                'action': 'none',
                'reason': 'optimal_utilization',
                'pressure_score': overall_pressure
            }

class CostOptimizer:
    """Optimizes scaling decisions based on cost considerations"""
    
    def __init__(self):
        self.cost_per_replica_hour = 0.1  # Default cost per replica per hour
        self.performance_cost_ratio = 1.0  # How much to weight performance vs cost
        
    def calculate_cost_impact(self, current_replicas: int, target_replicas: int) -> float:
        """Calculate the cost impact of a scaling decision"""
        replica_delta = target_replicas - current_replicas
        hourly_cost_change = replica_delta * self.cost_per_replica_hour
        
        # Assume scaling decisions are evaluated for 1 hour impact
        return hourly_cost_change
    
    def optimize_scaling_decision(self, scaling_decision: ScalingDecision, 
                                 performance_benefit: float) -> ScalingDecision:
        """Optimize a scaling decision considering cost vs performance"""
        cost_impact = self.calculate_cost_impact(
            scaling_decision.current_replicas, 
            scaling_decision.target_replicas
        )
        
        # Calculate cost-benefit ratio
        if cost_impact > 0:  # Scaling up
            cost_benefit_ratio = performance_benefit / cost_impact
        else:  # Scaling down
            cost_benefit_ratio = abs(cost_impact) / max(performance_benefit, 0.1)
        
        # Adjust decision based on cost-benefit analysis
        if cost_benefit_ratio < 2.0:  # Low benefit for cost
            if scaling_decision.target_replicas > scaling_decision.current_replicas:
                # Reduce scale-up
                scaling_decision.target_replicas = min(
                    scaling_decision.target_replicas,
                    scaling_decision.current_replicas + 1
                )
            elif scaling_decision.target_replicas < scaling_decision.current_replicas:
                # Increase scale-down
                scaling_decision.target_replicas = max(
                    scaling_decision.target_replicas,
                    scaling_decision.current_replicas - 1
                )
        
        scaling_decision.cost_impact = cost_impact
        return scaling_decision

class PredictiveAutoScaler:
    """Enhanced autoscaler with predictive capabilities"""
    
    def __init__(self, agent_registry: AgentRegistry, task_dispatcher: TaskDispatcher,
                 performance_monitor: PerformanceMonitor, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.agent_registry = agent_registry
        self.task_dispatcher = task_dispatcher
        self.performance_monitor = performance_monitor
        self.strategy = strategy
        
        # Components
        self.predictors: Dict[str, WorkloadPredictor] = {}
        self.resource_analyzer = ResourceAnalyzer()
        self.cost_optimizer = CostOptimizer()
        
        # Scaling history
        self.scaling_history: deque = deque(maxlen=1000)
        self.last_scaling_time: Dict[str, float] = {}
        
        # Configuration
        self.prediction_horizon = 15  # minutes
        self.min_scaling_interval = 300  # 5 minutes between scaling actions
        self.poll_interval = 60  # 1 minute
        self.max_replicas = 10
        self.min_replicas = 1
        
        # Thresholds
        self.scale_up_threshold = 0.8  # 80% utilization
        self.scale_down_threshold = 0.3  # 30% utilization
        self.prediction_confidence_threshold = 0.6
        
        # State
        self._running = False
        self._scaling_task: Optional[asyncio.Task] = None
        
        # Fallback to basic autoscaler if Kubernetes is not available
        self.fallback_autoscaler = AutoScaler() if k8s_apps_v1 is None else None
    
    async def start(self):
        """Start the predictive autoscaler"""
        if self._running:
            return
        
        if k8s_apps_v1 is None:
            logger.warning("Kubernetes client not available, falling back to basic autoscaler")
            if self.fallback_autoscaler:
                self.fallback_autoscaler.start()
            return
        
        self._running = True
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Predictive autoscaler started")
    
    async def stop(self):
        """Stop the predictive autoscaler"""
        self._running = False
        
        if self._scaling_task:
            self._scaling_task.cancel()
        
        if self.fallback_autoscaler:
            self.fallback_autoscaler.stop()
        
        logger.info("Predictive autoscaler stopped")
    
    async def _scaling_loop(self):
        """Main scaling loop"""
        while self._running:
            try:
                await self._evaluate_scaling_decisions()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in predictive autoscaler loop: {e}", exc_info=True)
                await asyncio.sleep(30)
    
    async def _evaluate_scaling_decisions(self):
        """Evaluate and make scaling decisions"""
        # Update predictors with current data
        await self._update_predictors()
        
        # Get current deployments
        deployments = await self._get_agent_deployments()
        
        for deployment_name, current_replicas in deployments.items():
            # Check if we should skip scaling this deployment
            if self._should_skip_scaling(deployment_name):
                continue
            
            # Make scaling decision
            decision = await self._make_scaling_decision(deployment_name, current_replicas)
            
            if decision and decision.target_replicas != current_replicas:
                await self._execute_scaling_decision(decision)
    
    async def _update_predictors(self):
        """Update workload predictors with current data"""
        # Get current metrics
        dashboard_data = self.performance_monitor.get_dashboard_data()
        current_metrics = dashboard_data.get('current_metrics', {})
        
        # Update predictors for each personality
        for agent in self.agent_registry.get_all_agents():
            personality = agent.personality
            
            if personality not in self.predictors:
                self.predictors[personality] = WorkloadPredictor()
            
            # Get current load for this personality
            current_load = agent.metrics.current_load
            self.predictors[personality].add_data_point(time.time(), current_load)
    
    async def _get_agent_deployments(self) -> Dict[str, int]:
        """Get current agent deployments and their replica counts"""
        deployments = {}
        
        try:
            # Get all deployments in the namespace
            k8s_deployments = k8s_apps_v1.list_namespaced_deployment(namespace="default")
            
            for deployment in k8s_deployments.items:
                if deployment.metadata.name.startswith("agentq-"):
                    deployments[deployment.metadata.name] = deployment.spec.replicas
        
        except Exception as e:
            logger.error(f"Error getting deployments: {e}")
        
        return deployments
    
    def _should_skip_scaling(self, deployment_name: str) -> bool:
        """Check if we should skip scaling for a deployment"""
        # Check minimum interval between scaling actions
        last_scaling = self.last_scaling_time.get(deployment_name, 0)
        if time.time() - last_scaling < self.min_scaling_interval:
            return True
        
        return False
    
    async def _make_scaling_decision(self, deployment_name: str, current_replicas: int) -> Optional[ScalingDecision]:
        """Make a scaling decision for a deployment"""
        personality = deployment_name.replace("agentq-", "")
        
        # Get current metrics
        agent_metrics = self._get_agent_metrics(personality)
        
        # Get prediction
        prediction = None
        if personality in self.predictors:
            prediction = self.predictors[personality].predict(self.prediction_horizon)
        
        # Get resource analysis
        resource_analysis = self.resource_analyzer.get_scaling_recommendation(
            deployment_name, current_replicas
        )
        
        # Make decision based on strategy
        if self.strategy == ScalingStrategy.REACTIVE:
            return self._make_reactive_decision(deployment_name, current_replicas, agent_metrics)
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            return self._make_predictive_decision(deployment_name, current_replicas, prediction)
        elif self.strategy == ScalingStrategy.HYBRID:
            return self._make_hybrid_decision(deployment_name, current_replicas, agent_metrics, prediction, resource_analysis)
        elif self.strategy == ScalingStrategy.COST_OPTIMIZED:
            return self._make_cost_optimized_decision(deployment_name, current_replicas, agent_metrics, prediction)
        elif self.strategy == ScalingStrategy.PERFORMANCE_OPTIMIZED:
            return self._make_performance_optimized_decision(deployment_name, current_replicas, agent_metrics, prediction)
        
        return None
    
    def _get_agent_metrics(self, personality: str) -> Dict[str, float]:
        """Get current metrics for an agent personality"""
        metrics = {}
        
        # Get agents with this personality
        agents = [a for a in self.agent_registry.get_all_agents() if a.personality == personality]
        
        if agents:
            # Aggregate metrics
            total_load = sum(a.metrics.current_load for a in agents)
            total_capacity = sum(a.capabilities.max_concurrent_tasks for a in agents)
            avg_success_rate = sum(a.metrics.success_rate for a in agents) / len(agents)
            avg_response_time = sum(a.metrics.response_time_p95 for a in agents) / len(agents)
            
            metrics = {
                'total_load': total_load,
                'total_capacity': total_capacity,
                'utilization': total_load / max(total_capacity, 1),
                'avg_success_rate': avg_success_rate,
                'avg_response_time': avg_response_time,
                'agent_count': len(agents)
            }
        
        return metrics
    
    def _make_reactive_decision(self, deployment_name: str, current_replicas: int, 
                               agent_metrics: Dict[str, float]) -> Optional[ScalingDecision]:
        """Make reactive scaling decision based on current metrics"""
        utilization = agent_metrics.get('utilization', 0)
        
        if utilization > self.scale_up_threshold and current_replicas < self.max_replicas:
            return ScalingDecision(
                deployment_name=deployment_name,
                current_replicas=current_replicas,
                target_replicas=min(current_replicas + 1, self.max_replicas),
                trigger=ScalingTrigger.PERFORMANCE_THRESHOLD,
                confidence=0.9,
                reasoning=f"High utilization: {utilization:.2f}"
            )
        elif utilization < self.scale_down_threshold and current_replicas > self.min_replicas:
            return ScalingDecision(
                deployment_name=deployment_name,
                current_replicas=current_replicas,
                target_replicas=max(current_replicas - 1, self.min_replicas),
                trigger=ScalingTrigger.PERFORMANCE_THRESHOLD,
                confidence=0.8,
                reasoning=f"Low utilization: {utilization:.2f}"
            )
        
        return None
    
    def _make_predictive_decision(self, deployment_name: str, current_replicas: int,
                                 prediction: Optional[PredictionData]) -> Optional[ScalingDecision]:
        """Make predictive scaling decision"""
        if not prediction or prediction.confidence < self.prediction_confidence_threshold:
            return None
        
        # Calculate required replicas based on prediction
        predicted_load = prediction.predicted_load
        required_replicas = max(1, int(predicted_load / 5) + 1)  # Assuming 5 tasks per replica
        
        if required_replicas > current_replicas and current_replicas < self.max_replicas:
            return ScalingDecision(
                deployment_name=deployment_name,
                current_replicas=current_replicas,
                target_replicas=min(required_replicas, self.max_replicas),
                trigger=ScalingTrigger.PREDICTION,
                confidence=prediction.confidence,
                reasoning=f"Predicted load: {predicted_load:.2f}"
            )
        elif required_replicas < current_replicas and current_replicas > self.min_replicas:
            return ScalingDecision(
                deployment_name=deployment_name,
                current_replicas=current_replicas,
                target_replicas=max(required_replicas, self.min_replicas),
                trigger=ScalingTrigger.PREDICTION,
                confidence=prediction.confidence,
                reasoning=f"Predicted load decrease: {predicted_load:.2f}"
            )
        
        return None
    
    def _make_hybrid_decision(self, deployment_name: str, current_replicas: int,
                             agent_metrics: Dict[str, float], prediction: Optional[PredictionData],
                             resource_analysis: Dict[str, Any]) -> Optional[ScalingDecision]:
        """Make hybrid scaling decision combining multiple factors"""
        # Start with reactive decision
        reactive_decision = self._make_reactive_decision(deployment_name, current_replicas, agent_metrics)
        
        # Consider predictive factors
        predictive_decision = self._make_predictive_decision(deployment_name, current_replicas, prediction)
        
        # Consider resource pressure
        resource_action = resource_analysis.get('action', 'none')
        
        # Combine decisions
        target_replicas = current_replicas
        trigger = ScalingTrigger.MANUAL
        confidence = 0.5
        reasoning = "Hybrid decision: "
        
        votes = []
        
        # Reactive vote
        if reactive_decision:
            votes.append(reactive_decision.target_replicas)
            reasoning += f"reactive({reactive_decision.target_replicas}), "
        
        # Predictive vote
        if predictive_decision:
            votes.append(predictive_decision.target_replicas)
            reasoning += f"predictive({predictive_decision.target_replicas}), "
        
        # Resource vote
        if resource_action == 'scale_up':
            votes.append(resource_analysis.get('recommended_replicas', current_replicas + 1))
            reasoning += f"resource(scale_up), "
        elif resource_action == 'scale_down':
            votes.append(resource_analysis.get('recommended_replicas', current_replicas - 1))
            reasoning += f"resource(scale_down), "
        
        # Make final decision
        if votes:
            # Use majority vote or average
            if len(votes) == 1:
                target_replicas = votes[0]
                confidence = 0.7
            else:
                # Take average and round
                target_replicas = max(self.min_replicas, min(self.max_replicas, 
                                                           round(sum(votes) / len(votes))))
                confidence = 0.8
            
            trigger = ScalingTrigger.PERFORMANCE_THRESHOLD
        
        if target_replicas != current_replicas:
            return ScalingDecision(
                deployment_name=deployment_name,
                current_replicas=current_replicas,
                target_replicas=target_replicas,
                trigger=trigger,
                confidence=confidence,
                reasoning=reasoning.rstrip(', ')
            )
        
        return None
    
    def _make_cost_optimized_decision(self, deployment_name: str, current_replicas: int,
                                     agent_metrics: Dict[str, float], prediction: Optional[PredictionData]) -> Optional[ScalingDecision]:
        """Make cost-optimized scaling decision"""
        # Start with hybrid decision
        hybrid_decision = self._make_hybrid_decision(deployment_name, current_replicas, agent_metrics, prediction, {})
        
        if hybrid_decision:
            # Optimize for cost
            performance_benefit = abs(hybrid_decision.target_replicas - current_replicas) * 0.2
            optimized_decision = self.cost_optimizer.optimize_scaling_decision(hybrid_decision, performance_benefit)
            optimized_decision.trigger = ScalingTrigger.COST_OPTIMIZATION
            return optimized_decision
        
        return None
    
    def _make_performance_optimized_decision(self, deployment_name: str, current_replicas: int,
                                           agent_metrics: Dict[str, float], prediction: Optional[PredictionData]) -> Optional[ScalingDecision]:
        """Make performance-optimized scaling decision"""
        utilization = agent_metrics.get('utilization', 0)
        avg_response_time = agent_metrics.get('avg_response_time', 0)
        
        # More aggressive scaling for performance
        if utilization > 0.6 or avg_response_time > 2000:  # 2 seconds
            return ScalingDecision(
                deployment_name=deployment_name,
                current_replicas=current_replicas,
                target_replicas=min(current_replicas + 2, self.max_replicas),
                trigger=ScalingTrigger.PERFORMANCE_THRESHOLD,
                confidence=0.9,
                reasoning=f"Performance optimization: util={utilization:.2f}, response_time={avg_response_time:.0f}ms"
            )
        elif utilization < 0.2 and avg_response_time < 500:  # 500ms
            return ScalingDecision(
                deployment_name=deployment_name,
                current_replicas=current_replicas,
                target_replicas=max(current_replicas - 1, self.min_replicas),
                trigger=ScalingTrigger.PERFORMANCE_THRESHOLD,
                confidence=0.7,
                reasoning=f"Performance optimization: low util={utilization:.2f}"
            )
        
        return None
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision"""
        try:
            # Update Kubernetes deployment
            patch_body = {"spec": {"replicas": decision.target_replicas}}
            k8s_apps_v1.patch_namespaced_deployment_scale(
                name=decision.deployment_name,
                namespace="default",
                body=patch_body
            )
            
            # Record the scaling action
            self.scaling_history.append(decision)
            self.last_scaling_time[decision.deployment_name] = time.time()
            
            logger.info(f"Scaled {decision.deployment_name} from {decision.current_replicas} to "
                       f"{decision.target_replicas} replicas. Reason: {decision.reasoning}")
            
        except Exception as e:
            logger.error(f"Failed to execute scaling decision for {decision.deployment_name}: {e}")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics"""
        recent_decisions = [d for d in self.scaling_history if time.time() - d.timestamp < 3600]
        
        return {
            'total_scaling_actions': len(self.scaling_history),
            'recent_scaling_actions': len(recent_decisions),
            'strategy': self.strategy.value,
            'predictors': len(self.predictors),
            'avg_confidence': statistics.mean([d.confidence for d in recent_decisions]) if recent_decisions else 0,
            'scale_up_actions': sum(1 for d in recent_decisions if d.target_replicas > d.current_replicas),
            'scale_down_actions': sum(1 for d in recent_decisions if d.target_replicas < d.current_replicas),
            'cost_savings': sum(d.cost_impact for d in recent_decisions if d.cost_impact < 0),
            'performance_improvements': sum(1 for d in recent_decisions 
                                          if d.trigger == ScalingTrigger.PERFORMANCE_THRESHOLD)
        }
    
    def set_strategy(self, strategy: ScalingStrategy):
        """Change the scaling strategy"""
        self.strategy = strategy
        logger.info(f"Scaling strategy changed to: {strategy.value}")
    
    def get_predictions(self) -> Dict[str, PredictionData]:
        """Get current predictions for all personalities"""
        predictions = {}
        for personality, predictor in self.predictors.items():
            predictions[personality] = predictor.predict(self.prediction_horizon)
        return predictions

# Singleton instance
predictive_autoscaler: Optional[PredictiveAutoScaler] = None 