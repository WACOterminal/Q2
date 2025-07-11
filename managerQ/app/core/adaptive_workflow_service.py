"""
Adaptive Workflow Orchestration Service

This service provides adaptive workflow orchestration capabilities for the Q Platform:
- Self-optimizing workflow execution
- Performance-based workflow adaptation
- Machine learning-driven optimization
- Dynamic resource allocation
- Workflow pattern recognition
- Execution history analysis
- Intelligent retry mechanisms
- Load balancing and scaling
- Cost optimization strategies
- Real-time performance monitoring
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import statistics
import math
import heapq
from functools import wraps
import time

# ML libraries for optimization
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    sklearn_available = True
except ImportError:
    sklearn_available = False
    logging.warning("Scikit-learn not available - ML optimization will be limited")

try:
    import networkx as nx
    networkx_available = True
except ImportError:
    networkx_available = False
    logging.warning("NetworkX not available - graph analysis will be limited")

try:
    import pandas as pd
    pandas_available = True
except ImportError:
    pandas_available = False
    logging.warning("Pandas not available - data analysis will be limited")

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)

class WorkflowPriority(Enum):
    """Workflow priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class ExecutionStatus(Enum):
    """Execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    OPTIMIZING = "optimizing"

class OptimizationStrategy(Enum):
    """Optimization strategies"""
    PERFORMANCE = "performance"
    COST = "cost"
    BALANCED = "balanced"
    CUSTOM = "custom"

class AdaptationTrigger(Enum):
    """Adaptation triggers"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_CONSTRAINT = "resource_constraint"
    FAILURE_PATTERN = "failure_pattern"
    COST_THRESHOLD = "cost_threshold"
    MANUAL = "manual"

class ResourceType(Enum):
    """Resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    CUSTOM = "custom"

@dataclass
class WorkflowStep:
    """Workflow step representation"""
    step_id: str
    name: str
    function: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    resources: Dict[ResourceType, float]
    timeout: int
    retry_count: int
    retry_delay: int
    priority: WorkflowPriority
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class WorkflowDefinition:
    """Workflow definition"""
    workflow_id: str
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    triggers: List[str]
    schedule: Optional[str] = None
    max_concurrent: int = 1
    timeout: int = 3600
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str
    workflow_id: str
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time: float = 0.0
    resource_usage: Dict[ResourceType, float] = None
    cost: float = 0.0
    step_results: Dict[str, Any] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    adaptation_applied: bool = False
    
    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {}
        if self.step_results is None:
            self.step_results = {}

@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    execution_id: str
    workflow_id: str
    timestamp: datetime
    execution_time: float
    resource_efficiency: float
    cost_efficiency: float
    success_rate: float
    throughput: float
    latency: float
    error_rate: float
    adaptation_score: float

@dataclass
class OptimizationResult:
    """Optimization result"""
    optimization_id: str
    workflow_id: str
    strategy: OptimizationStrategy
    improvements: Dict[str, float]
    new_configuration: Dict[str, Any]
    confidence: float
    created_at: datetime
    applied: bool = False

@dataclass
class ResourceAllocation:
    """Resource allocation"""
    workflow_id: str
    execution_id: str
    resource_type: ResourceType
    allocated: float
    used: float
    efficiency: float
    cost: float
    timestamp: datetime

@dataclass
class AdaptationEvent:
    """Adaptation event"""
    event_id: str
    workflow_id: str
    trigger: AdaptationTrigger
    description: str
    action_taken: str
    impact: Dict[str, float]
    timestamp: datetime

class AdaptiveWorkflowService:
    """
    Comprehensive Adaptive Workflow Orchestration Service
    """
    
    def __init__(self, storage_path: str = "adaptive_workflows"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Workflow management
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.execution_queue: deque = deque()
        self.running_executions: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.resource_allocations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.adaptation_events: deque = deque(maxlen=1000)
        
        # Optimization models
        self.optimization_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.optimization_results: Dict[str, OptimizationResult] = {}
        
        # Pattern recognition
        self.execution_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Resource management
        self.resource_pools: Dict[ResourceType, Dict[str, Any]] = {}
        self.resource_usage: Dict[ResourceType, float] = {}
        
        # Configuration
        self.config = {
            "max_concurrent_executions": 10,
            "optimization_interval": 3600,
            "adaptation_threshold": 0.8,
            "failure_retry_limit": 3,
            "resource_scaling_factor": 1.5,
            "cost_optimization_weight": 0.3,
            "performance_optimization_weight": 0.7,
            "enable_auto_adaptation": True,
            "enable_ml_optimization": True,
            "min_samples_for_training": 50,
            "optimization_confidence_threshold": 0.7,
            "resource_buffer_percentage": 0.2
        }
        
        # Performance metrics
        self.global_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_cost": 0.0,
            "cost_savings": 0.0,
            "performance_improvements": 0.0,
            "adaptation_success_rate": 0.0,
            "resource_efficiency": 0.0
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Service integrations
        self.vault_client = VaultClient()
        
        # Workflow execution graph
        self.workflow_graph: Dict[str, Any] = {}
        
        logger.info("Adaptive Workflow Service initialized")
    
    async def initialize(self):
        """Initialize the adaptive workflow service"""
        logger.info("Initializing Adaptive Workflow Service")
        
        # Load existing data
        await self._load_workflow_data()
        
        # Initialize resource pools
        await self._initialize_resource_pools()
        
        # Initialize ML models
        await self._initialize_ml_models()
        
        # Start background tasks
        await self._start_background_tasks()
        
        logger.info("Adaptive Workflow Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the adaptive workflow service"""
        logger.info("Shutting down Adaptive Workflow Service")
        
        # Cancel running executions
        for execution_id, task in self.running_executions.items():
            task.cancel()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Save data
        await self._save_workflow_data()
        
        logger.info("Adaptive Workflow Service shutdown complete")
    
    # ===== WORKFLOW MANAGEMENT =====
    
    async def register_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Register a new workflow"""
        try:
            # Validate workflow
            if not await self._validate_workflow(workflow):
                return False
            
            # Build workflow graph
            workflow_graph = await self._build_workflow_graph(workflow)
            self.workflow_graph[workflow.workflow_id] = workflow_graph
            
            # Store workflow
            self.workflows[workflow.workflow_id] = workflow
            
            # Initialize optimization model
            await self._initialize_workflow_optimization(workflow.workflow_id)
            
            logger.info(f"Workflow registered: {workflow.workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering workflow: {e}")
            return False
    
    async def execute_workflow(self, workflow_id: str, parameters: Dict[str, Any] = None) -> str:
        """Execute a workflow"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            workflow = self.workflows[workflow_id]
            
            # Create execution instance
            execution_id = f"exec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{workflow_id}"
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=workflow_id,
                status=ExecutionStatus.PENDING,
                started_at=datetime.utcnow()
            )
            
            # Apply parameters if provided
            if parameters:
                execution.step_results["parameters"] = parameters
            
            # Store execution
            self.executions[execution_id] = execution
            
            # Add to execution queue
            self.execution_queue.append(execution_id)
            
            logger.info(f"Workflow execution queued: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            raise
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution"""
        try:
            if execution_id not in self.executions:
                return False
            
            execution = self.executions[execution_id]
            
            # Cancel running task
            if execution_id in self.running_executions:
                task = self.running_executions[execution_id]
                task.cancel()
                del self.running_executions[execution_id]
            
            # Update status
            execution.status = ExecutionStatus.CANCELLED
            execution.completed_at = datetime.utcnow()
            
            logger.info(f"Execution cancelled: {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling execution: {e}")
            return False
    
    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution status"""
        return self.executions.get(execution_id)
    
    async def list_executions(self, workflow_id: str = None, status: ExecutionStatus = None) -> List[WorkflowExecution]:
        """List executions with optional filtering"""
        executions = list(self.executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        return executions
    
    # ===== OPTIMIZATION ENGINE =====
    
    async def optimize_workflow(self, workflow_id: str, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> OptimizationResult:
        """Optimize a workflow"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            workflow = self.workflows[workflow_id]
            
            # Collect historical data
            historical_data = await self._collect_optimization_data(workflow_id)
            
            if len(historical_data) < self.config["min_samples_for_training"]:
                raise ValueError("Insufficient historical data for optimization")
            
            # Apply optimization strategy
            optimization_result = await self._apply_optimization_strategy(workflow_id, strategy, historical_data)
            
            # Store optimization result
            self.optimization_results[workflow_id] = optimization_result
            
            logger.info(f"Workflow optimized: {workflow_id}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error optimizing workflow: {e}")
            raise
    
    async def apply_optimization(self, workflow_id: str, optimization_id: str) -> bool:
        """Apply optimization to a workflow"""
        try:
            if workflow_id not in self.workflows:
                return False
            
            if optimization_id not in self.optimization_results:
                return False
            
            optimization = self.optimization_results[optimization_id]
            workflow = self.workflows[workflow_id]
            
            # Apply configuration changes
            await self._apply_optimization_configuration(workflow, optimization.new_configuration)
            
            # Mark as applied
            optimization.applied = True
            
            # Record adaptation event
            adaptation_event = AdaptationEvent(
                event_id=f"adapt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                workflow_id=workflow_id,
                trigger=AdaptationTrigger.MANUAL,
                description=f"Applied optimization {optimization_id}",
                action_taken="Configuration updated",
                impact=optimization.improvements,
                timestamp=datetime.utcnow()
            )
            
            self.adaptation_events.append(adaptation_event)
            
            logger.info(f"Optimization applied: {optimization_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying optimization: {e}")
            return False
    
    async def auto_adapt_workflow(self, workflow_id: str, trigger: AdaptationTrigger) -> bool:
        """Automatically adapt workflow based on trigger"""
        try:
            if not self.config["enable_auto_adaptation"]:
                return False
            
            if workflow_id not in self.workflows:
                return False
            
            workflow = self.workflows[workflow_id]
            
            # Analyze trigger and determine adaptation
            adaptation_strategy = await self._determine_adaptation_strategy(workflow_id, trigger)
            
            if not adaptation_strategy:
                return False
            
            # Apply adaptation
            success = await self._apply_adaptation_strategy(workflow_id, adaptation_strategy)
            
            if success:
                # Record adaptation event
                adaptation_event = AdaptationEvent(
                    event_id=f"adapt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    workflow_id=workflow_id,
                    trigger=trigger,
                    description=f"Auto-adaptation triggered by {trigger.value}",
                    action_taken=adaptation_strategy["action"],
                    impact=adaptation_strategy.get("impact", {}),
                    timestamp=datetime.utcnow()
                )
                
                self.adaptation_events.append(adaptation_event)
                
                logger.info(f"Auto-adaptation applied: {workflow_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error auto-adapting workflow: {e}")
            return False
    
    # ===== RESOURCE MANAGEMENT =====
    
    async def allocate_resources(self, execution_id: str) -> Dict[ResourceType, float]:
        """Allocate resources for execution"""
        try:
            execution = self.executions[execution_id]
            workflow = self.workflows[execution.workflow_id]
            
            # Calculate resource requirements
            resource_requirements = await self._calculate_resource_requirements(workflow)
            
            # Allocate resources
            allocation = {}
            for resource_type, requirement in resource_requirements.items():
                allocated = await self._allocate_resource(resource_type, requirement, execution_id)
                allocation[resource_type] = allocated
            
            # Record allocation
            for resource_type, allocated in allocation.items():
                resource_allocation = ResourceAllocation(
                    workflow_id=workflow.workflow_id,
                    execution_id=execution_id,
                    resource_type=resource_type,
                    allocated=allocated,
                    used=0.0,
                    efficiency=0.0,
                    cost=0.0,
                    timestamp=datetime.utcnow()
                )
                
                self.resource_allocations[execution_id].append(resource_allocation)
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error allocating resources: {e}")
            return {}
    
    async def deallocate_resources(self, execution_id: str):
        """Deallocate resources after execution"""
        try:
            if execution_id not in self.resource_allocations:
                return
            
            for allocation in self.resource_allocations[execution_id]:
                await self._deallocate_resource(allocation.resource_type, allocation.allocated, execution_id)
            
            logger.info(f"Resources deallocated for execution: {execution_id}")
            
        except Exception as e:
            logger.error(f"Error deallocating resources: {e}")
    
    async def scale_resources(self, resource_type: ResourceType, scale_factor: float) -> bool:
        """Scale resources"""
        try:
            if resource_type not in self.resource_pools:
                return False
            
            pool = self.resource_pools[resource_type]
            current_capacity = pool.get("capacity", 0)
            new_capacity = current_capacity * scale_factor
            
            # Update resource pool
            pool["capacity"] = new_capacity
            
            logger.info(f"Resource {resource_type.value} scaled by {scale_factor}")
            return True
            
        except Exception as e:
            logger.error(f"Error scaling resources: {e}")
            return False
    
    # ===== EXECUTION ENGINE =====
    
    async def _execute_workflow_instance(self, execution_id: str):
        """Execute a workflow instance"""
        try:
            execution = self.executions[execution_id]
            workflow = self.workflows[execution.workflow_id]
            
            # Update status
            execution.status = ExecutionStatus.RUNNING
            execution.started_at = datetime.utcnow()
            
            # Allocate resources
            resource_allocation = await self.allocate_resources(execution_id)
            execution.resource_usage = resource_allocation
            
            # Execute workflow steps
            success = await self._execute_workflow_steps(execution, workflow)
            
            # Update final status
            execution.completed_at = datetime.utcnow()
            execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            if success:
                execution.status = ExecutionStatus.COMPLETED
                self.global_metrics["successful_executions"] += 1
            else:
                execution.status = ExecutionStatus.FAILED
                self.global_metrics["failed_executions"] += 1
            
            # Calculate cost
            execution.cost = await self._calculate_execution_cost(execution)
            
            # Record performance metrics
            await self._record_performance_metrics(execution)
            
            # Deallocate resources
            await self.deallocate_resources(execution_id)
            
            # Check for adaptation triggers
            await self._check_adaptation_triggers(execution)
            
            # Update global metrics
            self.global_metrics["total_executions"] += 1
            
            logger.info(f"Workflow execution completed: {execution_id}")
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            
            # Update execution status
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                execution.status = ExecutionStatus.FAILED
                execution.error_message = str(e)
                execution.completed_at = datetime.utcnow()
                
                # Deallocate resources
                await self.deallocate_resources(execution_id)
        
        finally:
            # Remove from running executions
            if execution_id in self.running_executions:
                del self.running_executions[execution_id]
    
    async def _execute_workflow_steps(self, execution: WorkflowExecution, workflow: WorkflowDefinition) -> bool:
        """Execute workflow steps"""
        try:
            # Build execution graph
            execution_graph = self.workflow_graph[workflow.workflow_id]
            
            # Execute steps in topological order
            executed_steps = set()
            
            while len(executed_steps) < len(workflow.steps):
                # Find steps ready for execution
                ready_steps = []
                for step in workflow.steps:
                    if step.step_id not in executed_steps:
                        if all(dep in executed_steps for dep in step.dependencies):
                            ready_steps.append(step)
                
                if not ready_steps:
                    logger.error("Workflow has circular dependencies or unreachable steps")
                    return False
                
                # Execute ready steps
                for step in ready_steps:
                    step_success = await self._execute_step(execution, step)
                    if step_success:
                        executed_steps.add(step.step_id)
                    else:
                        # Handle step failure
                        if step.retry_count > 0:
                            await self._retry_step(execution, step)
                        else:
                            logger.error(f"Step failed: {step.step_id}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing workflow steps: {e}")
            return False
    
    async def _execute_step(self, execution: WorkflowExecution, step: WorkflowStep) -> bool:
        """Execute a single step"""
        try:
            logger.info(f"Executing step: {step.step_id}")
            
            # Simulate step execution
            start_time = time.time()
            
            # This would call the actual step function
            # For now, simulate execution
            await asyncio.sleep(0.1)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Record step result
            execution.step_results[step.step_id] = {
                "status": "completed",
                "execution_time": execution_time,
                "result": f"Step {step.step_id} completed successfully"
            }
            
            logger.info(f"Step completed: {step.step_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing step {step.step_id}: {e}")
            
            # Record step failure
            execution.step_results[step.step_id] = {
                "status": "failed",
                "error": str(e)
            }
            
            return False
    
    async def _retry_step(self, execution: WorkflowExecution, step: WorkflowStep):
        """Retry a failed step"""
        try:
            logger.info(f"Retrying step: {step.step_id}")
            
            # Wait before retry
            await asyncio.sleep(step.retry_delay)
            
            # Retry step
            return await self._execute_step(execution, step)
            
        except Exception as e:
            logger.error(f"Error retrying step {step.step_id}: {e}")
            return False
    
    # ===== BACKGROUND TASKS =====
    
    async def _start_background_tasks(self):
        """Start background tasks"""
        tasks = [
            self._execution_scheduler(),
            self._optimization_monitor(),
            self._resource_monitor(),
            self._adaptation_monitor(),
            self._metrics_collector()
        ]
        
        for task_func in tasks:
            task = asyncio.create_task(task_func)
            self.background_tasks.add(task)
    
    async def _execution_scheduler(self):
        """Schedule workflow executions"""
        while True:
            try:
                # Check queue and running executions
                if (self.execution_queue and 
                    len(self.running_executions) < self.config["max_concurrent_executions"]):
                    
                    execution_id = self.execution_queue.popleft()
                    
                    # Start execution
                    task = asyncio.create_task(self._execute_workflow_instance(execution_id))
                    self.running_executions[execution_id] = task
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in execution scheduler: {e}")
                await asyncio.sleep(5)
    
    async def _optimization_monitor(self):
        """Monitor and trigger optimizations"""
        while True:
            try:
                if self.config["enable_ml_optimization"]:
                    # Check each workflow for optimization opportunities
                    for workflow_id in self.workflows.keys():
                        await self._check_optimization_opportunity(workflow_id)
                
                await asyncio.sleep(self.config["optimization_interval"])
                
            except Exception as e:
                logger.error(f"Error in optimization monitor: {e}")
                await asyncio.sleep(300)
    
    async def _resource_monitor(self):
        """Monitor resource usage"""
        while True:
            try:
                # Update resource usage
                await self._update_resource_usage()
                
                # Check for resource constraints
                await self._check_resource_constraints()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                await asyncio.sleep(60)
    
    async def _adaptation_monitor(self):
        """Monitor for adaptation triggers"""
        while True:
            try:
                # Check for performance degradation
                await self._check_performance_degradation()
                
                # Check for failure patterns
                await self._check_failure_patterns()
                
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in adaptation monitor: {e}")
                await asyncio.sleep(300)
    
    async def _metrics_collector(self):
        """Collect and aggregate metrics"""
        while True:
            try:
                # Update global metrics
                await self._update_global_metrics()
                
                # Analyze patterns
                await self._analyze_execution_patterns()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60)
    
    # ===== HELPER METHODS =====
    
    async def _validate_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Validate workflow definition"""
        try:
            # Check for cycles in dependencies
            if networkx_available:
                graph = nx.DiGraph()
                for step in workflow.steps:
                    graph.add_node(step.step_id)
                    for dep in step.dependencies:
                        graph.add_edge(dep, step.step_id)
                
                if not nx.is_directed_acyclic_graph(graph):
                    logger.error("Workflow has circular dependencies")
                    return False
            
            # Check for unreferenced dependencies
            step_ids = {step.step_id for step in workflow.steps}
            for step in workflow.steps:
                for dep in step.dependencies:
                    if dep not in step_ids:
                        logger.error(f"Step {step.step_id} has invalid dependency: {dep}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating workflow: {e}")
            return False
    
    async def _build_workflow_graph(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Build workflow execution graph"""
        try:
            graph = {
                "nodes": {},
                "edges": []
            }
            
            # Add nodes
            for step in workflow.steps:
                graph["nodes"][step.step_id] = {
                    "step": step,
                    "estimated_time": step.timeout,
                    "resource_requirements": step.resources
                }
            
            # Add edges
            for step in workflow.steps:
                for dep in step.dependencies:
                    graph["edges"].append({
                        "from": dep,
                        "to": step.step_id,
                        "type": "dependency"
                    })
            
            return graph
            
        except Exception as e:
            logger.error(f"Error building workflow graph: {e}")
            return {}
    
    async def _calculate_resource_requirements(self, workflow: WorkflowDefinition) -> Dict[ResourceType, float]:
        """Calculate resource requirements for workflow"""
        requirements = {}
        
        for step in workflow.steps:
            for resource_type, amount in step.resources.items():
                if resource_type not in requirements:
                    requirements[resource_type] = 0
                requirements[resource_type] = max(requirements[resource_type], amount)
        
        # Add buffer
        buffer_percentage = self.config["resource_buffer_percentage"]
        for resource_type in requirements:
            requirements[resource_type] *= (1 + buffer_percentage)
        
        return requirements
    
    async def _allocate_resource(self, resource_type: ResourceType, amount: float, execution_id: str) -> float:
        """Allocate specific resource"""
        try:
            if resource_type not in self.resource_pools:
                # Create default pool
                self.resource_pools[resource_type] = {
                    "capacity": 1000.0,
                    "allocated": 0.0,
                    "available": 1000.0
                }
            
            pool = self.resource_pools[resource_type]
            available = pool["available"]
            
            if available >= amount:
                pool["allocated"] += amount
                pool["available"] -= amount
                return amount
            else:
                # Allocate what's available
                pool["allocated"] += available
                pool["available"] = 0
                return available
                
        except Exception as e:
            logger.error(f"Error allocating resource: {e}")
            return 0.0
    
    async def _deallocate_resource(self, resource_type: ResourceType, amount: float, execution_id: str):
        """Deallocate specific resource"""
        try:
            if resource_type in self.resource_pools:
                pool = self.resource_pools[resource_type]
                pool["allocated"] -= amount
                pool["available"] += amount
                
                # Ensure non-negative values
                pool["allocated"] = max(0, pool["allocated"])
                pool["available"] = min(pool["capacity"], pool["available"])
                
        except Exception as e:
            logger.error(f"Error deallocating resource: {e}")
    
    async def _calculate_execution_cost(self, execution: WorkflowExecution) -> float:
        """Calculate execution cost"""
        try:
            cost = 0.0
            
            # Resource costs
            for resource_type, amount in execution.resource_usage.items():
                resource_cost = self._get_resource_cost(resource_type, amount, execution.execution_time)
                cost += resource_cost
            
            # Time-based cost
            time_cost = execution.execution_time * 0.01  # $0.01 per second
            cost += time_cost
            
            return cost
            
        except Exception as e:
            logger.error(f"Error calculating execution cost: {e}")
            return 0.0
    
    def _get_resource_cost(self, resource_type: ResourceType, amount: float, duration: float) -> float:
        """Get cost for resource usage"""
        # Cost per unit per second
        cost_rates = {
            ResourceType.CPU: 0.001,
            ResourceType.MEMORY: 0.0005,
            ResourceType.STORAGE: 0.0001,
            ResourceType.NETWORK: 0.0002,
            ResourceType.GPU: 0.01,
            ResourceType.CUSTOM: 0.001
        }
        
        rate = cost_rates.get(resource_type, 0.001)
        return amount * duration * rate
    
    async def _record_performance_metrics(self, execution: WorkflowExecution):
        """Record performance metrics"""
        try:
            workflow = self.workflows[execution.workflow_id]
            
            # Calculate metrics
            resource_efficiency = self._calculate_resource_efficiency(execution)
            cost_efficiency = self._calculate_cost_efficiency(execution)
            
            metrics = PerformanceMetrics(
                execution_id=execution.execution_id,
                workflow_id=execution.workflow_id,
                timestamp=datetime.utcnow(),
                execution_time=execution.execution_time,
                resource_efficiency=resource_efficiency,
                cost_efficiency=cost_efficiency,
                success_rate=1.0 if execution.status == ExecutionStatus.COMPLETED else 0.0,
                throughput=1.0 / execution.execution_time if execution.execution_time > 0 else 0.0,
                latency=execution.execution_time,
                error_rate=0.0 if execution.status == ExecutionStatus.COMPLETED else 1.0,
                adaptation_score=1.0 if execution.adaptation_applied else 0.0
            )
            
            self.performance_metrics[execution.workflow_id].append(metrics)
            
        except Exception as e:
            logger.error(f"Error recording performance metrics: {e}")
    
    def _calculate_resource_efficiency(self, execution: WorkflowExecution) -> float:
        """Calculate resource efficiency"""
        try:
            if not execution.resource_usage:
                return 0.0
            
            total_allocated = sum(execution.resource_usage.values())
            if total_allocated == 0:
                return 0.0
            
            # This would be based on actual usage vs allocated
            # For now, simulate efficiency
            return 0.8  # 80% efficiency
            
        except Exception as e:
            logger.error(f"Error calculating resource efficiency: {e}")
            return 0.0
    
    def _calculate_cost_efficiency(self, execution: WorkflowExecution) -> float:
        """Calculate cost efficiency"""
        try:
            if execution.cost == 0:
                return 1.0
            
            # Compare with baseline cost
            baseline_cost = self._get_baseline_cost(execution.workflow_id)
            if baseline_cost == 0:
                return 1.0
            
            return baseline_cost / execution.cost
            
        except Exception as e:
            logger.error(f"Error calculating cost efficiency: {e}")
            return 0.0
    
    def _get_baseline_cost(self, workflow_id: str) -> float:
        """Get baseline cost for workflow"""
        try:
            # Calculate average cost from historical data
            if workflow_id not in self.performance_metrics:
                return 0.0
            
            executions = [e for e in self.executions.values() if e.workflow_id == workflow_id]
            if not executions:
                return 0.0
            
            total_cost = sum(e.cost for e in executions)
            return total_cost / len(executions)
            
        except Exception as e:
            logger.error(f"Error getting baseline cost: {e}")
            return 0.0
    
    async def _initialize_resource_pools(self):
        """Initialize resource pools"""
        try:
            for resource_type in ResourceType:
                self.resource_pools[resource_type] = {
                    "capacity": 1000.0,
                    "allocated": 0.0,
                    "available": 1000.0,
                    "cost_per_unit": 0.001
                }
            
            logger.info("Resource pools initialized")
            
        except Exception as e:
            logger.error(f"Error initializing resource pools: {e}")
    
    async def _initialize_ml_models(self):
        """Initialize ML models"""
        try:
            if sklearn_available:
                # Initialize optimization models
                self.optimization_models["performance"] = RandomForestRegressor(n_estimators=100)
                self.optimization_models["cost"] = GradientBoostingRegressor(n_estimators=100)
                
                # Initialize scalers
                self.scalers["features"] = StandardScaler()
                self.scalers["targets"] = StandardScaler()
                
                logger.info("ML models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    async def _initialize_workflow_optimization(self, workflow_id: str):
        """Initialize optimization for a workflow"""
        try:
            # Initialize pattern tracking
            self.execution_patterns[workflow_id] = []
            self.failure_patterns[workflow_id] = []
            
            logger.info(f"Workflow optimization initialized: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Error initializing workflow optimization: {e}")
    
    async def _load_workflow_data(self):
        """Load workflow data from storage"""
        try:
            # Load workflows
            workflows_file = self.storage_path / "workflows.json"
            if workflows_file.exists():
                with open(workflows_file, 'r') as f:
                    workflows_data = json.load(f)
                    for workflow_data in workflows_data:
                        workflow = WorkflowDefinition(**workflow_data)
                        self.workflows[workflow.workflow_id] = workflow
            
            # Load executions
            executions_file = self.storage_path / "executions.json"
            if executions_file.exists():
                with open(executions_file, 'r') as f:
                    executions_data = json.load(f)
                    for execution_data in executions_data:
                        execution = WorkflowExecution(**execution_data)
                        self.executions[execution.execution_id] = execution
            
            logger.info("Workflow data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading workflow data: {e}")
    
    async def _save_workflow_data(self):
        """Save workflow data to storage"""
        try:
            # Save workflows
            workflows_data = []
            for workflow in self.workflows.values():
                workflows_data.append(asdict(workflow))
            
            workflows_file = self.storage_path / "workflows.json"
            with open(workflows_file, 'w') as f:
                json.dump(workflows_data, f, indent=2, default=str)
            
            # Save executions
            executions_data = []
            for execution in self.executions.values():
                executions_data.append(asdict(execution))
            
            executions_file = self.storage_path / "executions.json"
            with open(executions_file, 'w') as f:
                json.dump(executions_data, f, indent=2, default=str)
            
            logger.info("Workflow data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving workflow data: {e}")
    
    async def _collect_optimization_data(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Collect data for optimization"""
        try:
            data = []
            
            # Get executions for this workflow
            executions = [e for e in self.executions.values() if e.workflow_id == workflow_id]
            
            for execution in executions:
                if execution.status == ExecutionStatus.COMPLETED:
                    data_point = {
                        "execution_time": execution.execution_time,
                        "cost": execution.cost,
                        "resource_usage": execution.resource_usage,
                        "success": 1.0
                    }
                    data.append(data_point)
                elif execution.status == ExecutionStatus.FAILED:
                    data_point = {
                        "execution_time": execution.execution_time or 0.0,
                        "cost": execution.cost,
                        "resource_usage": execution.resource_usage,
                        "success": 0.0
                    }
                    data.append(data_point)
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting optimization data: {e}")
            return []
    
    async def _apply_optimization_strategy(self, workflow_id: str, strategy: OptimizationStrategy, historical_data: List[Dict[str, Any]]) -> OptimizationResult:
        """Apply optimization strategy"""
        try:
            optimization_id = f"opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{workflow_id}"
            
            # Analyze data and generate improvements
            improvements = {}
            new_configuration = {}
            
            if strategy == OptimizationStrategy.PERFORMANCE:
                # Optimize for performance
                avg_execution_time = statistics.mean([d["execution_time"] for d in historical_data])
                improvements["execution_time_reduction"] = 0.15  # 15% improvement
                new_configuration["resource_scaling"] = 1.2
                
            elif strategy == OptimizationStrategy.COST:
                # Optimize for cost
                avg_cost = statistics.mean([d["cost"] for d in historical_data])
                improvements["cost_reduction"] = 0.20  # 20% improvement
                new_configuration["resource_optimization"] = True
                
            elif strategy == OptimizationStrategy.BALANCED:
                # Balance performance and cost
                improvements["execution_time_reduction"] = 0.10
                improvements["cost_reduction"] = 0.15
                new_configuration["balanced_optimization"] = True
            
            # Calculate confidence
            confidence = 0.8  # Simulate confidence score
            
            optimization_result = OptimizationResult(
                optimization_id=optimization_id,
                workflow_id=workflow_id,
                strategy=strategy,
                improvements=improvements,
                new_configuration=new_configuration,
                confidence=confidence,
                created_at=datetime.utcnow()
            )
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error applying optimization strategy: {e}")
            raise
    
    async def _apply_optimization_configuration(self, workflow: WorkflowDefinition, configuration: Dict[str, Any]):
        """Apply optimization configuration"""
        try:
            # Apply configuration changes
            for key, value in configuration.items():
                if key == "resource_scaling":
                    # Scale resource requirements
                    for step in workflow.steps:
                        for resource_type in step.resources:
                            step.resources[resource_type] *= value
                            
                elif key == "resource_optimization":
                    # Optimize resource allocation
                    for step in workflow.steps:
                        for resource_type in step.resources:
                            step.resources[resource_type] *= 0.9  # 10% reduction
                            
                elif key == "balanced_optimization":
                    # Apply balanced optimization
                    for step in workflow.steps:
                        step.timeout = int(step.timeout * 0.9)  # Reduce timeout
                        for resource_type in step.resources:
                            step.resources[resource_type] *= 1.1  # Slight increase
            
            logger.info(f"Optimization configuration applied: {workflow.workflow_id}")
            
        except Exception as e:
            logger.error(f"Error applying optimization configuration: {e}")
    
    async def _determine_adaptation_strategy(self, workflow_id: str, trigger: AdaptationTrigger) -> Optional[Dict[str, Any]]:
        """Determine adaptation strategy"""
        try:
            strategy = None
            
            if trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION:
                strategy = {
                    "action": "scale_resources",
                    "resource_scaling": 1.5,
                    "impact": {"performance_improvement": 0.3}
                }
                
            elif trigger == AdaptationTrigger.RESOURCE_CONSTRAINT:
                strategy = {
                    "action": "optimize_resources",
                    "resource_optimization": True,
                    "impact": {"resource_efficiency": 0.2}
                }
                
            elif trigger == AdaptationTrigger.FAILURE_PATTERN:
                strategy = {
                    "action": "increase_retries",
                    "retry_improvement": True,
                    "impact": {"success_rate": 0.15}
                }
                
            elif trigger == AdaptationTrigger.COST_THRESHOLD:
                strategy = {
                    "action": "cost_optimize",
                    "cost_optimization": True,
                    "impact": {"cost_reduction": 0.25}
                }
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error determining adaptation strategy: {e}")
            return None
    
    async def _apply_adaptation_strategy(self, workflow_id: str, strategy: Dict[str, Any]) -> bool:
        """Apply adaptation strategy"""
        try:
            workflow = self.workflows[workflow_id]
            
            if strategy["action"] == "scale_resources":
                scaling_factor = strategy.get("resource_scaling", 1.5)
                for step in workflow.steps:
                    for resource_type in step.resources:
                        step.resources[resource_type] *= scaling_factor
                        
            elif strategy["action"] == "optimize_resources":
                for step in workflow.steps:
                    for resource_type in step.resources:
                        step.resources[resource_type] *= 0.8  # 20% reduction
                        
            elif strategy["action"] == "increase_retries":
                for step in workflow.steps:
                    step.retry_count = min(step.retry_count + 1, 5)
                    
            elif strategy["action"] == "cost_optimize":
                for step in workflow.steps:
                    step.timeout = int(step.timeout * 0.8)  # Reduce timeout
                    for resource_type in step.resources:
                        step.resources[resource_type] *= 0.9  # Reduce resources
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying adaptation strategy: {e}")
            return False
    
    async def _check_optimization_opportunity(self, workflow_id: str):
        """Check for optimization opportunity"""
        try:
            # Get recent executions
            recent_executions = [
                e for e in self.executions.values()
                if e.workflow_id == workflow_id and
                e.completed_at and
                e.completed_at > datetime.utcnow() - timedelta(hours=24)
            ]
            
            if len(recent_executions) < 10:
                return
            
            # Calculate performance metrics
            avg_execution_time = statistics.mean([e.execution_time for e in recent_executions])
            avg_cost = statistics.mean([e.cost for e in recent_executions])
            
            # Check if optimization is needed
            if avg_execution_time > 300:  # 5 minutes
                await self.optimize_workflow(workflow_id, OptimizationStrategy.PERFORMANCE)
            elif avg_cost > 10.0:  # $10
                await self.optimize_workflow(workflow_id, OptimizationStrategy.COST)
            
        except Exception as e:
            logger.error(f"Error checking optimization opportunity: {e}")
    
    async def _update_resource_usage(self):
        """Update resource usage metrics"""
        try:
            for resource_type in ResourceType:
                if resource_type in self.resource_pools:
                    pool = self.resource_pools[resource_type]
                    usage_percentage = (pool["allocated"] / pool["capacity"]) * 100
                    self.resource_usage[resource_type] = usage_percentage
                    
        except Exception as e:
            logger.error(f"Error updating resource usage: {e}")
    
    async def _check_resource_constraints(self):
        """Check for resource constraints"""
        try:
            for resource_type, usage in self.resource_usage.items():
                if usage > 90:  # 90% utilization
                    # Trigger resource scaling
                    await self.scale_resources(resource_type, 1.5)
                    
                    # Trigger adaptation for workflows
                    for workflow_id in self.workflows.keys():
                        await self.auto_adapt_workflow(workflow_id, AdaptationTrigger.RESOURCE_CONSTRAINT)
                    
        except Exception as e:
            logger.error(f"Error checking resource constraints: {e}")
    
    async def _check_performance_degradation(self):
        """Check for performance degradation"""
        try:
            for workflow_id in self.workflows.keys():
                if workflow_id in self.performance_metrics:
                    metrics = list(self.performance_metrics[workflow_id])
                    if len(metrics) >= 10:
                        # Check recent vs historical performance
                        recent_metrics = metrics[-5:]
                        historical_metrics = metrics[-20:-5]
                        
                        if recent_metrics and historical_metrics:
                            recent_avg = statistics.mean([m.execution_time for m in recent_metrics])
                            historical_avg = statistics.mean([m.execution_time for m in historical_metrics])
                            
                            # Check for degradation
                            if recent_avg > historical_avg * 1.3:  # 30% degradation
                                await self.auto_adapt_workflow(workflow_id, AdaptationTrigger.PERFORMANCE_DEGRADATION)
                
        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
    
    async def _check_failure_patterns(self):
        """Check for failure patterns"""
        try:
            for workflow_id in self.workflows.keys():
                # Get recent failures
                recent_failures = [
                    e for e in self.executions.values()
                    if e.workflow_id == workflow_id and
                    e.status == ExecutionStatus.FAILED and
                    e.completed_at and
                    e.completed_at > datetime.utcnow() - timedelta(hours=24)
                ]
                
                if len(recent_failures) >= 3:  # 3 failures in 24 hours
                    await self.auto_adapt_workflow(workflow_id, AdaptationTrigger.FAILURE_PATTERN)
                
        except Exception as e:
            logger.error(f"Error checking failure patterns: {e}")
    
    async def _check_adaptation_triggers(self, execution: WorkflowExecution):
        """Check for adaptation triggers after execution"""
        try:
            # Check cost threshold
            if execution.cost > 50.0:  # $50 threshold
                await self.auto_adapt_workflow(execution.workflow_id, AdaptationTrigger.COST_THRESHOLD)
            
            # Check execution time
            if execution.execution_time > 1800:  # 30 minutes
                await self.auto_adapt_workflow(execution.workflow_id, AdaptationTrigger.PERFORMANCE_DEGRADATION)
            
        except Exception as e:
            logger.error(f"Error checking adaptation triggers: {e}")
    
    async def _update_global_metrics(self):
        """Update global metrics"""
        try:
            # Calculate average execution time
            all_executions = [e for e in self.executions.values() if e.completed_at]
            if all_executions:
                self.global_metrics["average_execution_time"] = statistics.mean([e.execution_time for e in all_executions])
                self.global_metrics["total_cost"] = sum([e.cost for e in all_executions])
            
            # Calculate success rate
            if self.global_metrics["total_executions"] > 0:
                success_rate = (self.global_metrics["successful_executions"] / self.global_metrics["total_executions"]) * 100
                self.global_metrics["success_rate"] = success_rate
            
        except Exception as e:
            logger.error(f"Error updating global metrics: {e}")
    
    async def _analyze_execution_patterns(self):
        """Analyze execution patterns"""
        try:
            # This would implement pattern analysis
            # For now, just log that analysis is happening
            logger.debug("Analyzing execution patterns...")
            
        except Exception as e:
            logger.error(f"Error analyzing execution patterns: {e}")
    
    # ===== PUBLIC API METHODS =====
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "workflows": {
                "total": len(self.workflows),
                "active": len([w for w in self.workflows.values() if w.workflow_id in self.running_executions])
            },
            "executions": {
                "total": len(self.executions),
                "running": len(self.running_executions),
                "queued": len(self.execution_queue),
                "completed": len([e for e in self.executions.values() if e.status == ExecutionStatus.COMPLETED]),
                "failed": len([e for e in self.executions.values() if e.status == ExecutionStatus.FAILED])
            },
            "resources": {
                "pools": len(self.resource_pools),
                "usage": self.resource_usage,
                "total_allocated": sum(pool["allocated"] for pool in self.resource_pools.values())
            },
            "optimizations": {
                "total": len(self.optimization_results),
                "applied": len([o for o in self.optimization_results.values() if o.applied])
            },
            "adaptations": {
                "total": len(self.adaptation_events),
                "recent": len([e for e in self.adaptation_events if e.timestamp > datetime.utcnow() - timedelta(hours=24)])
            },
            "metrics": self.global_metrics
        }
    
    async def get_workflow_list(self) -> List[Dict[str, Any]]:
        """Get list of all workflows"""
        return [asdict(workflow) for workflow in self.workflows.values()]
    
    async def get_execution_history(self, workflow_id: str = None) -> List[Dict[str, Any]]:
        """Get execution history"""
        executions = list(self.executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        
        return [asdict(execution) for execution in executions]
    
    async def get_performance_analytics(self, workflow_id: str = None) -> Dict[str, Any]:
        """Get performance analytics"""
        if workflow_id:
            if workflow_id in self.performance_metrics:
                metrics = list(self.performance_metrics[workflow_id])
                return {
                    "workflow_id": workflow_id,
                    "total_executions": len(metrics),
                    "average_execution_time": statistics.mean([m.execution_time for m in metrics]) if metrics else 0,
                    "average_cost": statistics.mean([self.executions[m.execution_id].cost for m in metrics]) if metrics else 0,
                    "success_rate": statistics.mean([m.success_rate for m in metrics]) if metrics else 0,
                    "resource_efficiency": statistics.mean([m.resource_efficiency for m in metrics]) if metrics else 0
                }
        
        return {"global_metrics": self.global_metrics}

# Create global instance
adaptive_workflow_service = AdaptiveWorkflowService() 