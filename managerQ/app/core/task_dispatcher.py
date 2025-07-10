import logging
import uuid
import time
import asyncio
from typing import Dict, Any, Optional, Set, List, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq

from .agent_registry import AgentRegistry
from shared.pulsar_client import SharedPulsarClient

logger = logging.getLogger(__name__)

class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    
    @property
    def value_int(self) -> int:
        return {"critical": 4, "high": 3, "normal": 2, "low": 1}[self.value]

class RoutingStrategy(str, Enum):
    """Task routing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PRIORITY_BASED = "priority_based"
    AFFINITY_BASED = "affinity_based"
    RESOURCE_AWARE = "resource_aware"

@dataclass
class TaskRequest:
    """Enhanced task request with metadata"""
    task_id: str
    personality: str
    prompt: str
    priority: TaskPriority = TaskPriority.NORMAL
    workflow_id: Optional[str] = None
    user_id: Optional[str] = None
    tools_required: Set[str] = field(default_factory=set)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    affinity_rules: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    
    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority.value_int != other.priority.value_int:
            return self.priority.value_int > other.priority.value_int
        return self.created_at < other.created_at

@dataclass
class AffinityRule:
    """Agent affinity rules for task assignment"""
    rule_type: str  # "prefer", "avoid", "require"
    agent_id: Optional[str] = None
    agent_tags: Dict[str, str] = field(default_factory=dict)
    condition: Optional[Callable] = None
    weight: float = 1.0

@dataclass
class RoutingMetrics:
    """Metrics for routing decisions"""
    tasks_dispatched: int = 0
    tasks_failed: int = 0
    average_queue_time: float = 0.0
    routing_failures: int = 0
    last_reset: float = field(default_factory=time.time)

class CircuitBreaker:
    """Circuit breaker for agent failure handling"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True
    
    def record_success(self):
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class TaskDispatcher:
    def __init__(self, pulsar_client: SharedPulsarClient, agent_registry: AgentRegistry):
        self.pulsar_client = pulsar_client
        self.agent_registry = agent_registry
        
        # Enhanced load balancing features
        self.routing_strategy = RoutingStrategy.PRIORITY_BASED
        self.task_queue = []  # Priority queue
        self.pending_tasks: Dict[str, TaskRequest] = {}
        self.agent_round_robin: Dict[str, int] = defaultdict(int)
        self.affinity_rules: List[AffinityRule] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.routing_metrics = RoutingMetrics()
        
        # Task batching
        self.batch_size = 10
        self.batch_timeout = 5.0  # seconds
        self.batched_tasks: Dict[str, List[TaskRequest]] = defaultdict(list)
        self.last_batch_time: Dict[str, float] = defaultdict(float)
        
        # Performance tracking
        self.task_completion_times: deque = deque(maxlen=1000)
        self.agent_performance: Dict[str, Dict] = defaultdict(dict)
        
        # Queue processing
        self._queue_processor_running = False
    
    async def start_queue_processor(self):
        """Start the background queue processor"""
        if self._queue_processor_running:
            return
        
        self._queue_processor_running = True
        asyncio.create_task(self._process_queue_continuously())
        logger.info("Task queue processor started")
    
    async def stop_queue_processor(self):
        """Stop the background queue processor"""
        self._queue_processor_running = False
        logger.info("Task queue processor stopped")
    
    async def _process_queue_continuously(self):
        """Continuously process the task queue"""
        while self._queue_processor_running:
            try:
                await self._process_pending_tasks()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error(f"Error in queue processor: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    def dispatch_task(self, personality: str, prompt: str, workflow_id: Optional[str] = None,
                     priority: TaskPriority = TaskPriority.NORMAL, user_id: Optional[str] = None,
                     tools_required: Optional[Set[str]] = None, 
                     resource_requirements: Optional[Dict[str, float]] = None,
                     affinity_rules: Optional[Dict[str, Any]] = None,
                     timeout_seconds: int = 300) -> str:
        """
        Enhanced task dispatch with intelligent load balancing.
        Maintains backward compatibility while adding new features.
        """
        task_id = str(uuid.uuid4())
        
        task_request = TaskRequest(
            task_id=task_id,
            personality=personality,
            prompt=prompt,
            priority=priority,
            workflow_id=workflow_id,
            user_id=user_id,
            tools_required=tools_required or set(),
            resource_requirements=resource_requirements or {},
            affinity_rules=affinity_rules or {},
            timeout_seconds=timeout_seconds
        )
        
        # Try immediate dispatch for high priority tasks
        if priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
            agent = self._select_best_agent(task_request)
            if agent and self._can_dispatch_immediately(agent, task_request):
                return self._dispatch_to_agent(task_request, agent)
        
        # Queue the task for batch processing
        self._queue_task(task_request)
        logger.info(f"Queued task {task_id} with priority {priority.value}")
        return task_id
    
    def _queue_task(self, task_request: TaskRequest):
        """Add task to priority queue"""
        heapq.heappush(self.task_queue, task_request)
        self.pending_tasks[task_request.task_id] = task_request
    
    async def _process_pending_tasks(self):
        """Process tasks from the priority queue"""
        if not self.task_queue:
            return
        
        # Process high priority tasks first
        processed_count = 0
        max_process_per_cycle = 50
        
        while self.task_queue and processed_count < max_process_per_cycle:
            task_request = heapq.heappop(self.task_queue)
            
            # Check if task has expired
            if time.time() - task_request.created_at > task_request.timeout_seconds:
                logger.warning(f"Task {task_request.task_id} expired, removing from queue")
                self.pending_tasks.pop(task_request.task_id, None)
                continue
            
            # Try to find an agent for the task
            agent = self._select_best_agent(task_request)
            if agent:
                try:
                    self._dispatch_to_agent(task_request, agent)
                    self.pending_tasks.pop(task_request.task_id, None)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Failed to dispatch task {task_request.task_id}: {e}")
                    self._handle_dispatch_failure(task_request)
            else:
                # No suitable agent available, requeue with lower priority if not critical
                if task_request.priority != TaskPriority.CRITICAL:
                    heapq.heappush(self.task_queue, task_request)
                break
    
    def _select_best_agent(self, task_request: TaskRequest):
        """Select the best agent based on routing strategy and requirements"""
        available_agents = [
            agent for agent in self.agent_registry.get_agents_by_personality(task_request.personality)
            if self._agent_can_handle_task(agent, task_request)
        ]
        
        if not available_agents:
            return None
        
        # Apply routing strategy
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_agents, task_request.personality)
        elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
            return self._least_loaded_selection(available_agents)
        elif self.routing_strategy == RoutingStrategy.PRIORITY_BASED:
            return self._priority_based_selection(available_agents, task_request)
        elif self.routing_strategy == RoutingStrategy.AFFINITY_BASED:
            return self._affinity_based_selection(available_agents, task_request)
        elif self.routing_strategy == RoutingStrategy.RESOURCE_AWARE:
            return self._resource_aware_selection(available_agents, task_request)
        else:
            return available_agents[0]  # Fallback
    
    def _agent_can_handle_task(self, agent, task_request: TaskRequest) -> bool:
        """Check if agent can handle the task considering all constraints"""
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(agent.agent_id)
        if circuit_breaker and not circuit_breaker.can_execute():
            return False
        
        # Check basic capability
        if not agent.can_handle_task(task_request.personality, task_request.tools_required):
            return False
        
        # Check resource requirements
        if task_request.resource_requirements:
            for resource, required in task_request.resource_requirements.items():
                available = agent.capabilities.resource_requirements.get(resource, 0)
                if available < required:
                    return False
        
        # Check affinity rules
        if not self._check_affinity_rules(agent, task_request):
            return False
        
        return True
    
    def _check_affinity_rules(self, agent, task_request: TaskRequest) -> bool:
        """Check if agent satisfies affinity rules"""
        for rule_type, rule_value in task_request.affinity_rules.items():
            if rule_type == "prefer_agent" and agent.agent_id != rule_value:
                continue  # Preference, not requirement
            elif rule_type == "avoid_agent" and agent.agent_id == rule_value:
                return False
            elif rule_type == "require_agent" and agent.agent_id != rule_value:
                return False
            elif rule_type == "require_tag":
                tag_key, tag_value = rule_value.split("=")
                if agent.tags.get(tag_key) != tag_value:
                    return False
        
        return True
    
    def _round_robin_selection(self, agents, personality: str):
        """Round-robin agent selection"""
        if not agents:
            return None
        
        index = self.agent_round_robin[personality] % len(agents)
        self.agent_round_robin[personality] += 1
        return agents[index]
    
    def _least_loaded_selection(self, agents):
        """Select agent with lowest current load"""
        return min(agents, key=lambda a: a.metrics.load_factor)
    
    def _priority_based_selection(self, agents, task_request: TaskRequest):
        """Select agent based on priority score and task requirements"""
        best_agent = max(agents, key=lambda a: a.get_priority_score(
            task_request.personality, 
            task_request.affinity_rules.get("task_type")
        ))
        return best_agent
    
    def _affinity_based_selection(self, agents, task_request: TaskRequest):
        """Select agent based on affinity rules"""
        # Apply affinity preferences
        preferred_agents = []
        for agent in agents:
            score = 0
            for rule_type, rule_value in task_request.affinity_rules.items():
                if rule_type == "prefer_agent" and agent.agent_id == rule_value:
                    score += 10
                elif rule_type == "prefer_tag":
                    tag_key, tag_value = rule_value.split("=")
                    if agent.tags.get(tag_key) == tag_value:
                        score += 5
            preferred_agents.append((agent, score))
        
        # Sort by affinity score and select best
        preferred_agents.sort(key=lambda x: x[1], reverse=True)
        return preferred_agents[0][0] if preferred_agents else agents[0]
    
    def _resource_aware_selection(self, agents, task_request: TaskRequest):
        """Select agent considering resource utilization and requirements"""
        def resource_score(agent):
            # Lower resource usage is better
            cpu_score = 1.0 - (agent.metrics.cpu_usage / 100.0)
            memory_score = 1.0 - (agent.metrics.memory_usage / 100.0)
            load_score = 1.0 - agent.metrics.load_factor
            
            # Bonus for having required resources
            resource_bonus = 1.0
            if task_request.resource_requirements:
                for resource, required in task_request.resource_requirements.items():
                    available = agent.capabilities.resource_requirements.get(resource, 0)
                    if available >= required * 1.5:  # 50% headroom
                        resource_bonus += 0.2
            
            return (cpu_score + memory_score + load_score) * resource_bonus
        
        return max(agents, key=resource_score)
    
    def _can_dispatch_immediately(self, agent, task_request: TaskRequest) -> bool:
        """Check if task can be dispatched immediately without queuing"""
        return (agent.metrics.current_load < agent.capabilities.max_concurrent_tasks * 0.8 and
                agent.metrics.cpu_usage < 80)
    
    def _dispatch_to_agent(self, task_request: TaskRequest, agent) -> str:
        """Dispatch task to specific agent"""
        task_data = {
            "id": task_request.task_id,
            "prompt": task_request.prompt,
            "workflow_id": task_request.workflow_id,
            "agent_personality": task_request.personality,
            "priority": task_request.priority.value,
            "user_id": task_request.user_id,
            "tools_required": list(task_request.tools_required),
            "resource_requirements": task_request.resource_requirements,
            "timeout_seconds": task_request.timeout_seconds,
            "created_at": task_request.created_at,
            "scheduled_at": time.time()
        }
        
        try:
            self.pulsar_client.publish_message(agent.topic_name, task_data)
            
            # Update metrics and tracking
            self._update_dispatch_metrics(task_request, agent)
            task_request.scheduled_at = time.time()
            
            logger.info(f"Dispatched task {task_request.task_id} to agent {agent.agent_id} "
                       f"(priority: {task_request.priority.value}, queue_time: "
                       f"{task_request.scheduled_at - task_request.created_at:.2f}s)")
            
            return task_request.task_id
            
        except Exception as e:
            logger.error(f"Failed to dispatch task {task_request.task_id} to agent {agent.agent_id}: {e}")
            self._record_agent_failure(agent.agent_id)
            raise
    
    def _handle_dispatch_failure(self, task_request: TaskRequest):
        """Handle task dispatch failure with retry logic"""
        task_request.retry_count += 1
        
        if task_request.retry_count <= task_request.max_retries:
            # Exponential backoff for retries
            delay = min(2 ** task_request.retry_count, 30)
            logger.info(f"Retrying task {task_request.task_id} in {delay}s (attempt {task_request.retry_count})")
            # In a real implementation, you'd schedule this with a delay
            heapq.heappush(self.task_queue, task_request)
        else:
            logger.error(f"Task {task_request.task_id} exceeded max retries, dropping")
            self.pending_tasks.pop(task_request.task_id, None)
    
    def _update_dispatch_metrics(self, task_request: TaskRequest, agent):
        """Update metrics for successful dispatch"""
        self.routing_metrics.tasks_dispatched += 1
        
        if task_request.scheduled_at is not None:
            queue_time = task_request.scheduled_at - task_request.created_at
            
            # Update average queue time
            total_time = self.routing_metrics.average_queue_time * (self.routing_metrics.tasks_dispatched - 1)
            self.routing_metrics.average_queue_time = (total_time + queue_time) / self.routing_metrics.tasks_dispatched
        
        # Update agent performance tracking
        agent_perf = self.agent_performance[agent.agent_id]
        agent_perf['last_assigned'] = time.time()
        agent_perf['total_assignments'] = agent_perf.get('total_assignments', 0) + 1
    
    def _record_agent_failure(self, agent_id: str):
        """Record agent failure for circuit breaker"""
        if agent_id not in self.circuit_breakers:
            self.circuit_breakers[agent_id] = CircuitBreaker()
        
        self.circuit_breakers[agent_id].record_failure()
        logger.warning(f"Recorded failure for agent {agent_id}, circuit breaker state: "
                      f"{self.circuit_breakers[agent_id].state}")
    
    def record_task_completion(self, task_id: str, success: bool, duration: float):
        """Record task completion for performance tracking"""
        if success:
            self.task_completion_times.append(duration)
            # Update circuit breaker for successful completion
            task_request = self.pending_tasks.get(task_id)
            if task_request:
                # Find the agent that handled this task (simplified)
                for agent_id, circuit_breaker in self.circuit_breakers.items():
                    circuit_breaker.record_success()
        else:
            self.routing_metrics.tasks_failed += 1
    
    def add_affinity_rule(self, rule: AffinityRule):
        """Add an affinity rule for task routing"""
        self.affinity_rules.append(rule)
        logger.info(f"Added affinity rule: {rule.rule_type}")
    
    def set_routing_strategy(self, strategy: RoutingStrategy):
        """Change the routing strategy"""
        self.routing_strategy = strategy
        logger.info(f"Routing strategy changed to: {strategy.value}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue and routing statistics"""
        return {
            'queue_size': len(self.task_queue),
            'pending_tasks': len(self.pending_tasks),
            'routing_strategy': self.routing_strategy.value,
            'tasks_dispatched': self.routing_metrics.tasks_dispatched,
            'tasks_failed': self.routing_metrics.tasks_failed,
            'average_queue_time': self.routing_metrics.average_queue_time,
            'circuit_breakers': {
                agent_id: cb.state for agent_id, cb in self.circuit_breakers.items()
            },
            'agent_round_robin_state': dict(self.agent_round_robin)
        }

# Singleton instance
task_dispatcher: Optional[TaskDispatcher] = None 