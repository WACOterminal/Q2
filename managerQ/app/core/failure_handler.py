import logging
import time
import asyncio
import threading
from typing import Dict, List, Optional, Set, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from .agent_registry import AgentRegistry, Agent, AgentStatus
from .task_dispatcher import TaskDispatcher, TaskRequest
from shared.pulsar_client import SharedPulsarClient

logger = logging.getLogger(__name__)

class FailureType(str, Enum):
    """Types of failures that can occur"""
    AGENT_TIMEOUT = "agent_timeout"
    AGENT_CRASH = "agent_crash"
    TASK_FAILURE = "task_failure"
    COMMUNICATION_ERROR = "communication_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    AUTHENTICATION_ERROR = "authentication_error"
    WORKFLOW_FAILURE = "workflow_failure"
    DEPENDENCY_FAILURE = "dependency_failure"

class RecoveryAction(str, Enum):
    """Recovery actions that can be taken"""
    RETRY_TASK = "retry_task"
    REASSIGN_TASK = "reassign_task"
    RESTART_AGENT = "restart_agent"
    SCALE_UP = "scale_up"
    CIRCUIT_BREAK = "circuit_break"
    NOTIFY_ADMIN = "notify_admin"
    DEGRADED_MODE = "degraded_mode"
    FALLBACK_SERVICE = "fallback_service"

@dataclass
class FailureEvent:
    """Represents a failure event in the system"""
    event_id: str
    failure_type: FailureType
    timestamp: float
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None
    error_message: str = ""
    stack_trace: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"  # low, medium, high, critical
    resolved: bool = False
    resolution_time: Optional[float] = None

@dataclass
class RecoveryStrategy:
    """Defines how to recover from specific failure patterns"""
    name: str
    failure_patterns: List[FailureType]
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[RecoveryAction] = field(default_factory=list)
    max_attempts: int = 3
    backoff_strategy: str = "exponential"  # linear, exponential, fixed
    timeout_seconds: int = 300
    priority: int = 1  # Higher number = higher priority

class HealthChecker:
    """Monitors agent health and detects failures"""
    
    def __init__(self, agent_registry: AgentRegistry):
        self.agent_registry = agent_registry
        self.health_checks: Dict[str, Dict] = {}
        self.failure_detectors: List[Callable] = []
        self.check_interval = 30  # seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start health checking"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._thread.start()
        logger.info("Health checker started")
    
    def stop(self):
        """Stop health checking"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join()
        logger.info("Health checker stopped")
    
    def _health_check_loop(self):
        """Main health checking loop"""
        while self._running:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}", exc_info=True)
                time.sleep(5)
    
    def _perform_health_checks(self):
        """Perform health checks on all agents"""
        current_time = time.time()
        
        for agent in self.agent_registry.get_all_agents():
            agent_health = self.health_checks.get(agent.agent_id, {})
            
            # Check heartbeat freshness
            heartbeat_age = current_time - agent.metrics.last_heartbeat
            if heartbeat_age > 120:  # 2 minutes
                self._report_failure(FailureEvent(
                    event_id=f"heartbeat_timeout_{agent.agent_id}_{int(current_time)}",
                    failure_type=FailureType.AGENT_TIMEOUT,
                    timestamp=current_time,
                    agent_id=agent.agent_id,
                    error_message=f"Agent heartbeat timeout: {heartbeat_age:.1f}s",
                    severity="high"
                ))
            
            # Check resource exhaustion
            if agent.metrics.cpu_usage > 95:
                self._report_failure(FailureEvent(
                    event_id=f"cpu_exhaustion_{agent.agent_id}_{int(current_time)}",
                    failure_type=FailureType.RESOURCE_EXHAUSTION,
                    timestamp=current_time,
                    agent_id=agent.agent_id,
                    error_message=f"CPU usage critical: {agent.metrics.cpu_usage}%",
                    severity="high",
                    context={"cpu_usage": agent.metrics.cpu_usage}
                ))
            
            # Check error rates
            if agent.metrics.error_rate > 75:
                self._report_failure(FailureEvent(
                    event_id=f"high_error_rate_{agent.agent_id}_{int(current_time)}",
                    failure_type=FailureType.TASK_FAILURE,
                    timestamp=current_time,
                    agent_id=agent.agent_id,
                    error_message=f"High error rate: {agent.metrics.error_rate}%",
                    severity="medium",
                    context={"error_rate": agent.metrics.error_rate}
                ))
            
            # Update health status
            agent_health['last_check'] = current_time
            agent_health['status'] = self._calculate_health_status(agent)
            self.health_checks[agent.agent_id] = agent_health
    
    def _calculate_health_status(self, agent: Agent) -> str:
        """Calculate overall health status for an agent"""
        if agent.status == AgentStatus.OFFLINE:
            return "critical"
        elif agent.status == AgentStatus.UNHEALTHY:
            return "unhealthy"
        elif agent.metrics.cpu_usage > 90 or agent.metrics.error_rate > 50:
            return "degraded"
        else:
            return "healthy"
    
    def _report_failure(self, failure_event: FailureEvent):
        """Report a detected failure"""
        # This would typically send the failure to the failure handler
        logger.warning(f"Health check detected failure: {failure_event.failure_type} "
                      f"for agent {failure_event.agent_id}")

class FailureRecoveryManager:
    """Manages failure recovery strategies and actions"""
    
    def __init__(self, agent_registry: AgentRegistry, task_dispatcher: TaskDispatcher):
        self.agent_registry = agent_registry
        self.task_dispatcher = task_dispatcher
        self.recovery_strategies: List[RecoveryStrategy] = []
        self.failure_history: deque = deque(maxlen=10000)
        self.recovery_attempts: Dict[str, Dict] = defaultdict(dict)
        self.blocked_agents: Set[str] = set()
        
        # Load default recovery strategies
        self._load_default_strategies()
    
    def _load_default_strategies(self):
        """Load default recovery strategies"""
        
        # Agent timeout recovery
        self.recovery_strategies.append(RecoveryStrategy(
            name="agent_timeout_recovery",
            failure_patterns=[FailureType.AGENT_TIMEOUT],
            actions=[RecoveryAction.CIRCUIT_BREAK, RecoveryAction.REASSIGN_TASK],
            max_attempts=2,
            timeout_seconds=60,
            priority=3
        ))
        
        # Task failure recovery
        self.recovery_strategies.append(RecoveryStrategy(
            name="task_failure_recovery",
            failure_patterns=[FailureType.TASK_FAILURE],
            actions=[RecoveryAction.RETRY_TASK, RecoveryAction.REASSIGN_TASK],
            max_attempts=3,
            backoff_strategy="exponential",
            timeout_seconds=300,
            priority=2
        ))
        
        # Resource exhaustion recovery
        self.recovery_strategies.append(RecoveryStrategy(
            name="resource_exhaustion_recovery",
            failure_patterns=[FailureType.RESOURCE_EXHAUSTION],
            actions=[RecoveryAction.SCALE_UP, RecoveryAction.DEGRADED_MODE],
            max_attempts=1,
            timeout_seconds=600,
            priority=4
        ))
        
        # Communication error recovery
        self.recovery_strategies.append(RecoveryStrategy(
            name="communication_recovery",
            failure_patterns=[FailureType.COMMUNICATION_ERROR],
            actions=[RecoveryAction.RETRY_TASK, RecoveryAction.CIRCUIT_BREAK],
            max_attempts=5,
            backoff_strategy="exponential",
            timeout_seconds=120,
            priority=2
        ))
    
    async def handle_failure(self, failure_event: FailureEvent) -> bool:
        """Handle a failure event and attempt recovery"""
        logger.info(f"Handling failure: {failure_event.failure_type} - {failure_event.error_message}")
        
        # Record the failure
        self.failure_history.append(failure_event)
        
        # Find applicable recovery strategy
        strategy = self._find_recovery_strategy(failure_event)
        if not strategy:
            logger.warning(f"No recovery strategy found for failure type: {failure_event.failure_type}")
            return False
        
        # Check if we've exceeded max attempts for this failure pattern
        attempt_key = f"{failure_event.agent_id}_{failure_event.failure_type.value}"
        attempts = self.recovery_attempts[attempt_key]
        
        if attempts.get('count', 0) >= strategy.max_attempts:
            logger.error(f"Max recovery attempts exceeded for {attempt_key}")
            await self._escalate_failure(failure_event, strategy)
            return False
        
        # Execute recovery actions
        success = await self._execute_recovery_actions(failure_event, strategy)
        
        # Update attempt tracking
        attempts['count'] = attempts.get('count', 0) + 1
        attempts['last_attempt'] = time.time()
        attempts['success'] = success
        
        if success:
            failure_event.resolved = True
            failure_event.resolution_time = time.time()
            logger.info(f"Successfully recovered from failure: {failure_event.event_id}")
        
        return success
    
    def _find_recovery_strategy(self, failure_event: FailureEvent) -> Optional[RecoveryStrategy]:
        """Find the best recovery strategy for a failure event"""
        applicable_strategies = [
            strategy for strategy in self.recovery_strategies
            if failure_event.failure_type in strategy.failure_patterns
        ]
        
        if not applicable_strategies:
            return None
        
        # Sort by priority (higher first)
        applicable_strategies.sort(key=lambda s: s.priority, reverse=True)
        return applicable_strategies[0]
    
    async def _execute_recovery_actions(self, failure_event: FailureEvent, strategy: RecoveryStrategy) -> bool:
        """Execute recovery actions defined in strategy"""
        success = False
        
        for action in strategy.actions:
            try:
                if action == RecoveryAction.RETRY_TASK:
                    success = await self._retry_task(failure_event)
                elif action == RecoveryAction.REASSIGN_TASK:
                    success = await self._reassign_task(failure_event)
                elif action == RecoveryAction.RESTART_AGENT:
                    success = await self._restart_agent(failure_event)
                elif action == RecoveryAction.SCALE_UP:
                    success = await self._scale_up_agents(failure_event)
                elif action == RecoveryAction.CIRCUIT_BREAK:
                    success = await self._circuit_break_agent(failure_event)
                elif action == RecoveryAction.NOTIFY_ADMIN:
                    success = await self._notify_admin(failure_event)
                elif action == RecoveryAction.DEGRADED_MODE:
                    success = await self._enable_degraded_mode(failure_event)
                elif action == RecoveryAction.FALLBACK_SERVICE:
                    success = await self._activate_fallback_service(failure_event)
                
                if success:
                    logger.info(f"Recovery action {action.value} succeeded for {failure_event.event_id}")
                    break
                else:
                    logger.warning(f"Recovery action {action.value} failed for {failure_event.event_id}")
                    
            except Exception as e:
                logger.error(f"Error executing recovery action {action.value}: {e}", exc_info=True)
        
        return success
    
    async def _retry_task(self, failure_event: FailureEvent) -> bool:
        """Retry a failed task"""
        if not failure_event.task_id:
            return False
        
        # Find a different agent for the task
        if failure_event.agent_id:
            self.blocked_agents.add(failure_event.agent_id)
        
        try:
            # This would need integration with the task dispatcher's retry mechanism
            logger.info(f"Retrying task {failure_event.task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to retry task {failure_event.task_id}: {e}")
            return False
    
    async def _reassign_task(self, failure_event: FailureEvent) -> bool:
        """Reassign a task to a different agent"""
        if not failure_event.task_id:
            return False
        
        try:
            # Block the failed agent temporarily
            if failure_event.agent_id:
                self.blocked_agents.add(failure_event.agent_id)
            
            logger.info(f"Reassigning task {failure_event.task_id} to different agent")
            return True
        except Exception as e:
            logger.error(f"Failed to reassign task {failure_event.task_id}: {e}")
            return False
    
    async def _restart_agent(self, failure_event: FailureEvent) -> bool:
        """Restart a failed agent (placeholder for Kubernetes integration)"""
        if not failure_event.agent_id:
            return False
        
        try:
            # This would integrate with Kubernetes to restart the agent pod
            logger.info(f"Restarting agent {failure_event.agent_id}")
            # In a real implementation, this would use the Kubernetes API
            return True
        except Exception as e:
            logger.error(f"Failed to restart agent {failure_event.agent_id}: {e}")
            return False
    
    async def _scale_up_agents(self, failure_event: FailureEvent) -> bool:
        """Scale up agents to handle increased load"""
        try:
            # This would integrate with the autoscaler
            logger.info("Scaling up agents due to resource exhaustion")
            return True
        except Exception as e:
            logger.error(f"Failed to scale up agents: {e}")
            return False
    
    async def _circuit_break_agent(self, failure_event: FailureEvent) -> bool:
        """Circuit break an agent to prevent cascading failures"""
        if not failure_event.agent_id:
            return False
        
        try:
            self.blocked_agents.add(failure_event.agent_id)
            logger.info(f"Circuit breaking agent {failure_event.agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to circuit break agent {failure_event.agent_id}: {e}")
            return False
    
    async def _notify_admin(self, failure_event: FailureEvent) -> bool:
        """Notify administrators of critical failures"""
        try:
            # This would integrate with alerting systems (Slack, email, PagerDuty, etc.)
            logger.critical(f"ADMIN NOTIFICATION: Critical failure - {failure_event.error_message}")
            return True
        except Exception as e:
            logger.error(f"Failed to notify admin: {e}")
            return False
    
    async def _enable_degraded_mode(self, failure_event: FailureEvent) -> bool:
        """Enable degraded mode operation"""
        try:
            # This would reduce system functionality to maintain core operations
            logger.warning("Enabling degraded mode due to system failures")
            return True
        except Exception as e:
            logger.error(f"Failed to enable degraded mode: {e}")
            return False
    
    async def _activate_fallback_service(self, failure_event: FailureEvent) -> bool:
        """Activate fallback service for critical operations"""
        try:
            # This would route traffic to backup services
            logger.info("Activating fallback service")
            return True
        except Exception as e:
            logger.error(f"Failed to activate fallback service: {e}")
            return False
    
    async def _escalate_failure(self, failure_event: FailureEvent, strategy: RecoveryStrategy):
        """Escalate failure when recovery attempts are exhausted"""
        logger.critical(f"ESCALATION: Recovery attempts exhausted for {failure_event.failure_type}")
        
        # Notify administrators
        await self._notify_admin(failure_event)
        
        # Take emergency actions
        if failure_event.agent_id:
            self.blocked_agents.add(failure_event.agent_id)
        
        # Could trigger additional emergency protocols here

class FailureHandler:
    """Main failure handling coordinator"""
    
    def __init__(self, agent_registry: AgentRegistry, task_dispatcher: TaskDispatcher, 
                 pulsar_client: SharedPulsarClient):
        self.agent_registry = agent_registry
        self.task_dispatcher = task_dispatcher
        self.pulsar_client = pulsar_client
        
        # Initialize components
        self.health_checker = HealthChecker(agent_registry)
        self.recovery_manager = FailureRecoveryManager(agent_registry, task_dispatcher)
        
        # Failure tracking
        self.active_failures: Dict[str, FailureEvent] = {}
        self.failure_patterns: Dict[str, List[FailureEvent]] = defaultdict(list)
        
        # Pulsar integration for failure events
        self.failure_topic = "persistent://public/default/q.system.failures"
        self._consumer: Optional[Any] = None
        self._running = False
    
    async def start(self):
        """Start the failure handling system"""
        if self._running:
            return
        
        self._running = True
        
        # Start health checker
        self.health_checker.start()
        
        # Start Pulsar consumer for failure events
        await self._start_failure_consumer()
        
        logger.info("Failure handling system started")
    
    async def stop(self):
        """Stop the failure handling system"""
        self._running = False
        
        # Stop health checker
        self.health_checker.stop()
        
        # Close Pulsar consumer
        if self._consumer:
            self._consumer.close()
        
        logger.info("Failure handling system stopped")
    
    async def _start_failure_consumer(self):
        """Start consuming failure events from Pulsar"""
        try:
            self.pulsar_client._connect()
            if self.pulsar_client._client:
                self._consumer = self.pulsar_client._client.subscribe(
                    self.failure_topic,
                    subscription_name="failure-handler-sub"
                )
                
                # Start consuming in background
                asyncio.create_task(self._consume_failure_events())
                
        except Exception as e:
            logger.error(f"Failed to start failure event consumer: {e}", exc_info=True)
    
    async def _consume_failure_events(self):
        """Consume failure events from Pulsar"""
        while self._running and self._consumer:
            try:
                msg = self._consumer.receive(timeout_millis=1000)
                if msg:
                    failure_data = json.loads(msg.data().decode('utf-8'))
                    failure_event = FailureEvent(**failure_data)
                    
                    await self.handle_failure(failure_event)
                    self._consumer.acknowledge(msg)
                    
            except Exception as e:
                logger.error(f"Error consuming failure events: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def handle_failure(self, failure_event: FailureEvent) -> bool:
        """Handle a failure event"""
        logger.info(f"Processing failure event: {failure_event.event_id}")
        
        # Track the failure
        self.active_failures[failure_event.event_id] = failure_event
        self.failure_patterns[failure_event.failure_type.value].append(failure_event)
        
        # Attempt recovery
        success = await self.recovery_manager.handle_failure(failure_event)
        
        # Clean up if resolved
        if success and failure_event.resolved:
            self.active_failures.pop(failure_event.event_id, None)
        
        return success
    
    def report_failure(self, failure_type: FailureType, agent_id: Optional[str] = None,
                      task_id: Optional[str] = None, workflow_id: Optional[str] = None,
                      error_message: str = "", severity: str = "medium",
                      context: Optional[Dict[str, Any]] = None) -> str:
        """Report a failure event"""
        
        event_id = f"{failure_type.value}_{int(time.time() * 1000)}"
        
        failure_event = FailureEvent(
            event_id=event_id,
            failure_type=failure_type,
            timestamp=time.time(),
            agent_id=agent_id,
            task_id=task_id,
            workflow_id=workflow_id,
            error_message=error_message,
            severity=severity,
            context=context or {}
        )
        
        # Process immediately (could also publish to Pulsar for async processing)
        asyncio.create_task(self.handle_failure(failure_event))
        
        return event_id
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """Get failure statistics"""
        total_failures = len(self.failure_patterns)
        active_failures = len(self.active_failures)
        
        failure_counts = {
            failure_type: len(events) 
            for failure_type, events in self.failure_patterns.items()
        }
        
        return {
            'total_failure_types': total_failures,
            'active_failures': active_failures,
            'failure_counts_by_type': failure_counts,
            'blocked_agents': list(self.recovery_manager.blocked_agents),
            'health_status': {
                agent_id: health.get('status', 'unknown')
                for agent_id, health in self.health_checker.health_checks.items()
            }
        }

# Singleton instance
failure_handler: Optional[FailureHandler] = None 