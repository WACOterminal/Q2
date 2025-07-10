import threading
import pulsar
import logging
import json
from typing import Dict, List, Optional, Set
import time
import random
from enum import Enum
from dataclasses import dataclass, asdict, field
from shared.pulsar_client import SharedPulsarClient

logger = logging.getLogger(__name__)

class AgentStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

@dataclass
class AgentCapabilities:
    """Defines what an agent can do"""
    personalities: Set[str]
    max_concurrent_tasks: int = 5
    supported_tools: Set[str] = field(default_factory=set)
    resource_requirements: Dict[str, float] = field(default_factory=dict)  # cpu, memory, gpu
    preferred_task_types: Set[str] = field(default_factory=set)

@dataclass
class AgentMetrics:
    """Performance and health metrics for an agent"""
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_task_duration: float = 0.0
    current_load: int = 0  # number of active tasks
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_heartbeat: float = 0.0
    response_time_p95: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.total_tasks_completed + self.total_tasks_failed
        return (self.total_tasks_completed / total * 100) if total > 0 else 100.0
    
    @property
    def load_factor(self) -> float:
        """Returns load as percentage of max capacity"""
        return self.current_load / 5.0  # Default max concurrent tasks

class Agent:
    def __init__(self, agent_id: str, personality: str, topic_name: str, 
                 capabilities: Optional[AgentCapabilities] = None):
        self.agent_id = agent_id
        self.personality = personality  # Primary personality for backward compatibility
        self.topic_name = topic_name
        self.last_seen = time.time()
        self.registered_at = time.time()
        self.status = AgentStatus.HEALTHY
        
        # Enhanced features
        self.capabilities = capabilities or AgentCapabilities(personalities={personality})
        self.metrics = AgentMetrics(last_heartbeat=time.time())
        self.task_history: List[Dict] = []  # Recent task performance history
        self.tags: Dict[str, str] = {}  # Custom metadata tags
        
    def update_heartbeat(self, metrics_data: Optional[Dict] = None):
        """Update agent heartbeat and optionally metrics"""
        self.last_seen = time.time()
        self.metrics.last_heartbeat = self.last_seen
        
        if metrics_data:
            self.update_metrics(metrics_data)
        
        self._update_status()
    
    def update_metrics(self, metrics_data: Dict):
        """Update agent performance metrics"""
        for key, value in metrics_data.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
    
    def _update_status(self):
        """Update agent status based on metrics and heartbeat"""
        now = time.time()
        heartbeat_age = now - self.metrics.last_heartbeat
        
        if heartbeat_age > 300:  # 5 minutes
            self.status = AgentStatus.OFFLINE
        elif self.metrics.error_rate > 50 or self.metrics.cpu_usage > 90:
            self.status = AgentStatus.UNHEALTHY
        elif self.metrics.error_rate > 20 or self.metrics.load_factor > 0.8:
            self.status = AgentStatus.DEGRADED
        else:
            self.status = AgentStatus.HEALTHY
    
    def can_handle_task(self, personality: str, tools_required: Optional[Set[str]] = None) -> bool:
        """Check if agent can handle a task with given requirements"""
        if self.status in [AgentStatus.UNHEALTHY, AgentStatus.OFFLINE]:
            return False
        
        if personality not in self.capabilities.personalities:
            return False
        
        if self.metrics.current_load >= self.capabilities.max_concurrent_tasks:
            return False
        
        if tools_required and not tools_required.issubset(self.capabilities.supported_tools):
            return False
        
        return True
    
    def get_priority_score(self, personality: str, task_type: Optional[str] = None) -> float:
        """Calculate priority score for task assignment (higher = better)"""
        if not self.can_handle_task(personality):
            return 0.0
        
        score = 100.0
        
        # Health factor (0.5-1.0 multiplier)
        health_multiplier = {
            AgentStatus.HEALTHY: 1.0,
            AgentStatus.DEGRADED: 0.7,
            AgentStatus.UNHEALTHY: 0.3,
            AgentStatus.OFFLINE: 0.0
        }[self.status]
        score *= health_multiplier
        
        # Load factor (prefer less loaded agents)
        score *= (1.0 - self.metrics.load_factor)
        
        # Success rate bonus
        score *= (self.metrics.success_rate / 100.0)
        
        # Task type preference bonus
        if task_type and task_type in self.capabilities.preferred_task_types:
            score *= 1.2
        
        # Primary personality bonus
        if personality == self.personality:
            score *= 1.1
        
        return score
    
    def to_dict(self) -> Dict:
        """Serialize agent to dictionary"""
        return {
            'agent_id': self.agent_id,
            'personality': self.personality,
            'topic_name': self.topic_name,
            'last_seen': self.last_seen,
            'registered_at': self.registered_at,
            'status': self.status.value,
            'capabilities': asdict(self.capabilities),
            'metrics': asdict(self.metrics),
            'tags': self.tags
        }

class AgentRegistry(threading.Thread):
    def __init__(self, pulsar_client: SharedPulsarClient, 
                 registration_topic: str = "persistent://public/default/q.agentq.registrations",
                 heartbeat_topic: str = "persistent://public/default/q.agentq.heartbeats"):
        super().__init__(daemon=True)
        self.pulsar_client = pulsar_client
        self.registration_topic = registration_topic
        self.heartbeat_topic = heartbeat_topic
        self.agents: Dict[str, Agent] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.agent_timeout = 120  # seconds
        self.last_health_check = time.time()
        
        # Performance tracking
        self.assignment_history: List[Dict] = []
        self.max_history_size = 1000

    def run(self):
        """Main registry loop handling registrations and heartbeats"""
        self.pulsar_client._connect()
        if not self.pulsar_client._client:
            logger.error("Pulsar client not available in AgentRegistry. Thread will exit.")
            return

        # Subscribe to registration and heartbeat topics
        registration_consumer = self.pulsar_client._client.subscribe(
            self.registration_topic, "managerq-registry-sub"
        )
        heartbeat_consumer = self.pulsar_client._client.subscribe(
            self.heartbeat_topic, "managerq-heartbeat-sub"
        )
        
        logger.info(f"AgentRegistry started. Listening on {self.registration_topic} and {self.heartbeat_topic}")
        
        while not self._stop_event.is_set():
            try:
                # Handle registrations
                try:
                    msg = registration_consumer.receive(timeout_millis=500)
                    if msg:
                        self._handle_registration(msg)
                        registration_consumer.acknowledge(msg)
                except pulsar.Timeout:
                    pass
                
                # Handle heartbeats
                try:
                    msg = heartbeat_consumer.receive(timeout_millis=500)
                    if msg:
                        self._handle_heartbeat(msg)
                        heartbeat_consumer.acknowledge(msg)
                except pulsar.Timeout:
                    pass
                
                # Periodic health check
                if time.time() - self.last_health_check > self.health_check_interval:
                    self._perform_health_check()
                    self.last_health_check = time.time()
                    
            except Exception as e:
                logger.error(f"Error in AgentRegistry consumer loop: {e}", exc_info=True)

    def _handle_registration(self, msg):
        """Handle agent registration message"""
        try:
            data = msg.data().decode('utf-8')
            
            # Support both old format (CSV) and new format (JSON)
            if data.startswith('{'):
                # New JSON format with capabilities
                registration_data = json.loads(data)
                agent_id = registration_data['agent_id']
                personality = registration_data['personality']
                topic_name = registration_data['topic_name']
                
                # Parse capabilities if provided
                capabilities = None
                if 'capabilities' in registration_data:
                    cap_data = registration_data['capabilities']
                    capabilities = AgentCapabilities(
                        personalities=set(cap_data.get('personalities', [personality])),
                        max_concurrent_tasks=cap_data.get('max_concurrent_tasks', 5),
                        supported_tools=set(cap_data.get('supported_tools', [])),
                        resource_requirements=cap_data.get('resource_requirements', {}),
                        preferred_task_types=set(cap_data.get('preferred_task_types', []))
                    )
            else:
                # Legacy CSV format: "agent_id,personality,topic_name"
                agent_id, personality, topic_name = data.split(',')
                capabilities = None
            
            with self._lock:
                if agent_id in self.agents:
                    # Update existing agent
                    agent = self.agents[agent_id]
                    agent.update_heartbeat()
                    if capabilities:
                        agent.capabilities = capabilities
                else:
                    # Register new agent
                    agent = Agent(agent_id, personality, topic_name, capabilities)
                    self.agents[agent_id] = agent
                
                logger.info(f"Registered/updated agent: {agent_id} ({personality}) - Status: {agent.status.value}")
                
        except Exception as e:
            logger.error(f"Error handling registration: {e}", exc_info=True)

    def _handle_heartbeat(self, msg):
        """Handle agent heartbeat message"""
        try:
            data = json.loads(msg.data().decode('utf-8'))
            agent_id = data.get('agent_id')
            
            if not agent_id:
                return
            
            with self._lock:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    agent.update_heartbeat(data.get('metrics', {}))
                else:
                    logger.warning(f"Received heartbeat for unknown agent: {agent_id}")
                    
        except Exception as e:
            logger.error(f"Error handling heartbeat: {e}", exc_info=True)

    def _perform_health_check(self):
        """Perform periodic health check on all agents"""
        current_time = time.time()
        
        with self._lock:
            for agent in self.agents.values():
                agent._update_status()
                
                # Remove agents that have been offline too long
                if (current_time - agent.metrics.last_heartbeat) > self.agent_timeout:
                    logger.warning(f"Agent {agent.agent_id} has been offline for too long, removing from registry")
                    
        # Clean up offline agents
        offline_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if (current_time - agent.metrics.last_heartbeat) > self.agent_timeout
        ]
        
        for agent_id in offline_agents:
            del self.agents[agent_id]

    def stop(self):
        self._stop_event.set()

    def get_agent(self, personality: str, tools_required: Optional[Set[str]] = None, 
                  task_type: Optional[str] = None) -> Optional[Agent]:
        """Get best available agent for given requirements"""
        with self._lock:
            available_agents = [
                agent for agent in self.agents.values()
                if agent.can_handle_task(personality, tools_required)
            ]
            
            if not available_agents:
                return None
            
            # Sort by priority score (highest first)
            available_agents.sort(
                key=lambda a: a.get_priority_score(personality, task_type),
                reverse=True
            )
            
            selected_agent = available_agents[0]
            
            # Track assignment for analytics
            self._track_assignment(selected_agent, personality, task_type)
            
            return selected_agent

    def get_agents_by_personality(self, personality: str) -> List[Agent]:
        """Get all agents that support a given personality"""
        with self._lock:
            return [
                agent for agent in self.agents.values()
                if personality in agent.capabilities.personalities
            ]

    def get_all_agents(self) -> List[Agent]:
        """Get all registered agents"""
        with self._lock:
            return list(self.agents.values())

    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Get specific agent by ID"""
        with self._lock:
            return self.agents.get(agent_id)

    def get_registry_stats(self) -> Dict:
        """Get comprehensive registry statistics"""
        with self._lock:
            total_agents = len(self.agents)
            healthy_agents = sum(1 for a in self.agents.values() if a.status == AgentStatus.HEALTHY)
            degraded_agents = sum(1 for a in self.agents.values() if a.status == AgentStatus.DEGRADED)
            unhealthy_agents = sum(1 for a in self.agents.values() if a.status == AgentStatus.UNHEALTHY)
            
            personalities = set()
            total_capacity = 0
            current_load = 0
            
            for agent in self.agents.values():
                personalities.update(agent.capabilities.personalities)
                total_capacity += agent.capabilities.max_concurrent_tasks
                current_load += agent.metrics.current_load
            
            return {
                'total_agents': total_agents,
                'healthy_agents': healthy_agents,
                'degraded_agents': degraded_agents,
                'unhealthy_agents': unhealthy_agents,
                'supported_personalities': list(personalities),
                'total_capacity': total_capacity,
                'current_load': current_load,
                'load_percentage': (current_load / total_capacity * 100) if total_capacity > 0 else 0,
                'average_success_rate': sum(a.metrics.success_rate for a in self.agents.values()) / total_agents if total_agents > 0 else 0
            }

    def _track_assignment(self, agent: Agent, personality: str, task_type: Optional[str]):
        """Track agent assignment for analytics"""
        assignment = {
            'timestamp': time.time(),
            'agent_id': agent.agent_id,
            'personality': personality,
            'task_type': task_type,
            'agent_status': agent.status.value,
            'agent_load': agent.metrics.current_load,
            'agent_success_rate': agent.metrics.success_rate
        }
        
        self.assignment_history.append(assignment)
        
        # Keep history size manageable
        if len(self.assignment_history) > self.max_history_size:
            self.assignment_history = self.assignment_history[-self.max_history_size:]

# Singleton instance
# pulsar_client must be initialized and passed in when the app starts
agent_registry: Optional[AgentRegistry] = None 