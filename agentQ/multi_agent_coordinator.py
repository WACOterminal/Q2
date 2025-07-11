"""
Multi-Agent Reinforcement Learning Coordinator for Q Platform

This service provides comprehensive multi-agent coordination capabilities:
- Multi-Agent Reinforcement Learning (MARL) algorithms
- Agent coordination and communication protocols
- Distributed decision making and consensus
- Hierarchical agent organization
- Shared experience replay and knowledge transfer
- Dynamic team formation and role assignment
- Multi-agent task decomposition and allocation
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Agent roles in multi-agent system"""
    LEADER = "leader"
    FOLLOWER = "follower"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    EXPLORER = "explorer"
    EXECUTOR = "executor"

class CoordinationProtocol(Enum):
    """Coordination protocols"""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"
    AUCTION = "auction"
    NEGOTIATION = "negotiation"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AgentState(Enum):
    """Agent states"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    OFFLINE = "offline"
    LEARNING = "learning"

@dataclass
class AgentProfile:
    """Agent profile with capabilities and preferences"""
    agent_id: str
    agent_name: str
    agent_type: str
    role: AgentRole
    capabilities: Dict[str, float]
    preferences: Dict[str, Any]
    current_state: AgentState
    performance_metrics: Dict[str, float]
    learning_progress: Dict[str, float]
    communication_channels: List[str]
    created_at: datetime
    last_active: datetime
    metadata: Dict[str, Any] = None

@dataclass
class Task:
    """Task definition for multi-agent coordination"""
    task_id: str
    task_name: str
    task_type: str
    description: str
    priority: TaskPriority
    required_capabilities: Dict[str, float]
    estimated_duration: int
    deadline: Optional[datetime]
    dependencies: List[str]
    assigned_agents: List[str]
    status: str
    created_at: datetime
    created_by: str
    metadata: Dict[str, Any] = None

@dataclass
class CoordinationEvent:
    """Coordination event between agents"""
    event_id: str
    event_type: str
    source_agent: str
    target_agents: List[str]
    message: Dict[str, Any]
    timestamp: datetime
    protocol: CoordinationProtocol
    response_required: bool = False
    responses: Dict[str, Any] = None

@dataclass
class Experience:
    """Experience tuple for multi-agent RL"""
    agent_id: str
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    joint_action: Optional[np.ndarray] = None
    team_reward: Optional[float] = None
    timestamp: datetime = None

@dataclass
class Coalition:
    """Coalition of agents working together"""
    coalition_id: str
    coalition_name: str
    member_agents: List[str]
    leader_agent: str
    objective: str
    formation_time: datetime
    expected_duration: int
    performance_metrics: Dict[str, float]
    communication_protocol: CoordinationProtocol
    status: str = "active"

# Neural Network Models for MARL

class MADDPGActor(nn.Module):
    """Multi-Agent Deep Deterministic Policy Gradient Actor"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(MADDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action

class MADDPGCritic(nn.Module):
    """Multi-Agent Deep Deterministic Policy Gradient Critic"""
    
    def __init__(self, state_dim: int, action_dim: int, n_agents: int, hidden_dim: int = 256):
        super(MADDPGCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim * n_agents + action_dim * n_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class QMIXNet(nn.Module):
    """QMIX mixing network for multi-agent value function"""
    
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 64):
        super(QMIXNet, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Hypernetworks for generating weights
        self.hyper_w1 = nn.Linear(state_dim, hidden_dim * n_agents)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, agent_q_values, global_state):
        batch_size = agent_q_values.size(0)
        
        # Generate weights and biases
        w1 = torch.abs(self.hyper_w1(global_state)).view(batch_size, self.n_agents, self.hidden_dim)
        w2 = torch.abs(self.hyper_w2(global_state)).view(batch_size, self.hidden_dim, 1)
        b1 = self.hyper_b1(global_state).view(batch_size, 1, self.hidden_dim)
        b2 = self.hyper_b2(global_state).view(batch_size, 1, 1)
        
        # Forward pass
        hidden = F.elu(torch.bmm(agent_q_values.unsqueeze(1), w1) + b1)
        q_total = torch.bmm(hidden, w2) + b2
        
        return q_total.squeeze()

class MultiAgentCoordinator:
    """
    Multi-Agent Reinforcement Learning Coordinator
    """
    
    def __init__(self, 
                 state_dim: int = 64,
                 action_dim: int = 8,
                 max_agents: int = 10,
                 kg_client: Optional[KnowledgeGraphClient] = None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_agents = max_agents
        self.kg_client = kg_client or KnowledgeGraphClient()
        
        # Agent management
        self.agents: Dict[str, AgentProfile] = {}
        self.tasks: Dict[str, Task] = {}
        self.coalitions: Dict[str, Coalition] = {}
        self.coordination_events: Dict[str, CoordinationEvent] = {}
        
        # MARL components
        self.actors: Dict[str, MADDPGActor] = {}
        self.critics: Dict[str, MADDPGCritic] = {}
        self.target_actors: Dict[str, MADDPGActor] = {}
        self.target_critics: Dict[str, MADDPGCritic] = {}
        self.actor_optimizers: Dict[str, optim.Adam] = {}
        self.critic_optimizers: Dict[str, optim.Adam] = {}
        
        # QMIX for value function decomposition
        self.qmix_net = QMIXNet(max_agents, state_dim)
        self.target_qmix_net = QMIXNet(max_agents, state_dim)
        self.qmix_optimizer = optim.Adam(self.qmix_net.parameters(), lr=0.001)
        
        # Experience replay
        self.experience_buffer: deque = deque(maxlen=10000)
        self.shared_buffer: deque = deque(maxlen=50000)
        
        # Configuration
        self.config = {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 64,
            "update_frequency": 100,
            "exploration_noise": 0.1,
            "target_update_frequency": 1000,
            "max_coalition_size": 5,
            "task_timeout": 3600,
            "communication_range": 10.0
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Performance metrics
        self.coordination_metrics = {
            "total_agents": 0,
            "active_agents": 0,
            "completed_tasks": 0,
            "active_coalitions": 0,
            "average_task_completion_time": 0.0,
            "coordination_efficiency": 0.0,
            "learning_progress": 0.0,
            "communication_overhead": 0.0
        }
        
        # Initialize target networks
        self._initialize_target_networks()
        
    async def initialize(self):
        """Initialize the multi-agent coordinator"""
        logger.info("Initializing Multi-Agent Coordinator")
        
        # Initialize KnowledgeGraph client
        await self.kg_client.initialize()
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        # Start background tasks
        self.background_tasks.add(asyncio.create_task(self._coordination_loop()))
        self.background_tasks.add(asyncio.create_task(self._learning_loop()))
        self.background_tasks.add(asyncio.create_task(self._coalition_management()))
        self.background_tasks.add(asyncio.create_task(self._metrics_tracking()))
        
        logger.info("Multi-Agent Coordinator initialized successfully")
    
    async def shutdown(self):
        """Shutdown the multi-agent coordinator"""
        logger.info("Shutting down Multi-Agent Coordinator")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Multi-Agent Coordinator shut down successfully")
    
    # ===== AGENT MANAGEMENT =====
    
    async def register_agent(
        self,
        agent_name: str,
        agent_type: str,
        capabilities: Dict[str, float],
        preferences: Optional[Dict[str, Any]] = None,
        role: AgentRole = AgentRole.FOLLOWER,
        communication_channels: Optional[List[str]] = None
    ) -> str:
        """Register a new agent in the multi-agent system"""
        
        agent_id = f"agent_{uuid.uuid4().hex[:12]}"
        
        agent_profile = AgentProfile(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            role=role,
            capabilities=capabilities,
            preferences=preferences or {},
            current_state=AgentState.IDLE,
            performance_metrics={
                "success_rate": 0.0,
                "average_response_time": 0.0,
                "collaboration_score": 0.0,
                "learning_rate": 0.0
            },
            learning_progress={
                "episodes": 0,
                "total_reward": 0.0,
                "average_reward": 0.0,
                "epsilon": 1.0
            },
            communication_channels=communication_channels or ["general"],
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
            metadata={}
        )
        
        self.agents[agent_id] = agent_profile
        
        # Initialize MARL components for the agent
        await self._initialize_agent_networks(agent_id)
        
        # Store in KnowledgeGraph
        await self._store_agent_in_kg(agent_profile)
        
        # Publish agent registration event
        await shared_pulsar_client.publish(
            "q.agents.coordination.agent.registered",
            {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_type": agent_type,
                "role": role.value,
                "capabilities": capabilities,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Registered agent: {agent_name} ({agent_id})")
        return agent_id
    
    async def _initialize_agent_networks(self, agent_id: str):
        """Initialize neural networks for an agent"""
        
        # Actor network
        actor = MADDPGActor(self.state_dim, self.action_dim)
        target_actor = MADDPGActor(self.state_dim, self.action_dim)
        target_actor.load_state_dict(actor.state_dict())
        
        # Critic network
        critic = MADDPGCritic(self.state_dim, self.action_dim, self.max_agents)
        target_critic = MADDPGCritic(self.state_dim, self.action_dim, self.max_agents)
        target_critic.load_state_dict(critic.state_dict())
        
        # Optimizers
        actor_optimizer = optim.Adam(actor.parameters(), lr=self.config["learning_rate"])
        critic_optimizer = optim.Adam(critic.parameters(), lr=self.config["learning_rate"])
        
        # Store networks
        self.actors[agent_id] = actor
        self.target_actors[agent_id] = target_actor
        self.critics[agent_id] = critic
        self.target_critics[agent_id] = target_critic
        self.actor_optimizers[agent_id] = actor_optimizer
        self.critic_optimizers[agent_id] = critic_optimizer
    
    async def update_agent_state(self, agent_id: str, state: AgentState):
        """Update agent state"""
        
        if agent_id not in self.agents:
            return False
        
        self.agents[agent_id].current_state = state
        self.agents[agent_id].last_active = datetime.utcnow()
        
        # Publish state update event
        await shared_pulsar_client.publish(
            "q.agents.coordination.agent.state_updated",
            {
                "agent_id": agent_id,
                "state": state.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return True
    
    async def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Get agent profile"""
        return self.agents.get(agent_id)
    
    async def list_agents(
        self,
        role: Optional[AgentRole] = None,
        state: Optional[AgentState] = None,
        agent_type: Optional[str] = None
    ) -> List[AgentProfile]:
        """List agents with optional filtering"""
        
        agents = list(self.agents.values())
        
        if role:
            agents = [a for a in agents if a.role == role]
        
        if state:
            agents = [a for a in agents if a.current_state == state]
        
        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]
        
        return agents
    
    # ===== TASK MANAGEMENT =====
    
    async def create_task(
        self,
        task_name: str,
        task_type: str,
        description: str,
        required_capabilities: Dict[str, float],
        priority: TaskPriority = TaskPriority.MEDIUM,
        estimated_duration: int = 3600,
        deadline: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new task"""
        
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        task = Task(
            task_id=task_id,
            task_name=task_name,
            task_type=task_type,
            description=description,
            priority=priority,
            required_capabilities=required_capabilities,
            estimated_duration=estimated_duration,
            deadline=deadline,
            dependencies=dependencies or [],
            assigned_agents=[],
            status="pending",
            created_at=datetime.utcnow(),
            created_by=created_by,
            metadata={}
        )
        
        self.tasks[task_id] = task
        
        # Trigger task allocation
        await self._allocate_task(task_id)
        
        # Store in KnowledgeGraph
        await self._store_task_in_kg(task)
        
        # Publish task creation event
        await shared_pulsar_client.publish(
            "q.agents.coordination.task.created",
            {
                "task_id": task_id,
                "task_name": task_name,
                "task_type": task_type,
                "priority": priority.value,
                "required_capabilities": required_capabilities,
                "created_by": created_by,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Created task: {task_name} ({task_id})")
        return task_id
    
    async def _allocate_task(self, task_id: str):
        """Allocate task to suitable agents"""
        
        task = self.tasks.get(task_id)
        if not task:
            return
        
        # Find suitable agents
        suitable_agents = await self._find_suitable_agents(task)
        
        if not suitable_agents:
            logger.warning(f"No suitable agents found for task: {task_id}")
            return
        
        # Select best agents based on capabilities and availability
        selected_agents = await self._select_best_agents(task, suitable_agents)
        
        # Assign task to selected agents
        task.assigned_agents = selected_agents
        task.status = "assigned"
        
        # Create coordination event
        await self._create_coordination_event(
            event_type="task_assignment",
            source_agent="coordinator",
            target_agents=selected_agents,
            message={
                "task_id": task_id,
                "task_name": task.task_name,
                "description": task.description,
                "priority": task.priority.value,
                "deadline": task.deadline.isoformat() if task.deadline else None
            },
            protocol=CoordinationProtocol.CENTRALIZED
        )
        
        logger.info(f"Allocated task {task_id} to agents: {selected_agents}")
    
    async def _find_suitable_agents(self, task: Task) -> List[str]:
        """Find agents suitable for a task"""
        
        suitable_agents = []
        
        for agent_id, agent in self.agents.items():
            if agent.current_state not in [AgentState.ACTIVE, AgentState.IDLE]:
                continue
            
            # Check capability match
            capability_match = True
            for capability, required_level in task.required_capabilities.items():
                agent_level = agent.capabilities.get(capability, 0.0)
                if agent_level < required_level:
                    capability_match = False
                    break
            
            if capability_match:
                suitable_agents.append(agent_id)
        
        return suitable_agents
    
    async def _select_best_agents(self, task: Task, suitable_agents: List[str]) -> List[str]:
        """Select best agents for a task"""
        
        # Calculate scores for each agent
        agent_scores = []
        
        for agent_id in suitable_agents:
            agent = self.agents[agent_id]
            
            # Calculate capability score
            capability_score = 0.0
            for capability, required_level in task.required_capabilities.items():
                agent_level = agent.capabilities.get(capability, 0.0)
                capability_score += min(agent_level / required_level, 1.0)
            
            capability_score /= len(task.required_capabilities)
            
            # Calculate performance score
            performance_score = agent.performance_metrics.get("success_rate", 0.0)
            
            # Calculate availability score
            availability_score = 1.0 if agent.current_state == AgentState.IDLE else 0.5
            
            # Combined score
            total_score = (capability_score * 0.5 + 
                          performance_score * 0.3 + 
                          availability_score * 0.2)
            
            agent_scores.append((agent_id, total_score))
        
        # Sort by score (descending)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top agents (limit based on task complexity)
        max_agents = min(3, len(agent_scores))  # Maximum 3 agents per task
        selected_agents = [agent_id for agent_id, _ in agent_scores[:max_agents]]
        
        return selected_agents
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark task as completed"""
        
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        task.status = "completed"
        task.metadata["result"] = result
        task.metadata["completion_time"] = datetime.utcnow().isoformat()
        
        # Update agent performance metrics
        for agent_id in task.assigned_agents:
            await self._update_agent_performance(agent_id, True)
        
        # Publish task completion event
        await shared_pulsar_client.publish(
            "q.agents.coordination.task.completed",
            {
                "task_id": task_id,
                "task_name": task.task_name,
                "assigned_agents": task.assigned_agents,
                "result": result,
                "completion_time": task.metadata["completion_time"],
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Completed task: {task_id}")
        return True
    
    # ===== COORDINATION PROTOCOLS =====
    
    async def _create_coordination_event(
        self,
        event_type: str,
        source_agent: str,
        target_agents: List[str],
        message: Dict[str, Any],
        protocol: CoordinationProtocol,
        response_required: bool = False
    ) -> str:
        """Create coordination event"""
        
        event_id = f"event_{uuid.uuid4().hex[:12]}"
        
        event = CoordinationEvent(
            event_id=event_id,
            event_type=event_type,
            source_agent=source_agent,
            target_agents=target_agents,
            message=message,
            timestamp=datetime.utcnow(),
            protocol=protocol,
            response_required=response_required,
            responses={}
        )
        
        self.coordination_events[event_id] = event
        
        # Publish coordination event
        await shared_pulsar_client.publish(
            "q.agents.coordination.event",
            {
                "event_id": event_id,
                "event_type": event_type,
                "source_agent": source_agent,
                "target_agents": target_agents,
                "message": message,
                "protocol": protocol.value,
                "response_required": response_required,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return event_id
    
    async def send_message(
        self,
        source_agent: str,
        target_agents: List[str],
        message: Dict[str, Any],
        protocol: CoordinationProtocol = CoordinationProtocol.DECENTRALIZED
    ) -> str:
        """Send message between agents"""
        
        return await self._create_coordination_event(
            event_type="message",
            source_agent=source_agent,
            target_agents=target_agents,
            message=message,
            protocol=protocol
        )
    
    async def broadcast_message(
        self,
        source_agent: str,
        message: Dict[str, Any],
        channel: str = "general"
    ) -> str:
        """Broadcast message to all agents in a channel"""
        
        # Get agents in the channel
        target_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if channel in agent.communication_channels
        ]
        
        return await self._create_coordination_event(
            event_type="broadcast",
            source_agent=source_agent,
            target_agents=target_agents,
            message={"channel": channel, "content": message},
            protocol=CoordinationProtocol.DECENTRALIZED
        )
    
    async def negotiate_task_allocation(
        self,
        task_id: str,
        candidate_agents: List[str]
    ) -> Dict[str, Any]:
        """Negotiate task allocation using auction mechanism"""
        
        task = self.tasks.get(task_id)
        if not task:
            return {}
        
        # Create auction event
        auction_event = await self._create_coordination_event(
            event_type="auction",
            source_agent="coordinator",
            target_agents=candidate_agents,
            message={
                "task_id": task_id,
                "task_description": task.description,
                "required_capabilities": task.required_capabilities,
                "deadline": task.deadline.isoformat() if task.deadline else None,
                "bid_deadline": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
            },
            protocol=CoordinationProtocol.AUCTION,
            response_required=True
        )
        
        # Wait for bids (simplified - in real implementation, this would be event-driven)
        await asyncio.sleep(5)
        
        # Process bids
        bids = self.coordination_events[auction_event].responses
        
        if not bids:
            return {"status": "no_bids", "winner": None}
        
        # Select winner based on best bid
        best_bid = min(bids.items(), key=lambda x: x[1].get("cost", float('inf')))
        winner_agent = best_bid[0]
        
        return {
            "status": "auction_complete",
            "winner": winner_agent,
            "winning_bid": best_bid[1],
            "total_bids": len(bids)
        }
    
    # ===== COALITION FORMATION =====
    
    async def form_coalition(
        self,
        coalition_name: str,
        objective: str,
        required_capabilities: Dict[str, float],
        max_size: int = 5
    ) -> str:
        """Form a coalition of agents"""
        
        coalition_id = f"coalition_{uuid.uuid4().hex[:12]}"
        
        # Find suitable agents
        suitable_agents = []
        for agent_id, agent in self.agents.items():
            if agent.current_state == AgentState.ACTIVE:
                capability_match = all(
                    agent.capabilities.get(cap, 0.0) >= level
                    for cap, level in required_capabilities.items()
                )
                if capability_match:
                    suitable_agents.append(agent_id)
        
        # Select coalition members
        selected_agents = suitable_agents[:max_size]
        
        if not selected_agents:
            logger.warning(f"No suitable agents found for coalition: {coalition_name}")
            return None
        
        # Select leader (agent with highest overall capability)
        leader_agent = max(selected_agents, key=lambda aid: sum(self.agents[aid].capabilities.values()))
        
        # Create coalition
        coalition = Coalition(
            coalition_id=coalition_id,
            coalition_name=coalition_name,
            member_agents=selected_agents,
            leader_agent=leader_agent,
            objective=objective,
            formation_time=datetime.utcnow(),
            expected_duration=3600,
            performance_metrics={
                "tasks_completed": 0,
                "success_rate": 0.0,
                "efficiency": 0.0
            },
            communication_protocol=CoordinationProtocol.HIERARCHICAL,
            status="active"
        )
        
        self.coalitions[coalition_id] = coalition
        
        # Notify agents of coalition formation
        await self._create_coordination_event(
            event_type="coalition_formation",
            source_agent="coordinator",
            target_agents=selected_agents,
            message={
                "coalition_id": coalition_id,
                "coalition_name": coalition_name,
                "objective": objective,
                "leader_agent": leader_agent,
                "members": selected_agents
            },
            protocol=CoordinationProtocol.HIERARCHICAL
        )
        
        logger.info(f"Formed coalition: {coalition_name} with {len(selected_agents)} agents")
        return coalition_id
    
    async def disband_coalition(self, coalition_id: str) -> bool:
        """Disband a coalition"""
        
        coalition = self.coalitions.get(coalition_id)
        if not coalition:
            return False
        
        # Notify agents
        await self._create_coordination_event(
            event_type="coalition_disbandment",
            source_agent="coordinator",
            target_agents=coalition.member_agents,
            message={
                "coalition_id": coalition_id,
                "coalition_name": coalition.coalition_name,
                "reason": "objective_completed"
            },
            protocol=CoordinationProtocol.HIERARCHICAL
        )
        
        # Remove coalition
        coalition.status = "disbanded"
        
        logger.info(f"Disbanded coalition: {coalition_id}")
        return True
    
    # ===== MULTI-AGENT REINFORCEMENT LEARNING =====
    
    async def add_experience(
        self,
        agent_id: str,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        joint_action: Optional[np.ndarray] = None,
        team_reward: Optional[float] = None
    ):
        """Add experience to replay buffer"""
        
        experience = Experience(
            agent_id=agent_id,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            joint_action=joint_action,
            team_reward=team_reward,
            timestamp=datetime.utcnow()
        )
        
        self.experience_buffer.append(experience)
        self.shared_buffer.append(experience)
        
        # Update agent learning progress
        if agent_id in self.agents:
            self.agents[agent_id].learning_progress["episodes"] += 1
            self.agents[agent_id].learning_progress["total_reward"] += reward
            
            episodes = self.agents[agent_id].learning_progress["episodes"]
            total_reward = self.agents[agent_id].learning_progress["total_reward"]
            self.agents[agent_id].learning_progress["average_reward"] = total_reward / episodes
    
    async def get_action(self, agent_id: str, state: np.ndarray) -> int:
        """Get action from agent's policy"""
        
        if agent_id not in self.actors:
            return random.randint(0, self.action_dim - 1)
        
        actor = self.actors[agent_id]
        
        # Add exploration noise
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_values = actor(state_tensor)
            
            # Add noise for exploration
            noise = torch.randn_like(action_values) * self.config["exploration_noise"]
            action_values += noise
            
            # Get discrete action
            action = torch.argmax(action_values).item()
        
        return action
    
    async def train_agents(self):
        """Train agents using MADDPG algorithm"""
        
        if len(self.experience_buffer) < self.config["batch_size"]:
            return
        
        # Sample batch
        batch = random.sample(self.experience_buffer, self.config["batch_size"])
        
        # Prepare batch data
        states = torch.FloatTensor([exp.state for exp in batch])
        actions = torch.LongTensor([exp.action for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch])
        next_states = torch.FloatTensor([exp.next_state for exp in batch])
        dones = torch.BoolTensor([exp.done for exp in batch])
        
        # Train each agent
        for agent_id in self.actors.keys():
            if agent_id not in self.agents:
                continue
            
            actor = self.actors[agent_id]
            critic = self.critics[agent_id]
            target_actor = self.target_actors[agent_id]
            target_critic = self.target_critics[agent_id]
            actor_optimizer = self.actor_optimizers[agent_id]
            critic_optimizer = self.critic_optimizers[agent_id]
            
            # Update critic
            with torch.no_grad():
                next_actions = target_actor(next_states)
                target_q = target_critic(next_states, next_actions)
                target_q = rewards + (self.config["gamma"] * target_q * (~dones))
            
            current_q = critic(states, actions.unsqueeze(1).float())
            critic_loss = F.mse_loss(current_q, target_q.unsqueeze(1))
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            # Update actor
            predicted_actions = actor(states)
            actor_loss = -critic(states, predicted_actions).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            # Update target networks
            self._soft_update(target_actor, actor, self.config["tau"])
            self._soft_update(target_critic, critic, self.config["tau"])
    
    def _soft_update(self, target, source, tau):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def _initialize_target_networks(self):
        """Initialize target networks for QMIX"""
        self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
    
    async def train_qmix(self):
        """Train QMIX network"""
        
        if len(self.shared_buffer) < self.config["batch_size"]:
            return
        
        # Sample batch from shared buffer
        batch = random.sample(self.shared_buffer, self.config["batch_size"])
        
        # Group experiences by episode/timestamp
        episodes = defaultdict(list)
        for exp in batch:
            episode_key = int(exp.timestamp.timestamp() / 60)  # Group by minute
            episodes[episode_key].append(exp)
        
        # Train on episodes with team rewards
        for episode_experiences in episodes.values():
            if len(episode_experiences) < 2:  # Need at least 2 agents
                continue
            
            # Get team rewards
            team_rewards = [exp.team_reward for exp in episode_experiences if exp.team_reward is not None]
            if not team_rewards:
                continue
            
            # Prepare data for QMIX
            agent_q_values = torch.zeros(len(episode_experiences), 1)
            for i, exp in enumerate(episode_experiences):
                if exp.agent_id in self.critics:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(exp.state).unsqueeze(0)
                        action_tensor = torch.LongTensor([exp.action]).unsqueeze(0).float()
                        q_value = self.critics[exp.agent_id](state_tensor, action_tensor)
                        agent_q_values[i] = q_value
            
            # Get global state (simplified as mean of agent states)
            global_state = torch.mean(torch.stack([torch.FloatTensor(exp.state) for exp in episode_experiences]), dim=0)
            
            # Forward pass through QMIX
            q_total = self.qmix_net(agent_q_values.unsqueeze(0), global_state.unsqueeze(0))
            
            # Calculate loss
            target_total = torch.FloatTensor([np.mean(team_rewards)])
            loss = F.mse_loss(q_total, target_total)
            
            # Update QMIX network
            self.qmix_optimizer.zero_grad()
            loss.backward()
            self.qmix_optimizer.step()
    
    async def _update_agent_performance(self, agent_id: str, success: bool):
        """Update agent performance metrics"""
        
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        # Update success rate
        current_success_rate = agent.performance_metrics.get("success_rate", 0.0)
        total_tasks = agent.performance_metrics.get("total_tasks", 0)
        
        if success:
            new_success_rate = (current_success_rate * total_tasks + 1.0) / (total_tasks + 1)
        else:
            new_success_rate = (current_success_rate * total_tasks) / (total_tasks + 1)
        
        agent.performance_metrics["success_rate"] = new_success_rate
        agent.performance_metrics["total_tasks"] = total_tasks + 1
        
        # Update collaboration score (simplified)
        agent.performance_metrics["collaboration_score"] = min(1.0, agent.performance_metrics["collaboration_score"] + 0.1)
    
    # ===== BACKGROUND TASKS =====
    
    async def _coordination_loop(self):
        """Main coordination loop"""
        
        while True:
            try:
                # Process pending tasks
                pending_tasks = [task for task in self.tasks.values() if task.status == "pending"]
                
                for task in pending_tasks:
                    await self._allocate_task(task.task_id)
                
                # Process coordination events
                active_events = [event for event in self.coordination_events.values() 
                               if event.response_required and len(event.responses) == 0]
                
                for event in active_events:
                    # Check if event has timed out
                    if (datetime.utcnow() - event.timestamp).total_seconds() > 300:  # 5 minutes
                        logger.warning(f"Coordination event {event.event_id} timed out")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(10)
    
    async def _learning_loop(self):
        """Main learning loop"""
        
        while True:
            try:
                # Train agents
                await self.train_agents()
                
                # Train QMIX
                await self.train_qmix()
                
                # Update exploration parameters
                for agent_id in self.agents:
                    if agent_id in self.agents:
                        epsilon = self.agents[agent_id].learning_progress.get("epsilon", 1.0)
                        epsilon = max(0.1, epsilon * 0.995)  # Decay epsilon
                        self.agents[agent_id].learning_progress["epsilon"] = epsilon
                
                await asyncio.sleep(self.config["update_frequency"])
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(self.config["update_frequency"])
    
    async def _coalition_management(self):
        """Manage coalition lifecycle"""
        
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Check for coalitions that need to be disbanded
                for coalition_id, coalition in list(self.coalitions.items()):
                    if coalition.status != "active":
                        continue
                    
                    # Check if coalition has exceeded expected duration
                    if (current_time - coalition.formation_time).total_seconds() > coalition.expected_duration:
                        await self.disband_coalition(coalition_id)
                        continue
                    
                    # Check if coalition is still effective
                    if coalition.performance_metrics.get("efficiency", 0.0) < 0.3:
                        logger.info(f"Coalition {coalition_id} underperforming, considering disbandment")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in coalition management: {e}")
                await asyncio.sleep(300)
    
    async def _metrics_tracking(self):
        """Track coordination metrics"""
        
        while True:
            try:
                # Update metrics
                self.coordination_metrics["total_agents"] = len(self.agents)
                self.coordination_metrics["active_agents"] = len([
                    a for a in self.agents.values() if a.current_state == AgentState.ACTIVE
                ])
                self.coordination_metrics["completed_tasks"] = len([
                    t for t in self.tasks.values() if t.status == "completed"
                ])
                self.coordination_metrics["active_coalitions"] = len([
                    c for c in self.coalitions.values() if c.status == "active"
                ])
                
                # Calculate average learning progress
                if self.agents:
                    avg_reward = np.mean([
                        agent.learning_progress.get("average_reward", 0.0)
                        for agent in self.agents.values()
                    ])
                    self.coordination_metrics["learning_progress"] = avg_reward
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in metrics tracking: {e}")
                await asyncio.sleep(60)
    
    # ===== KNOWLEDGEGRAPH INTEGRATION =====
    
    async def _store_agent_in_kg(self, agent_profile: AgentProfile):
        """Store agent profile in KnowledgeGraph"""
        
        try:
            vertex_data = {
                "agent_id": agent_profile.agent_id,
                "agent_name": agent_profile.agent_name,
                "agent_type": agent_profile.agent_type,
                "role": agent_profile.role.value,
                "capabilities": agent_profile.capabilities,
                "preferences": agent_profile.preferences,
                "current_state": agent_profile.current_state.value,
                "performance_metrics": agent_profile.performance_metrics,
                "learning_progress": agent_profile.learning_progress,
                "communication_channels": agent_profile.communication_channels,
                "created_at": agent_profile.created_at.isoformat(),
                "last_active": agent_profile.last_active.isoformat(),
                "metadata": agent_profile.metadata
            }
            
            await self.kg_client.add_vertex(
                "Agent", 
                agent_profile.agent_id, 
                vertex_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store agent in KnowledgeGraph: {e}")
    
    async def _store_task_in_kg(self, task: Task):
        """Store task in KnowledgeGraph"""
        
        try:
            vertex_data = {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "task_type": task.task_type,
                "description": task.description,
                "priority": task.priority.value,
                "required_capabilities": task.required_capabilities,
                "estimated_duration": task.estimated_duration,
                "deadline": task.deadline.isoformat() if task.deadline else None,
                "dependencies": task.dependencies,
                "assigned_agents": task.assigned_agents,
                "status": task.status,
                "created_at": task.created_at.isoformat(),
                "created_by": task.created_by,
                "metadata": task.metadata
            }
            
            await self.kg_client.add_vertex(
                "Task", 
                task.task_id, 
                vertex_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store task in KnowledgeGraph: {e}")
    
    # ===== UTILITY METHODS =====
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for coordination"""
        
        topics = [
            "q.agents.coordination.agent.registered",
            "q.agents.coordination.agent.state_updated",
            "q.agents.coordination.task.created",
            "q.agents.coordination.task.completed",
            "q.agents.coordination.event",
            "q.agents.coordination.coalition.formed",
            "q.agents.coordination.coalition.disbanded"
        ]
        
        logger.info("Multi-agent coordination Pulsar topics configured")
    
    # ===== PUBLIC API =====
    
    async def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination metrics"""
        
        return {
            "coordination_metrics": self.coordination_metrics,
            "config": self.config,
            "active_events": len([e for e in self.coordination_events.values() if e.response_required]),
            "experience_buffer_size": len(self.experience_buffer),
            "shared_buffer_size": len(self.shared_buffer)
        }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "status": task.status,
            "assigned_agents": task.assigned_agents,
            "created_at": task.created_at.isoformat(),
            "deadline": task.deadline.isoformat() if task.deadline else None,
            "metadata": task.metadata
        }
    
    async def get_coalition_status(self, coalition_id: str) -> Optional[Dict[str, Any]]:
        """Get coalition status"""
        
        coalition = self.coalitions.get(coalition_id)
        if not coalition:
            return None
        
        return {
            "coalition_id": coalition.coalition_id,
            "coalition_name": coalition.coalition_name,
            "member_agents": coalition.member_agents,
            "leader_agent": coalition.leader_agent,
            "status": coalition.status,
            "formation_time": coalition.formation_time.isoformat(),
            "performance_metrics": coalition.performance_metrics
        }

# Global instance
multi_agent_coordinator = MultiAgentCoordinator() 