"""
Reinforcement Learning Service

This service develops RL agents that learn from workflow execution outcomes:
- Creates RL environments based on workflow states and outcomes
- Trains agents using various RL algorithms (PPO, A3C, DQN)
- Tracks reward functions based on workflow success metrics
- Implements multi-agent reinforcement learning
- Integrates with existing workflow engine for continuous learning
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
from pathlib import Path
import pickle

# RL Libraries
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import tensorboard

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.q_workflow_schemas.models import Workflow, WorkflowStep, WorkflowStatus
from shared.q_vectorstore_client.client import VectorStoreClient
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)

class RLAlgorithm(Enum):
    """Supported RL algorithms"""
    PPO = "ppo"
    A2C = "a2c"
    DQN = "dqn"
    DDPG = "ddpg"
    SAC = "sac"
    TD3 = "td3"

class RLEnvironmentType(Enum):
    """Types of RL environments"""
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    AGENT_COORDINATION = "agent_coordination"
    RESOURCE_ALLOCATION = "resource_allocation"
    TASK_SCHEDULING = "task_scheduling"
    MULTI_AGENT_COLLABORATION = "multi_agent_collaboration"

class RLTrainingStatus(Enum):
    """RL training status"""
    INITIALIZING = "initializing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class RLTrainingSession:
    """RL training session"""
    session_id: str
    agent_name: str
    environment_type: RLEnvironmentType
    algorithm: RLAlgorithm
    training_config: Dict[str, Any]
    environment_config: Dict[str, Any]
    status: RLTrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_timesteps: int = 0
    current_timesteps: int = 0
    best_reward: float = float('-inf')
    current_reward: float = 0.0
    training_metrics: Dict[str, List[float]] = None
    model_path: Optional[str] = None

@dataclass
class WorkflowExperience:
    """Experience from workflow execution"""
    workflow_id: str
    agent_id: str
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    done: bool
    timestamp: datetime
    workflow_metadata: Dict[str, Any]

@dataclass
class RLAgent:
    """RL agent definition"""
    agent_id: str
    agent_name: str
    algorithm: RLAlgorithm
    environment_type: RLEnvironmentType
    model_path: str
    training_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    last_updated: datetime
    active: bool = True

class WorkflowEnvironment(gym.Env):
    """Custom Gym environment for workflow optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.max_steps = config.get("max_steps", 100)
        self.current_step = 0
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(config.get("num_actions", 10))
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(config.get("state_size", 20),), dtype=np.float32
        )
        
        # Workflow state
        self.workflow_state = {}
        self.workflow_history = []
        self.reward_components = {
            "success": 0.0,
            "efficiency": 0.0,
            "resource_usage": 0.0,
            "time_to_completion": 0.0
        }
        
        # Experience replay
        self.experience_buffer = deque(maxlen=10000)
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.workflow_state = self._generate_initial_state()
        self.workflow_history = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        
        # Execute action
        next_state = self._execute_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(action, next_state)
        
        # Update state
        self.workflow_state = next_state
        self.current_step += 1
        
        # Check if done
        done = self._is_done()
        
        # Store experience
        self.workflow_history.append({
            "step": self.current_step,
            "action": action,
            "state": self.workflow_state.copy(),
            "reward": reward
        })
        
        info = {
            "workflow_state": self.workflow_state,
            "reward_components": self.reward_components
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _generate_initial_state(self) -> Dict[str, Any]:
        """Generate initial workflow state"""
        
        return {
            "workflow_progress": 0.0,
            "resource_utilization": np.random.uniform(0.3, 0.7),
            "agent_availability": np.random.uniform(0.5, 1.0),
            "task_complexity": np.random.uniform(0.2, 0.8),
            "time_pressure": np.random.uniform(0.1, 0.9),
            "error_rate": np.random.uniform(0.0, 0.2),
            "collaboration_score": np.random.uniform(0.4, 0.9),
            "workflow_type": np.random.choice([0, 1, 2, 3, 4])  # Different workflow types
        }
    
    def _execute_action(self, action: int) -> Dict[str, Any]:
        """Execute action and return next state"""
        
        current_state = self.workflow_state.copy()
        
        # Action mapping
        action_effects = {
            0: {"workflow_progress": 0.1, "resource_utilization": 0.05},  # Accelerate
            1: {"workflow_progress": 0.05, "resource_utilization": -0.05},  # Optimize
            2: {"collaboration_score": 0.1, "agent_availability": 0.05},  # Collaborate
            3: {"error_rate": -0.05, "task_complexity": -0.05},  # Simplify
            4: {"time_pressure": -0.1, "workflow_progress": -0.02},  # Delay
            5: {"resource_utilization": 0.1, "workflow_progress": 0.15},  # Resource boost
            6: {"collaboration_score": 0.15, "workflow_progress": 0.08},  # Team focus
            7: {"error_rate": -0.1, "workflow_progress": 0.05},  # Quality focus
            8: {"agent_availability": 0.1, "resource_utilization": 0.08},  # Scale up
            9: {"workflow_progress": 0.12, "time_pressure": 0.05}  # Push deadline
        }
        
        # Apply action effects
        effects = action_effects.get(action, {})
        for key, change in effects.items():
            if key in current_state:
                current_state[key] = np.clip(current_state[key] + change, 0.0, 1.0)
        
        # Add some randomness
        for key in current_state:
            if key != "workflow_type":
                noise = np.random.normal(0, 0.02)
                current_state[key] = np.clip(current_state[key] + noise, 0.0, 1.0)
        
        return current_state
    
    def _calculate_reward(self, action: int, next_state: Dict[str, Any]) -> float:
        """Calculate reward based on action and resulting state"""
        
        # Reward components
        success_reward = next_state["workflow_progress"] * 10
        efficiency_reward = (1.0 - next_state["resource_utilization"]) * 2
        quality_reward = (1.0 - next_state["error_rate"]) * 3
        collaboration_reward = next_state["collaboration_score"] * 2
        time_penalty = next_state["time_pressure"] * -1
        
        # Completion bonus
        completion_bonus = 0
        if next_state["workflow_progress"] >= 0.95:
            completion_bonus = 50
        
        # Total reward
        reward = (success_reward + efficiency_reward + quality_reward + 
                 collaboration_reward + time_penalty + completion_bonus)
        
        # Store reward components
        self.reward_components = {
            "success": success_reward,
            "efficiency": efficiency_reward,
            "quality": quality_reward,
            "collaboration": collaboration_reward,
            "time_penalty": time_penalty,
            "completion_bonus": completion_bonus
        }
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        
        # Done if workflow completed or max steps reached
        workflow_complete = self.workflow_state["workflow_progress"] >= 0.95
        max_steps_reached = self.current_step >= self.max_steps
        workflow_failed = self.workflow_state["error_rate"] >= 0.8
        
        return workflow_complete or max_steps_reached or workflow_failed
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        
        obs = np.array([
            self.workflow_state["workflow_progress"],
            self.workflow_state["resource_utilization"],
            self.workflow_state["agent_availability"],
            self.workflow_state["task_complexity"],
            self.workflow_state["time_pressure"],
            self.workflow_state["error_rate"],
            self.workflow_state["collaboration_score"],
            self.workflow_state["workflow_type"] / 4.0,  # Normalize
            self.current_step / self.max_steps,  # Progress
            # Add some derived features
            self.workflow_state["workflow_progress"] * self.workflow_state["collaboration_score"],
            self.workflow_state["resource_utilization"] * self.workflow_state["agent_availability"],
            1.0 - self.workflow_state["error_rate"],  # Success rate
            np.mean(list(self.workflow_state.values())[:-1]),  # Average state
            # Add more features to reach state_size
            *([0.0] * 7)  # Padding
        ], dtype=np.float32)
        
        return obs

class RLTrainingCallback(BaseCallback):
    """Custom callback for RL training"""
    
    def __init__(self, rl_service, session_id: str):
        super().__init__()
        self.rl_service = rl_service
        self.session_id = session_id
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """Called at each step"""
        
        # Update session metrics
        if self.session_id in self.rl_service.active_training_sessions:
            session = self.rl_service.active_training_sessions[self.session_id]
            session.current_timesteps = self.num_timesteps
            session.current_reward = self.training_env.get_attr("reward")[0] if hasattr(self.training_env, 'get_attr') else 0.0
        
        return True
    
    def _on_episode_end(self) -> None:
        """Called at the end of each episode"""
        
        # Log episode metrics
        if hasattr(self.locals, 'episode_rewards'):
            self.episode_rewards.append(self.locals['episode_rewards'])
        
        if hasattr(self.locals, 'episode_lengths'):
            self.episode_lengths.append(self.locals['episode_lengths'])

class ReinforcementLearningService:
    """
    Reinforcement Learning Service for Q Platform
    """
    
    def __init__(self, 
                 model_storage_path: str = "models/rl",
                 tensorboard_log_dir: str = "logs/rl"):
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.tensorboard_log_dir = Path(tensorboard_log_dir)
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Active training sessions
        self.active_training_sessions: Dict[str, RLTrainingSession] = {}
        self.trained_agents: Dict[str, RLAgent] = {}
        
        # Experience collection
        self.workflow_experiences: Dict[str, List[WorkflowExperience]] = defaultdict(list)
        self.experience_buffer = deque(maxlen=50000)
        
        # Environment registry
        self.environment_registry: Dict[str, gym.Env] = {}
        
        # Configuration
        self.config = {
            "default_timesteps": 100000,
            "evaluation_episodes": 10,
            "save_frequency": 10000,
            "experience_batch_size": 256,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95
        }
        
        # Performance tracking
        self.rl_metrics = {
            "agents_trained": 0,
            "total_episodes": 0,
            "average_reward": 0.0,
            "best_reward": float('-inf'),
            "training_time": 0.0,
            "successful_workflows": 0
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
    async def initialize(self):
        """Initialize the RL service"""
        logger.info("Initializing Reinforcement Learning Service")
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        # Register environment types
        await self._register_environments()
        
        # Load existing agents
        await self._load_trained_agents()
        
        # Start background tasks
        self.background_tasks.add(asyncio.create_task(self._experience_collection_loop()))
        self.background_tasks.add(asyncio.create_task(self._training_monitor()))
        self.background_tasks.add(asyncio.create_task(self._performance_tracking()))
        
        logger.info("Reinforcement Learning Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the RL service"""
        logger.info("Shutting down Reinforcement Learning Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save trained agents
        await self._save_trained_agents()
        
        logger.info("Reinforcement Learning Service shut down successfully")
    
    # ===== TRAINING MANAGEMENT =====
    
    async def start_rl_training(
        self,
        agent_name: str,
        environment_type: RLEnvironmentType,
        algorithm: RLAlgorithm = RLAlgorithm.PPO,
        training_config: Optional[Dict[str, Any]] = None,
        environment_config: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 100000
    ) -> str:
        """
        Start RL training session
        
        Args:
            agent_name: Name of the agent to train
            environment_type: Type of environment
            algorithm: RL algorithm to use
            training_config: Training configuration
            environment_config: Environment configuration
            total_timesteps: Total training timesteps
            
        Returns:
            Session ID
        """
        session_id = f"rl_training_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Starting RL training session: {agent_name}")
        
        # Create training session
        session = RLTrainingSession(
            session_id=session_id,
            agent_name=agent_name,
            environment_type=environment_type,
            algorithm=algorithm,
            training_config=training_config or {},
            environment_config=environment_config or {},
            status=RLTrainingStatus.INITIALIZING,
            created_at=datetime.utcnow(),
            total_timesteps=total_timesteps,
            training_metrics=defaultdict(list)
        )
        
        self.active_training_sessions[session_id] = session
        
        # Start training in background
        asyncio.create_task(self._run_rl_training(session))
        
        # Publish training started event
        await shared_pulsar_client.publish(
            "q.ml.rl.training.started",
            {
                "session_id": session_id,
                "agent_name": agent_name,
                "environment_type": environment_type.value,
                "algorithm": algorithm.value,
                "total_timesteps": total_timesteps,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return session_id
    
    async def _run_rl_training(self, session: RLTrainingSession):
        """Run RL training session"""
        
        try:
            session.status = RLTrainingStatus.TRAINING
            session.started_at = datetime.utcnow()
            
            logger.info(f"Running RL training: {session.agent_name}")
            
            # Create environment
            env = self._create_environment(session.environment_type, session.environment_config)
            
            # Create vectorized environment
            vec_env = make_vec_env(
                lambda: env,
                n_envs=session.training_config.get("n_envs", 4)
            )
            
            # Create RL model
            model = self._create_rl_model(session.algorithm, vec_env, session.training_config)
            
            # Create callback
            callback = RLTrainingCallback(self, session.session_id)
            
            # Train model
            model.learn(
                total_timesteps=session.total_timesteps,
                callback=callback,
                tb_log_name=f"{session.agent_name}_{session.session_id}"
            )
            
            # Save model
            model_path = self.model_storage_path / f"{session.agent_name}_{session.session_id}.zip"
            model.save(str(model_path))
            session.model_path = str(model_path)
            
            # Evaluate model
            evaluation_results = await self._evaluate_model(model, env, session.training_config)
            
            # Create trained agent
            trained_agent = RLAgent(
                agent_id=f"rl_agent_{uuid.uuid4().hex[:8]}",
                agent_name=session.agent_name,
                algorithm=session.algorithm,
                environment_type=session.environment_type,
                model_path=session.model_path,
                training_history=[{
                    "session_id": session.session_id,
                    "training_timesteps": session.total_timesteps,
                    "final_reward": evaluation_results.get("mean_reward", 0.0),
                    "training_time": (datetime.utcnow() - session.started_at).total_seconds()
                }],
                performance_metrics=evaluation_results,
                last_updated=datetime.utcnow()
            )
            
            self.trained_agents[trained_agent.agent_id] = trained_agent
            
            # Complete session
            session.status = RLTrainingStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            session.best_reward = evaluation_results.get("mean_reward", 0.0)
            
            # Update metrics
            self.rl_metrics["agents_trained"] += 1
            
            logger.info(f"Completed RL training: {session.agent_name}")
            
            # Publish training completed event
            await shared_pulsar_client.publish(
                "q.ml.rl.training.completed",
                {
                    "session_id": session.session_id,
                    "agent_id": trained_agent.agent_id,
                    "final_reward": session.best_reward,
                    "training_time": (session.completed_at - session.started_at).total_seconds(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"RL training failed: {session.agent_name}: {e}")
            session.status = RLTrainingStatus.FAILED
            session.completed_at = datetime.utcnow()
    
    def _create_environment(
        self,
        environment_type: RLEnvironmentType,
        config: Dict[str, Any]
    ) -> gym.Env:
        """Create RL environment"""
        
        if environment_type == RLEnvironmentType.WORKFLOW_OPTIMIZATION:
            return WorkflowEnvironment(config)
        elif environment_type == RLEnvironmentType.AGENT_COORDINATION:
            return self._create_coordination_environment(config)
        elif environment_type == RLEnvironmentType.RESOURCE_ALLOCATION:
            return self._create_resource_environment(config)
        else:
            # Default to workflow environment
            return WorkflowEnvironment(config)
    
    def _create_coordination_environment(self, config: Dict[str, Any]) -> gym.Env:
        """Create agent coordination environment"""
        
        # Simplified coordination environment
        return WorkflowEnvironment(config)  # For now, use workflow environment
    
    def _create_resource_environment(self, config: Dict[str, Any]) -> gym.Env:
        """Create resource allocation environment"""
        
        # Simplified resource environment
        return WorkflowEnvironment(config)  # For now, use workflow environment
    
    def _create_rl_model(
        self,
        algorithm: RLAlgorithm,
        env: VecEnv,
        config: Dict[str, Any]
    ):
        """Create RL model"""
        
        model_config = {
            "learning_rate": config.get("learning_rate", self.config["learning_rate"]),
            "gamma": config.get("gamma", self.config["gamma"]),
            "verbose": 1,
            "tensorboard_log": str(self.tensorboard_log_dir)
        }
        
        if algorithm == RLAlgorithm.PPO:
            model_config.update({
                "gae_lambda": config.get("gae_lambda", self.config["gae_lambda"]),
                "n_steps": config.get("n_steps", 2048),
                "batch_size": config.get("batch_size", 64),
                "n_epochs": config.get("n_epochs", 10)
            })
            return PPO("MlpPolicy", env, **model_config)
        
        elif algorithm == RLAlgorithm.A2C:
            model_config.update({
                "gae_lambda": config.get("gae_lambda", self.config["gae_lambda"]),
                "n_steps": config.get("n_steps", 5)
            })
            return A2C("MlpPolicy", env, **model_config)
        
        elif algorithm == RLAlgorithm.DQN:
            model_config.update({
                "buffer_size": config.get("buffer_size", 100000),
                "learning_starts": config.get("learning_starts", 1000),
                "target_update_interval": config.get("target_update_interval", 1000)
            })
            return DQN("MlpPolicy", env, **model_config)
        
        else:
            # Default to PPO
            return PPO("MlpPolicy", env, **model_config)
    
    async def _evaluate_model(
        self,
        model,
        env: gym.Env,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate trained model"""
        
        n_episodes = config.get("evaluation_episodes", self.config["evaluation_episodes"])
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if truncated:
                    done = True
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths)
        }
    
    # ===== EXPERIENCE COLLECTION =====
    
    async def collect_workflow_experience(
        self,
        workflow_id: str,
        agent_id: str,
        workflow_data: Dict[str, Any]
    ):
        """Collect experience from workflow execution"""
        
        # Extract state, action, reward from workflow data
        state = self._extract_workflow_state(workflow_data)
        action = self._extract_workflow_action(workflow_data)
        reward = self._calculate_workflow_reward(workflow_data)
        next_state = self._extract_next_workflow_state(workflow_data)
        done = workflow_data.get("status") in ["completed", "failed"]
        
        # Create experience
        experience = WorkflowExperience(
            workflow_id=workflow_id,
            agent_id=agent_id,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            timestamp=datetime.utcnow(),
            workflow_metadata=workflow_data
        )
        
        # Store experience
        self.workflow_experiences[workflow_id].append(experience)
        self.experience_buffer.append(experience)
        
        # Update metrics
        if done and workflow_data.get("status") == "completed":
            self.rl_metrics["successful_workflows"] += 1
        
        logger.debug(f"Collected experience for workflow {workflow_id}")
    
    def _extract_workflow_state(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract state representation from workflow data"""
        
        return {
            "workflow_progress": workflow_data.get("progress", 0.0),
            "resource_utilization": workflow_data.get("resource_usage", 0.5),
            "agent_availability": workflow_data.get("agent_availability", 0.8),
            "task_complexity": workflow_data.get("complexity", 0.5),
            "time_pressure": workflow_data.get("urgency", 0.3),
            "error_rate": workflow_data.get("error_rate", 0.0),
            "collaboration_score": workflow_data.get("collaboration", 0.7)
        }
    
    def _extract_workflow_action(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract action from workflow data"""
        
        return {
            "action_type": workflow_data.get("last_action", "continue"),
            "agent_assignment": workflow_data.get("assigned_agents", []),
            "resource_allocation": workflow_data.get("resources", {}),
            "priority": workflow_data.get("priority", "medium")
        }
    
    def _calculate_workflow_reward(self, workflow_data: Dict[str, Any]) -> float:
        """Calculate reward based on workflow outcome"""
        
        base_reward = 0.0
        
        # Success reward
        if workflow_data.get("status") == "completed":
            base_reward += 10.0
        elif workflow_data.get("status") == "failed":
            base_reward -= 5.0
        
        # Efficiency reward
        efficiency = workflow_data.get("efficiency", 0.5)
        base_reward += efficiency * 5.0
        
        # Time penalty
        time_taken = workflow_data.get("execution_time", 0)
        expected_time = workflow_data.get("expected_time", 1)
        if expected_time > 0:
            time_ratio = time_taken / expected_time
            if time_ratio > 1.2:  # 20% over expected
                base_reward -= (time_ratio - 1.0) * 3.0
        
        # Quality reward
        quality = workflow_data.get("quality_score", 0.8)
        base_reward += quality * 3.0
        
        return base_reward
    
    def _extract_next_workflow_state(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract next state from workflow data"""
        
        # For now, return the current state
        # In a real implementation, this would be the state after the action
        return self._extract_workflow_state(workflow_data)
    
    # ===== AGENT DEPLOYMENT =====
    
    async def deploy_rl_agent(
        self,
        agent_id: str,
        target_environment: str = "production"
    ) -> bool:
        """Deploy trained RL agent to production"""
        
        if agent_id not in self.trained_agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        agent = self.trained_agents[agent_id]
        
        try:
            # Load model
            if agent.algorithm == RLAlgorithm.PPO:
                model = PPO.load(agent.model_path)
            elif agent.algorithm == RLAlgorithm.A2C:
                model = A2C.load(agent.model_path)
            elif agent.algorithm == RLAlgorithm.DQN:
                model = DQN.load(agent.model_path)
            else:
                logger.error(f"Unsupported algorithm: {agent.algorithm}")
                return False
            
            # Register with agent registry
            await self._register_rl_agent(agent, model)
            
            # Publish deployment event
            await shared_pulsar_client.publish(
                "q.ml.rl.agent.deployed",
                {
                    "agent_id": agent_id,
                    "agent_name": agent.agent_name,
                    "environment": target_environment,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Deployed RL agent: {agent.agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy agent {agent_id}: {e}")
            return False
    
    async def _register_rl_agent(self, agent: RLAgent, model):
        """Register RL agent with the system"""
        
        # This would integrate with the existing agent registry
        # For now, just log the registration
        logger.info(f"Registered RL agent: {agent.agent_name}")
    
    async def get_rl_agent_action(
        self,
        agent_id: str,
        state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get action from RL agent for given state"""
        
        if agent_id not in self.trained_agents:
            return None
        
        agent = self.trained_agents[agent_id]
        
        try:
            # Load model
            if agent.algorithm == RLAlgorithm.PPO:
                model = PPO.load(agent.model_path)
            elif agent.algorithm == RLAlgorithm.A2C:
                model = A2C.load(agent.model_path)
            elif agent.algorithm == RLAlgorithm.DQN:
                model = DQN.load(agent.model_path)
            else:
                return None
            
            # Convert state to observation
            obs = self._state_to_observation(state)
            
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            
            # Convert action to readable format
            return self._action_to_dict(action, agent.environment_type)
            
        except Exception as e:
            logger.error(f"Failed to get action from agent {agent_id}: {e}")
            return None
    
    def _state_to_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert state dict to observation array"""
        
        # Convert state to numerical observation
        obs = np.array([
            state.get("workflow_progress", 0.0),
            state.get("resource_utilization", 0.5),
            state.get("agent_availability", 0.8),
            state.get("task_complexity", 0.5),
            state.get("time_pressure", 0.3),
            state.get("error_rate", 0.0),
            state.get("collaboration_score", 0.7),
            0.0,  # workflow_type placeholder
            0.0,  # step placeholder
            # Add derived features
            state.get("workflow_progress", 0.0) * state.get("collaboration_score", 0.7),
            state.get("resource_utilization", 0.5) * state.get("agent_availability", 0.8),
            1.0 - state.get("error_rate", 0.0),
            np.mean([state.get("workflow_progress", 0.0), state.get("collaboration_score", 0.7)]),
            # Padding
            *([0.0] * 7)
        ], dtype=np.float32)
        
        return obs
    
    def _action_to_dict(self, action: int, environment_type: RLEnvironmentType) -> Dict[str, Any]:
        """Convert action integer to readable dict"""
        
        action_mapping = {
            0: {"type": "accelerate", "description": "Accelerate workflow progress"},
            1: {"type": "optimize", "description": "Optimize resource usage"},
            2: {"type": "collaborate", "description": "Increase collaboration"},
            3: {"type": "simplify", "description": "Simplify tasks"},
            4: {"type": "delay", "description": "Reduce time pressure"},
            5: {"type": "boost_resources", "description": "Increase resource allocation"},
            6: {"type": "team_focus", "description": "Focus on team collaboration"},
            7: {"type": "quality_focus", "description": "Focus on quality"},
            8: {"type": "scale_up", "description": "Scale up resources"},
            9: {"type": "push_deadline", "description": "Push toward deadline"}
        }
        
        return action_mapping.get(action, {"type": "unknown", "description": "Unknown action"})
    
    # ===== UTILITY METHODS =====
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for RL"""
        
        topics = [
            "q.ml.rl.training.started",
            "q.ml.rl.training.completed",
            "q.ml.rl.agent.deployed",
            "q.ml.rl.experience.collected",
            "q.ml.rl.action.taken"
        ]
        
        logger.info("RL Pulsar topics configured")
    
    async def _register_environments(self):
        """Register available environments"""
        
        environments = {
            RLEnvironmentType.WORKFLOW_OPTIMIZATION: WorkflowEnvironment,
            RLEnvironmentType.AGENT_COORDINATION: WorkflowEnvironment,  # Placeholder
            RLEnvironmentType.RESOURCE_ALLOCATION: WorkflowEnvironment,  # Placeholder
            RLEnvironmentType.TASK_SCHEDULING: WorkflowEnvironment,  # Placeholder
            RLEnvironmentType.MULTI_AGENT_COLLABORATION: WorkflowEnvironment  # Placeholder
        }
        
        for env_type, env_class in environments.items():
            self.environment_registry[env_type.value] = env_class
        
        logger.info("RL environments registered")
    
    async def _load_trained_agents(self):
        """Load existing trained agents"""
        
        agents_file = self.model_storage_path / "trained_agents.json"
        if agents_file.exists():
            try:
                with open(agents_file, 'r') as f:
                    agents_data = json.load(f)
                
                for agent_data in agents_data:
                    agent = RLAgent(**agent_data)
                    self.trained_agents[agent.agent_id] = agent
                
                logger.info(f"Loaded {len(self.trained_agents)} trained agents")
            except Exception as e:
                logger.error(f"Failed to load trained agents: {e}")
    
    async def _save_trained_agents(self):
        """Save trained agents to storage"""
        
        agents_file = self.model_storage_path / "trained_agents.json"
        try:
            agents_data = [asdict(agent) for agent in self.trained_agents.values()]
            
            with open(agents_file, 'w') as f:
                json.dump(agents_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.trained_agents)} trained agents")
        except Exception as e:
            logger.error(f"Failed to save trained agents: {e}")
    
    # ===== BACKGROUND TASKS =====
    
    async def _experience_collection_loop(self):
        """Background loop for collecting experiences"""
        
        while True:
            try:
                # Listen for workflow events
                # This would integrate with the workflow engine
                
                # For now, just sleep
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in experience collection: {e}")
                await asyncio.sleep(10)
    
    async def _training_monitor(self):
        """Monitor active training sessions"""
        
        while True:
            try:
                current_time = datetime.utcnow()
                
                for session_id, session in list(self.active_training_sessions.items()):
                    if session.status == RLTrainingStatus.TRAINING:
                        # Check for timeout or completion
                        if session.started_at and \
                           (current_time - session.started_at).total_seconds() > 24 * 3600:  # 24 hours
                            
                            session.status = RLTrainingStatus.FAILED
                            session.completed_at = current_time
                            
                            logger.warning(f"Training session {session_id} timed out")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in training monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _performance_tracking(self):
        """Track RL performance metrics"""
        
        while True:
            try:
                # Update metrics
                completed_sessions = [
                    session for session in self.active_training_sessions.values()
                    if session.status == RLTrainingStatus.COMPLETED
                ]
                
                if completed_sessions:
                    # Calculate average reward
                    rewards = [session.best_reward for session in completed_sessions if session.best_reward != float('-inf')]
                    if rewards:
                        self.rl_metrics["average_reward"] = np.mean(rewards)
                        self.rl_metrics["best_reward"] = max(rewards)
                
                # Update total episodes
                self.rl_metrics["total_episodes"] = len(self.experience_buffer)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(300)
    
    # ===== PUBLIC API METHODS =====
    
    async def get_training_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of RL training session"""
        
        if session_id not in self.active_training_sessions:
            return None
        
        session = self.active_training_sessions[session_id]
        
        return {
            "session_id": session_id,
            "agent_name": session.agent_name,
            "status": session.status.value,
            "environment_type": session.environment_type.value,
            "algorithm": session.algorithm.value,
            "progress": session.current_timesteps / session.total_timesteps if session.total_timesteps > 0 else 0.0,
            "current_reward": session.current_reward,
            "best_reward": session.best_reward,
            "created_at": session.created_at,
            "started_at": session.started_at,
            "completed_at": session.completed_at
        }
    
    async def list_trained_agents(self) -> List[Dict[str, Any]]:
        """List all trained RL agents"""
        
        return [
            {
                "agent_id": agent.agent_id,
                "agent_name": agent.agent_name,
                "algorithm": agent.algorithm.value,
                "environment_type": agent.environment_type.value,
                "performance_metrics": agent.performance_metrics,
                "last_updated": agent.last_updated,
                "active": agent.active
            }
            for agent in self.trained_agents.values()
        ]
    
    async def get_rl_metrics(self) -> Dict[str, Any]:
        """Get RL service metrics"""
        
        return {
            "service_metrics": self.rl_metrics,
            "active_training_sessions": len([s for s in self.active_training_sessions.values() if s.status == RLTrainingStatus.TRAINING]),
            "trained_agents": len(self.trained_agents),
            "experience_buffer_size": len(self.experience_buffer),
            "workflow_experiences": len(self.workflow_experiences)
        }

# Global instance
rl_service = ReinforcementLearningService() 