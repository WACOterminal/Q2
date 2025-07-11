"""
Federated Learning Orchestrator

This service coordinates distributed ML training across the multi-agent network:
- Manages federated learning rounds and model aggregation
- Handles secure model parameter sharing
- Coordinates training across heterogeneous agents
- Implements differential privacy and secure aggregation
- Tracks model performance and convergence
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import hashlib
import pickle
from pathlib import Path

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.q_memory_schemas.memory_models import AgentMemory
from shared.q_vectorstore_client.client import VectorStoreClient
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)

class FederatedLearningStatus(Enum):
    """Status of federated learning rounds"""
    INITIALIZING = "initializing"
    ROUND_STARTED = "round_started"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AggregationStrategy(Enum):
    """Model aggregation strategies"""
    FEDERATED_AVERAGING = "federated_averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    SECURE_AGGREGATION = "secure_aggregation"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    ADAPTIVE_WEIGHTING = "adaptive_weighting"

@dataclass
class FederatedLearningRound:
    """Represents a federated learning round"""
    round_id: str
    session_id: str
    round_number: int
    global_model_version: str
    participating_agents: List[str]
    aggregation_strategy: AggregationStrategy
    target_agents: int
    min_agents: int
    max_wait_time: int
    privacy_budget: float
    differential_privacy_epsilon: float
    status: FederatedLearningStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    global_model_params: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    convergence_metrics: Optional[Dict[str, float]] = None

@dataclass
class AgentTrainingUpdate:
    """Training update from an agent"""
    agent_id: str
    round_id: str
    local_model_params: Dict[str, Any]
    training_samples: int
    local_loss: float
    local_accuracy: float
    training_time: float
    privacy_metrics: Dict[str, float]
    timestamp: datetime

@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    model_architecture: str
    model_params: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: datetime
    round_id: str
    agent_contributions: List[str]

class FederatedLearningOrchestrator:
    """
    Orchestrates federated learning across the multi-agent network
    """
    
    def __init__(self, 
                 model_storage_path: str = "models/federated",
                 min_agents_per_round: int = 3,
                 max_wait_time: int = 300,
                 privacy_budget: float = 10.0):
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.min_agents_per_round = min_agents_per_round
        self.max_wait_time = max_wait_time
        self.privacy_budget = privacy_budget
        
        # Active federated learning state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.active_rounds: Dict[str, FederatedLearningRound] = {}
        self.agent_updates: Dict[str, List[AgentTrainingUpdate]] = defaultdict(list)
        self.model_versions: Dict[str, ModelVersion] = {}
        
        # Agent capabilities tracking
        self.agent_capabilities: Dict[str, Dict[str, Any]] = {}
        self.agent_training_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Privacy and security
        self.vault_client = VaultClient(role="managerq-role")
        self.encryption_keys: Dict[str, str] = {}
        
        # Performance tracking
        self.fl_metrics = {
            "total_rounds": 0,
            "successful_rounds": 0,
            "failed_rounds": 0,
            "average_convergence_time": 0.0,
            "average_model_accuracy": 0.0,
            "privacy_violations": 0
        }
        
        # Configuration
        self.config = {
            "differential_privacy_epsilon": 1.0,
            "secure_aggregation_threshold": 0.9,
            "model_update_compression": True,
            "byzantine_tolerance": False,
            "adaptive_learning_rate": True
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
    async def initialize(self):
        """Initialize the federated learning orchestrator"""
        logger.info("Initializing Federated Learning Orchestrator")
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        # Load existing model versions
        await self._load_model_versions()
        
        # Start background tasks
        self.background_tasks.add(asyncio.create_task(self._monitor_active_rounds()))
        self.background_tasks.add(asyncio.create_task(self._cleanup_expired_rounds()))
        self.background_tasks.add(asyncio.create_task(self._performance_monitoring()))
        
        logger.info("Federated Learning Orchestrator initialized successfully")
    
    async def shutdown(self):
        """Shutdown the federated learning orchestrator"""
        logger.info("Shutting down Federated Learning Orchestrator")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save model versions
        await self._save_model_versions()
        
        logger.info("Federated Learning Orchestrator shut down successfully")
    
    # ===== SESSION MANAGEMENT =====
    
    async def start_federated_learning_session(
        self,
        model_architecture: str,
        dataset_config: Dict[str, Any],
        training_config: Dict[str, Any],
        participating_agents: Optional[List[str]] = None,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDERATED_AVERAGING,
        privacy_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new federated learning session
        
        Args:
            model_architecture: Model architecture specification
            dataset_config: Dataset configuration
            training_config: Training hyperparameters
            participating_agents: List of specific agents to include
            aggregation_strategy: Model aggregation strategy
            privacy_config: Privacy configuration
            
        Returns:
            Session ID
        """
        session_id = f"fl_session_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Starting federated learning session: {session_id}")
        
        # Initialize global model
        global_model = await self._initialize_global_model(
            model_architecture, 
            training_config
        )
        
        # Create session
        session = {
            "session_id": session_id,
            "model_architecture": model_architecture,
            "dataset_config": dataset_config,
            "training_config": training_config,
            "aggregation_strategy": aggregation_strategy,
            "privacy_config": privacy_config or {},
            "global_model": global_model,
            "participating_agents": participating_agents or [],
            "rounds": [],
            "status": "active",
            "created_at": datetime.utcnow(),
            "performance_history": []
        }
        
        self.active_sessions[session_id] = session
        
        # Start first round
        await self._start_federated_round(session_id)
        
        # Publish session started event
        await shared_pulsar_client.publish(
            "q.ml.federated.session.started",
            {
                "session_id": session_id,
                "model_architecture": model_architecture,
                "participating_agents": len(participating_agents) if participating_agents else "auto",
                "aggregation_strategy": aggregation_strategy.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return session_id
    
    async def _start_federated_round(self, session_id: str) -> str:
        """Start a new federated learning round"""
        session = self.active_sessions[session_id]
        round_number = len(session["rounds"]) + 1
        round_id = f"fl_round_{uuid.uuid4().hex[:12]}"
        
        # Select participating agents
        participating_agents = await self._select_participating_agents(
            session_id, 
            session["participating_agents"]
        )
        
        if len(participating_agents) < self.min_agents_per_round:
            logger.warning(f"Not enough agents for round {round_number}: {len(participating_agents)}")
            return None
        
        # Create round
        fl_round = FederatedLearningRound(
            round_id=round_id,
            session_id=session_id,
            round_number=round_number,
            global_model_version=f"v{round_number}",
            participating_agents=participating_agents,
            aggregation_strategy=session["aggregation_strategy"],
            target_agents=len(participating_agents),
            min_agents=self.min_agents_per_round,
            max_wait_time=self.max_wait_time,
            privacy_budget=self.privacy_budget,
            differential_privacy_epsilon=self.config["differential_privacy_epsilon"],
            status=FederatedLearningStatus.INITIALIZING,
            created_at=datetime.utcnow()
        )
        
        self.active_rounds[round_id] = fl_round
        session["rounds"].append(round_id)
        
        # Send training tasks to agents
        await self._send_training_tasks(fl_round, session)
        
        # Update round status
        fl_round.status = FederatedLearningStatus.ROUND_STARTED
        fl_round.started_at = datetime.utcnow()
        
        logger.info(f"Started federated learning round {round_number} with {len(participating_agents)} agents")
        
        return round_id
    
    # ===== AGENT COORDINATION =====
    
    async def _select_participating_agents(
        self,
        session_id: str,
        preferred_agents: List[str]
    ) -> List[str]:
        """Select agents for federated learning round"""
        
        # Get available agents with ML capabilities
        available_agents = await self._get_available_ml_agents()
        
        if preferred_agents:
            # Filter to preferred agents that are available
            participating_agents = [
                agent_id for agent_id in preferred_agents 
                if agent_id in available_agents
            ]
        else:
            # Select best available agents
            participating_agents = await self._select_best_agents(
                available_agents, 
                session_id
            )
        
        return participating_agents
    
    async def _get_available_ml_agents(self) -> List[str]:
        """Get list of agents with ML capabilities"""
        # Query agent registry for ML-capable agents
        # This would integrate with the existing agent registry
        ml_agents = []
        
        # For now, return all agents that have registered ML capabilities
        for agent_id, capabilities in self.agent_capabilities.items():
            if capabilities.get("ml_training", False):
                ml_agents.append(agent_id)
        
        return ml_agents
    
    async def _select_best_agents(
        self,
        available_agents: List[str],
        session_id: str
    ) -> List[str]:
        """Select best agents for training based on performance history"""
        
        # Score agents based on past performance
        agent_scores = {}
        for agent_id in available_agents:
            score = await self._calculate_agent_score(agent_id, session_id)
            agent_scores[agent_id] = score
        
        # Select top agents
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        selected_agents = [agent_id for agent_id, score in sorted_agents[:10]]  # Top 10
        
        return selected_agents
    
    async def _calculate_agent_score(self, agent_id: str, session_id: str) -> float:
        """Calculate agent score for selection"""
        base_score = 1.0
        
        # Historical performance
        history = self.agent_training_history.get(agent_id, [])
        if history:
            avg_accuracy = np.mean([h.get("accuracy", 0.0) for h in history])
            avg_speed = np.mean([h.get("training_time", 300.0) for h in history])
            reliability = len([h for h in history if h.get("successful", False)]) / len(history)
            
            performance_score = (avg_accuracy * 0.4 + 
                               (300.0 / max(avg_speed, 1.0)) * 0.3 + 
                               reliability * 0.3)
            base_score *= performance_score
        
        # Current load
        current_load = len([r for r in self.active_rounds.values() 
                          if agent_id in r.participating_agents])
        load_penalty = max(0.1, 1.0 - (current_load * 0.2))
        base_score *= load_penalty
        
        return base_score
    
    async def _send_training_tasks(
        self,
        fl_round: FederatedLearningRound,
        session: Dict[str, Any]
    ):
        """Send training tasks to participating agents"""
        
        for agent_id in fl_round.participating_agents:
            training_task = {
                "task_type": "federated_learning_training",
                "round_id": fl_round.round_id,
                "session_id": fl_round.session_id,
                "global_model_params": session["global_model"]["params"],
                "model_architecture": session["model_architecture"],
                "training_config": session["training_config"],
                "dataset_config": session["dataset_config"],
                "privacy_config": session["privacy_config"],
                "round_number": fl_round.round_number,
                "aggregation_strategy": fl_round.aggregation_strategy.value,
                "max_training_time": fl_round.max_wait_time,
                "differential_privacy_epsilon": fl_round.differential_privacy_epsilon
            }
            
            # Send task to agent
            await shared_pulsar_client.publish(
                f"q.agentq.tasks.{agent_id}",
                training_task
            )
            
            logger.debug(f"Sent training task to agent {agent_id} for round {fl_round.round_id}")
    
    # ===== MODEL AGGREGATION =====
    
    async def receive_training_update(self, update: AgentTrainingUpdate):
        """Receive training update from an agent"""
        
        round_id = update.round_id
        if round_id not in self.active_rounds:
            logger.warning(f"Received update for unknown round: {round_id}")
            return
        
        fl_round = self.active_rounds[round_id]
        
        # Validate update
        if update.agent_id not in fl_round.participating_agents:
            logger.warning(f"Received update from non-participating agent: {update.agent_id}")
            return
        
        # Store update
        self.agent_updates[round_id].append(update)
        
        logger.info(f"Received training update from agent {update.agent_id} for round {round_id}")
        
        # Check if we have enough updates to start aggregation
        if len(self.agent_updates[round_id]) >= fl_round.min_agents:
            await self._start_model_aggregation(round_id)
    
    async def _start_model_aggregation(self, round_id: str):
        """Start model aggregation for a round"""
        
        fl_round = self.active_rounds[round_id]
        if fl_round.status != FederatedLearningStatus.ROUND_STARTED:
            return
        
        fl_round.status = FederatedLearningStatus.AGGREGATING
        
        logger.info(f"Starting model aggregation for round {round_id}")
        
        updates = self.agent_updates[round_id]
        
        # Perform aggregation based on strategy
        if fl_round.aggregation_strategy == AggregationStrategy.FEDERATED_AVERAGING:
            aggregated_params = await self._federated_averaging(updates)
        elif fl_round.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGING:
            aggregated_params = await self._weighted_averaging(updates)
        elif fl_round.aggregation_strategy == AggregationStrategy.SECURE_AGGREGATION:
            aggregated_params = await self._secure_aggregation(updates)
        elif fl_round.aggregation_strategy == AggregationStrategy.DIFFERENTIAL_PRIVACY:
            aggregated_params = await self._differential_privacy_aggregation(updates, fl_round.differential_privacy_epsilon)
        else:
            aggregated_params = await self._adaptive_weighted_averaging(updates)
        
        # Update global model
        session = self.active_sessions[fl_round.session_id]
        session["global_model"]["params"] = aggregated_params
        session["global_model"]["version"] = fl_round.global_model_version
        
        # Store model version
        model_version = ModelVersion(
            version_id=fl_round.global_model_version,
            model_architecture=session["model_architecture"],
            model_params=aggregated_params,
            performance_metrics=await self._calculate_round_metrics(updates),
            created_at=datetime.utcnow(),
            round_id=round_id,
            agent_contributions=[update.agent_id for update in updates]
        )
        
        self.model_versions[model_version.version_id] = model_version
        
        # Complete round
        fl_round.status = FederatedLearningStatus.COMPLETED
        fl_round.completed_at = datetime.utcnow()
        fl_round.global_model_params = aggregated_params
        fl_round.performance_metrics = model_version.performance_metrics
        
        # Update metrics
        self.fl_metrics["total_rounds"] += 1
        self.fl_metrics["successful_rounds"] += 1
        
        # Start next round if session is still active
        if session["status"] == "active":
            await self._start_federated_round(fl_round.session_id)
        
        # Publish round completed event
        await shared_pulsar_client.publish(
            "q.ml.federated.round.completed",
            {
                "round_id": round_id,
                "session_id": fl_round.session_id,
                "round_number": fl_round.round_number,
                "participating_agents": fl_round.participating_agents,
                "performance_metrics": model_version.performance_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Completed federated learning round {round_id}")
    
    async def _federated_averaging(self, updates: List[AgentTrainingUpdate]) -> Dict[str, Any]:
        """Perform federated averaging aggregation"""
        
        total_samples = sum(update.training_samples for update in updates)
        aggregated_params = {}
        
        # Weight by number of training samples
        for update in updates:
            weight = update.training_samples / total_samples
            
            for param_name, param_value in update.local_model_params.items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = np.zeros_like(param_value)
                
                aggregated_params[param_name] += weight * param_value
        
        return aggregated_params
    
    async def _weighted_averaging(self, updates: List[AgentTrainingUpdate]) -> Dict[str, Any]:
        """Perform weighted averaging based on model performance"""
        
        total_weight = 0
        aggregated_params = {}
        
        # Weight by accuracy and inverse of loss
        for update in updates:
            weight = update.local_accuracy * (1.0 / max(update.local_loss, 0.001))
            total_weight += weight
            
            for param_name, param_value in update.local_model_params.items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = np.zeros_like(param_value)
                
                aggregated_params[param_name] += weight * param_value
        
        # Normalize
        for param_name in aggregated_params:
            aggregated_params[param_name] /= total_weight
        
        return aggregated_params
    
    async def _secure_aggregation(self, updates: List[AgentTrainingUpdate]) -> Dict[str, Any]:
        """Perform secure aggregation with encryption"""
        
        # Simplified secure aggregation - in production would use proper crypto
        # For now, use federated averaging with noise
        aggregated_params = await self._federated_averaging(updates)
        
        # Add calibrated noise for privacy
        noise_scale = 0.01  # Configurable
        for param_name, param_value in aggregated_params.items():
            noise = np.random.normal(0, noise_scale, param_value.shape)
            aggregated_params[param_name] += noise
        
        return aggregated_params
    
    async def _differential_privacy_aggregation(
        self,
        updates: List[AgentTrainingUpdate],
        epsilon: float
    ) -> Dict[str, Any]:
        """Perform differential privacy aggregation"""
        
        # Federated averaging with differential privacy
        aggregated_params = await self._federated_averaging(updates)
        
        # Add calibrated noise based on epsilon
        sensitivity = 1.0  # Model sensitivity
        noise_scale = sensitivity / epsilon
        
        for param_name, param_value in aggregated_params.items():
            noise = np.random.laplace(0, noise_scale, param_value.shape)
            aggregated_params[param_name] += noise
        
        return aggregated_params
    
    async def _adaptive_weighted_averaging(self, updates: List[AgentTrainingUpdate]) -> Dict[str, Any]:
        """Perform adaptive weighted averaging"""
        
        # Combine multiple weighting factors
        total_weight = 0
        aggregated_params = {}
        
        for update in updates:
            # Multi-factor weighting
            sample_weight = update.training_samples / max(sum(u.training_samples for u in updates), 1)
            performance_weight = update.local_accuracy * (1.0 / max(update.local_loss, 0.001))
            speed_weight = 1.0 / max(update.training_time, 1.0)
            
            # Combine weights
            weight = (sample_weight * 0.4 + 
                     performance_weight * 0.4 + 
                     speed_weight * 0.2)
            
            total_weight += weight
            
            for param_name, param_value in update.local_model_params.items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = np.zeros_like(param_value)
                
                aggregated_params[param_name] += weight * param_value
        
        # Normalize
        for param_name in aggregated_params:
            aggregated_params[param_name] /= total_weight
        
        return aggregated_params
    
    async def _calculate_round_metrics(self, updates: List[AgentTrainingUpdate]) -> Dict[str, float]:
        """Calculate performance metrics for a round"""
        
        if not updates:
            return {}
        
        metrics = {
            "average_accuracy": np.mean([update.local_accuracy for update in updates]),
            "average_loss": np.mean([update.local_loss for update in updates]),
            "total_samples": sum(update.training_samples for update in updates),
            "average_training_time": np.mean([update.training_time for update in updates]),
            "participating_agents": len(updates),
            "accuracy_std": np.std([update.local_accuracy for update in updates]),
            "loss_std": np.std([update.local_loss for update in updates])
        }
        
        return metrics
    
    # ===== UTILITY METHODS =====
    
    async def _initialize_global_model(
        self,
        model_architecture: str,
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize global model"""
        
        # Create initial model parameters
        # This would depend on the specific model architecture
        global_model = {
            "architecture": model_architecture,
            "params": {},  # Would be populated based on architecture
            "version": "v1",
            "created_at": datetime.utcnow().isoformat()
        }
        
        return global_model
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for federated learning"""
        
        topics = [
            "q.ml.federated.session.started",
            "q.ml.federated.round.started",
            "q.ml.federated.round.completed",
            "q.ml.federated.training.update",
            "q.ml.federated.model.aggregated"
        ]
        
        # Topics would be created automatically by Pulsar
        logger.info("Federated learning Pulsar topics configured")
    
    async def _load_model_versions(self):
        """Load existing model versions from storage"""
        
        versions_file = self.model_storage_path / "model_versions.json"
        if versions_file.exists():
            try:
                with open(versions_file, 'r') as f:
                    versions_data = json.load(f)
                
                for version_data in versions_data:
                    version = ModelVersion(**version_data)
                    self.model_versions[version.version_id] = version
                
                logger.info(f"Loaded {len(self.model_versions)} model versions")
            except Exception as e:
                logger.error(f"Failed to load model versions: {e}")
    
    async def _save_model_versions(self):
        """Save model versions to storage"""
        
        versions_file = self.model_storage_path / "model_versions.json"
        try:
            versions_data = [asdict(version) for version in self.model_versions.values()]
            
            with open(versions_file, 'w') as f:
                json.dump(versions_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.model_versions)} model versions")
        except Exception as e:
            logger.error(f"Failed to save model versions: {e}")
    
    # ===== BACKGROUND TASKS =====
    
    async def _monitor_active_rounds(self):
        """Monitor active rounds and handle timeouts"""
        
        while True:
            try:
                current_time = datetime.utcnow()
                
                for round_id, fl_round in list(self.active_rounds.items()):
                    if fl_round.status == FederatedLearningStatus.ROUND_STARTED:
                        # Check for timeout
                        if fl_round.started_at and \
                           (current_time - fl_round.started_at).total_seconds() > fl_round.max_wait_time:
                            
                            # Check if we have minimum updates
                            if len(self.agent_updates[round_id]) >= fl_round.min_agents:
                                await self._start_model_aggregation(round_id)
                            else:
                                # Timeout with insufficient updates
                                fl_round.status = FederatedLearningStatus.FAILED
                                logger.warning(f"Round {round_id} timed out with insufficient updates")
                                
                                # Update metrics
                                self.fl_metrics["failed_rounds"] += 1
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in round monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_expired_rounds(self):
        """Clean up expired rounds and sessions"""
        
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Clean up completed rounds older than 24 hours
                for round_id, fl_round in list(self.active_rounds.items()):
                    if fl_round.status in [FederatedLearningStatus.COMPLETED, FederatedLearningStatus.FAILED]:
                        if fl_round.completed_at and \
                           (current_time - fl_round.completed_at).total_seconds() > 86400:  # 24 hours
                            
                            # Remove round
                            del self.active_rounds[round_id]
                            if round_id in self.agent_updates:
                                del self.agent_updates[round_id]
                            
                            logger.debug(f"Cleaned up expired round: {round_id}")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def _performance_monitoring(self):
        """Monitor federated learning performance"""
        
        while True:
            try:
                # Calculate performance metrics
                if self.fl_metrics["total_rounds"] > 0:
                    self.fl_metrics["success_rate"] = (
                        self.fl_metrics["successful_rounds"] / self.fl_metrics["total_rounds"]
                    )
                
                # Update convergence time
                completed_rounds = [r for r in self.active_rounds.values() 
                                  if r.status == FederatedLearningStatus.COMPLETED]
                if completed_rounds:
                    convergence_times = [
                        (r.completed_at - r.started_at).total_seconds()
                        for r in completed_rounds
                        if r.started_at and r.completed_at
                    ]
                    if convergence_times:
                        self.fl_metrics["average_convergence_time"] = np.mean(convergence_times)
                
                # Update model accuracy
                if self.model_versions:
                    accuracies = [
                        v.performance_metrics.get("average_accuracy", 0.0)
                        for v in self.model_versions.values()
                        if v.performance_metrics
                    ]
                    if accuracies:
                        self.fl_metrics["average_model_accuracy"] = np.mean(accuracies)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(300)
    
    # ===== PUBLIC API METHODS =====
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a federated learning session"""
        
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Get round statuses
        round_statuses = []
        for round_id in session["rounds"]:
            if round_id in self.active_rounds:
                fl_round = self.active_rounds[round_id]
                round_statuses.append({
                    "round_id": round_id,
                    "round_number": fl_round.round_number,
                    "status": fl_round.status.value,
                    "participating_agents": fl_round.participating_agents,
                    "performance_metrics": fl_round.performance_metrics
                })
        
        return {
            "session_id": session_id,
            "status": session["status"],
            "model_architecture": session["model_architecture"],
            "participating_agents": session["participating_agents"],
            "rounds": round_statuses,
            "current_model_version": session["global_model"]["version"],
            "created_at": session["created_at"]
        }
    
    async def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version"""
        return self.model_versions.get(version_id)
    
    async def list_model_versions(self, session_id: Optional[str] = None) -> List[ModelVersion]:
        """List model versions, optionally filtered by session"""
        
        versions = list(self.model_versions.values())
        
        if session_id:
            versions = [v for v in versions if v.round_id in self.active_sessions.get(session_id, {}).get("rounds", [])]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    async def get_federated_learning_metrics(self) -> Dict[str, Any]:
        """Get federated learning performance metrics"""
        
        return {
            "orchestrator_metrics": self.fl_metrics,
            "active_sessions": len(self.active_sessions),
            "active_rounds": len(self.active_rounds),
            "model_versions": len(self.model_versions),
            "agent_capabilities": len(self.agent_capabilities)
        }

# Global instance
federated_learning_orchestrator = FederatedLearningOrchestrator() 