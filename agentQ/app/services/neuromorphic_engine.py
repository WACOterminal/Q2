"""
Neuromorphic Computing Engine

This service provides high-level neuromorphic computing capabilities:
- Coordinated spiking neural networks
- Bio-inspired cognitive architectures
- Adaptive learning and memory formation
- Energy-efficient computation
- Real-time sensory processing
- Emergent behavior coordination
"""

import logging
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict, dataclass
from enum import Enum
import uuid
import json

# Q Platform imports
from app.services.spiking_neural_networks import (
    SpikingNeuralNetworksService, SpikeEvent, SpikingNetwork, NeuronType
)
from app.services.pulsar_service import PulsarService
from app.services.ignite_service import IgniteService
from app.services.memory_service import MemoryService
from shared.q_memory_schemas.memory_models import AgentMemory, MemoryType

logger = logging.getLogger(__name__)

class CognitiveTask(Enum):
    """Types of cognitive tasks"""
    PATTERN_RECOGNITION = "pattern_recognition"
    TEMPORAL_LEARNING = "temporal_learning"
    ASSOCIATIVE_MEMORY = "associative_memory"
    SENSORY_PROCESSING = "sensory_processing"
    DECISION_MAKING = "decision_making"
    MOTOR_CONTROL = "motor_control"
    ATTENTION = "attention"
    ADAPTATION = "adaptation"

class ArchitectureType(Enum):
    """Neuromorphic architecture types"""
    FEEDFORWARD = "feedforward"
    RECURRENT = "recurrent"
    RESERVOIR = "reservoir"
    HIERARCHICAL = "hierarchical"
    MODULAR = "modular"
    CORTICAL_COLUMN = "cortical_column"

@dataclass
class CognitiveModule:
    """Cognitive processing module"""
    module_id: str
    name: str
    task_type: CognitiveTask
    network_ids: List[str]
    input_channels: List[str]
    output_channels: List[str]
    energy_budget: float
    adaptation_rate: float
    performance_score: float
    created_at: datetime

@dataclass
class NeuroTask:
    """Neuromorphic computing task"""
    task_id: str
    name: str
    task_type: CognitiveTask
    input_data: np.ndarray
    expected_output: Optional[np.ndarray]
    
    # Processing configuration
    processing_time: float
    energy_constraint: Optional[float]
    real_time_required: bool
    
    # Results
    output_spikes: List[SpikeEvent]
    energy_consumed: float
    processing_latency: float
    accuracy: Optional[float]
    
    status: str
    created_at: datetime
    completed_at: Optional[datetime]

@dataclass
class NeuromorphicArchitecture:
    """Complete neuromorphic architecture"""
    architecture_id: str
    name: str
    architecture_type: ArchitectureType
    modules: Dict[str, CognitiveModule]
    connections: List[Tuple[str, str]]  # Module connections
    
    # Performance metrics
    total_energy_consumption: float
    total_processing_time: float
    task_performance: Dict[str, float]
    
    # Adaptation
    learning_enabled: bool
    adaptation_history: List[Dict[str, Any]]
    
    created_at: datetime

class NeuromorphicEngine:
    """
    High-level neuromorphic computing engine
    """
    
    def __init__(self):
        self.snn_service = SpikingNeuralNetworksService()
        self.pulsar_service = PulsarService()
        self.ignite_service = IgniteService()
        self.memory_service = MemoryService()
        
        # Architectures and tasks
        self.architectures: Dict[str, NeuromorphicArchitecture] = {}
        self.active_tasks: Dict[str, NeuroTask] = {}
        self.completed_tasks: List[NeuroTask] = []
        
        # Performance optimization
        self.energy_optimizer = EnergyOptimizer()
        self.adaptation_controller = AdaptationController()
        
        # Configuration
        self.config = {
            "max_energy_budget": 1000.0,  # pJ
            "real_time_threshold": 100.0,  # ms
            "adaptation_enabled": True,
            "energy_optimization_enabled": True,
            "performance_tracking": True
        }
        
        # Metrics
        self.metrics = {
            "architectures_created": 0,
            "tasks_completed": 0,
            "total_energy_saved": 0.0,
            "average_accuracy": 0.0,
            "adaptation_events": 0
        }
        self._market_data_consumer = None
    
    async def initialize(self):
        """Initialize the neuromorphic engine"""
        logger.info("Initializing Neuromorphic Computing Engine")
        
        # Initialize sub-services
        await self.snn_service.initialize()
        
        # --- NEW: Subscribe to Market Data Topic ---
        try:
            self._market_data_consumer = self.pulsar_service.subscribe(
                topic='persistent://public/default/market-data',
                subscription_name='neuromorphic-engine-market-data-sub',
                callback=self._handle_market_data
            )
            logger.info("Subscribed to market data topic.")
        except Exception as e:
            logger.error("Failed to subscribe to market data topic", exc_info=True)
        
        # Create sample architectures
        await self._create_sample_architectures()
        
        # Start background tasks
        asyncio.create_task(self._task_processing_loop())
        asyncio.create_task(self._energy_optimization_loop())
        asyncio.create_task(self._adaptation_loop())
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        logger.info("Neuromorphic Computing Engine initialized successfully")
    
    # ===== ARCHITECTURE CREATION =====
    
    async def create_cognitive_architecture(
        self,
        name: str,
        architecture_type: ArchitectureType,
        task_types: List[CognitiveTask]
    ) -> str:
        """Create a neuromorphic cognitive architecture"""
        architecture_id = f"arch_{uuid.uuid4().hex[:12]}"
        
        modules = {}
        connections = []
        
        # Create modules for each task type
        for i, task_type in enumerate(task_types):
            module = await self._create_cognitive_module(f"module_{i}", task_type)
            modules[module.module_id] = module
        
        # Create connections based on architecture type
        if architecture_type == ArchitectureType.FEEDFORWARD:
            connections = await self._create_feedforward_connections(modules)
        elif architecture_type == ArchitectureType.RECURRENT:
            connections = await self._create_recurrent_connections(modules)
        elif architecture_type == ArchitectureType.HIERARCHICAL:
            connections = await self._create_hierarchical_connections(modules)
        
        architecture = NeuromorphicArchitecture(
            architecture_id=architecture_id,
            name=name,
            architecture_type=architecture_type,
            modules=modules,
            connections=connections,
            total_energy_consumption=0.0,
            total_processing_time=0.0,
            task_performance={},
            learning_enabled=True,
            adaptation_history=[],
            created_at=datetime.utcnow()
        )
        
        self.architectures[architecture_id] = architecture
        self.metrics["architectures_created"] += 1
        
        logger.info(f"Created neuromorphic architecture: {architecture_id}")
        return architecture_id
    
    async def _create_cognitive_module(self, module_id: str, task_type: CognitiveTask) -> CognitiveModule:
        """Create a cognitive module for specific task"""
        # Determine network configuration based on task type
        if task_type == CognitiveTask.PATTERN_RECOGNITION:
            network_config = {"input": 784, "hidden": 128, "output": 10}  # MNIST-like
        elif task_type == CognitiveTask.TEMPORAL_LEARNING:
            network_config = {"input": 100, "hidden": 200, "output": 50}
        elif task_type == CognitiveTask.ASSOCIATIVE_MEMORY:
            network_config = {"input": 256, "hidden": 512, "output": 256}
        else:
            network_config = {"input": 64, "hidden": 128, "output": 32}
        
        # Create spiking neural network
        network_id = await self.snn_service.create_spiking_network(
            name=f"{module_id}_network",
            num_input_neurons=network_config["input"],
            num_hidden_neurons=network_config["hidden"],
            num_output_neurons=network_config["output"],
            neuron_type=NeuronType.LEAKY_INTEGRATE_FIRE
        )
        
        module = CognitiveModule(
            module_id=module_id,
            name=f"{task_type.value}_module",
            task_type=task_type,
            network_ids=[network_id],
            input_channels=[f"{module_id}_input"],
            output_channels=[f"{module_id}_output"],
            energy_budget=100.0,  # pJ
            adaptation_rate=0.01,
            performance_score=0.5,
            created_at=datetime.utcnow()
        )
        
        return module
    
    # ===== TASK PROCESSING =====
    
    async def process_cognitive_task(
        self,
        architecture_id: str,
        task_name: str,
        task_type: CognitiveTask,
        input_data: np.ndarray,
        expected_output: Optional[np.ndarray] = None
    ) -> str:
        """Process a cognitive task using neuromorphic architecture"""
        if architecture_id not in self.architectures:
            raise ValueError(f"Architecture not found: {architecture_id}")
        
        task = NeuroTask(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            name=task_name,
            task_type=task_type,
            input_data=input_data,
            expected_output=expected_output,
            processing_time=0.0,
            energy_constraint=self.config["max_energy_budget"],
            real_time_required=True,
            output_spikes=[],
            energy_consumed=0.0,
            processing_latency=0.0,
            accuracy=None,
            status="pending",
            created_at=datetime.utcnow(),
            completed_at=None
        )
        
        self.active_tasks[task.task_id] = task
        
        # Start processing
        await self._execute_cognitive_task(architecture_id, task)
        
        return task.task_id
    
    async def _execute_cognitive_task(self, architecture_id: str, task: NeuroTask):
        """Execute cognitive task on neuromorphic architecture"""
        architecture = self.architectures[architecture_id]
        start_time = datetime.utcnow()
        
        try:
            task.status = "processing"
            
            # Find appropriate module for task
            target_module = None
            for module in architecture.modules.values():
                if module.task_type == task.task_type:
                    target_module = module
                    break
            
            if not target_module:
                # Use first available module
                target_module = list(architecture.modules.values())[0]
            
            # Convert input data to spike events
            input_spikes = await self._encode_input_to_spikes(task.input_data)
            
            # Process through spiking neural network
            simulation_result = await self.snn_service.simulate_network(
                target_module.network_ids[0],
                input_spikes,
                simulation_time=100.0  # ms
            )
            
            # Extract results
            task.output_spikes = simulation_result.get("output_spikes", [])
            task.energy_consumed = simulation_result.get("total_energy", 0.0)
            task.processing_latency = (datetime.utcnow() - start_time).total_seconds() * 1000  # ms
            
            # Calculate accuracy if expected output provided
            if task.expected_output is not None:
                task.accuracy = await self._calculate_task_accuracy(task)
            
            # Update module performance
            target_module.performance_score = task.accuracy or 0.5
            target_module.energy_budget -= task.energy_consumed
            
            # Update architecture metrics
            architecture.total_energy_consumption += task.energy_consumed
            architecture.total_processing_time += task.processing_latency
            
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            del self.active_tasks[task.task_id]
            
            # Trigger adaptation if performance is low
            if task.accuracy and task.accuracy < 0.7:
                await self._trigger_adaptation(architecture_id, target_module.module_id)
            
            # Update global metrics
            self.metrics["tasks_completed"] += 1
            if task.accuracy:
                self.metrics["average_accuracy"] = (
                    (self.metrics["average_accuracy"] * (self.metrics["tasks_completed"] - 1) + task.accuracy) /
                    self.metrics["tasks_completed"]
                )
            
            logger.info(f"Cognitive task completed: {task.task_id}, accuracy: {task.accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Cognitive task failed: {task.task_id}, error: {e}")
            task.status = "failed"
            task.completed_at = datetime.utcnow()
    
    async def _encode_input_to_spikes(self, input_data: np.ndarray, num_input_neurons: int = 100) -> List[SpikeEvent]:
        """
        Converts input data to spike events using basis function encoding.
        Each neuron represents a part of the input's value range.
        """
        spikes = []
        
        # Define the value range and basis functions (e.g., for a stock price)
        min_val, max_val = 100, 2000 # Assumed range for stock prices
        centers = np.linspace(min_val, max_val, num_input_neurons)
        width = (max_val - min_val) / num_input_neurons # Width of each neuron's receptive field
        
        for value in input_data.flatten():
            # Find which neuron's "field" the value falls into
            # Using a Gaussian (bell curve) to determine spike probability
            distances = np.abs(value - centers)
            
            # The closer a neuron's center is to the value, the higher its spike probability
            spike_probabilities = np.exp(-(distances**2) / (2 * width**2))
            
            for i, prob in enumerate(spike_probabilities):
                # Neurons with high probability will fire
                if np.random.random() < prob:
                    # The spike time can encode additional information, but we'll keep it simple for now
                    spike = SpikeEvent(
                        neuron_id=f"input_{i}",
                        timestamp=np.random.uniform(0, 10), # Spike within first 10ms
                        intensity=prob # Intensity can represent confidence
                    )
                    spikes.append(spike)
        
        return spikes
    
    async def _calculate_task_accuracy(self, task: NeuroTask) -> float:
        """Calculate task accuracy based on output spikes"""
        if not task.output_spikes or task.expected_output is None:
            return 0.0
        
        # Convert spikes to output vector (simplified)
        output_vector = np.zeros(len(task.expected_output))
        
        for spike in task.output_spikes:
            if spike.neuron_id.startswith("output_"):
                neuron_idx = int(spike.neuron_id.split("_")[1])
                if neuron_idx < len(output_vector):
                    output_vector[neuron_idx] += 1
        
        # Normalize
        if np.sum(output_vector) > 0:
            output_vector = output_vector / np.sum(output_vector)
        
        # Calculate similarity to expected output
        if np.sum(task.expected_output) > 0:
            expected_normalized = task.expected_output / np.sum(task.expected_output)
            # Use cosine similarity
            similarity = np.dot(output_vector, expected_normalized) / (
                np.linalg.norm(output_vector) * np.linalg.norm(expected_normalized) + 1e-8
            )
            return max(0.0, similarity)
        
        return 0.5  # Default accuracy
    
    # ===== ADAPTATION AND OPTIMIZATION =====
    
    async def _trigger_adaptation(self, architecture_id: str, module_id: str):
        """Trigger adaptation for underperforming module"""
        architecture = self.architectures[architecture_id]
        module = architecture.modules[module_id]
        
        # Increase adaptation rate
        module.adaptation_rate *= 1.1
        
        # Record adaptation event
        adaptation_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "module_id": module_id,
            "trigger": "performance_degradation",
            "old_adaptation_rate": module.adaptation_rate / 1.1,
            "new_adaptation_rate": module.adaptation_rate
        }
        
        architecture.adaptation_history.append(adaptation_event)
        self.metrics["adaptation_events"] += 1
        
        logger.info(f"Triggered adaptation for module: {module_id}")
    
    # ===== BACKGROUND TASKS =====
    
    async def _task_processing_loop(self):
        """Background task processing"""
        while True:
            try:
                await asyncio.sleep(0.1)  # High frequency processing
                
                # Check for real-time constraints
                current_time = datetime.utcnow()
                for task in list(self.active_tasks.values()):
                    if task.real_time_required:
                        elapsed = (current_time - task.created_at).total_seconds() * 1000
                        if elapsed > self.config["real_time_threshold"]:
                            logger.warning(f"Task {task.task_id} exceeded real-time threshold")
                
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
    
    async def _energy_optimization_loop(self):
        """Background energy optimization"""
        while True:
            try:
                await asyncio.sleep(10.0)  # Every 10 seconds
                
                if self.config["energy_optimization_enabled"]:
                    for architecture in self.architectures.values():
                        energy_saved = await self.energy_optimizer.optimize_architecture(architecture)
                        self.metrics["total_energy_saved"] += energy_saved
                
            except Exception as e:
                logger.error(f"Error in energy optimization loop: {e}")
    
    async def _adaptation_loop(self):
        """Background adaptation control"""
        while True:
            try:
                await asyncio.sleep(30.0)  # Every 30 seconds
                
                if self.config["adaptation_enabled"]:
                    for architecture in self.architectures.values():
                        await self.adaptation_controller.update_adaptation(architecture)
                
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
    
    async def _handle_market_data(self, msg):
        """Callback to process incoming market data from Pulsar."""
        try:
            data = json.loads(msg.data().decode('utf-8'))
            logger.debug("Received market data", data=data)
            
            # Convert the data to spikes and process it through the network
            # This is a conceptual link to the next steps.
            spikes = await self._encode_input_to_spikes(np.array([data['price']]))
            
            # Find the relevant network to process this (simplified)
            for architecture in self.architectures.values():
                if "AnomalyDetector" in architecture.name:
                    await self.snn_service.simulate_network(
                        list(architecture.modules.values())[0].network_ids[0],
                        spikes,
                        simulation_time=20.0 # Short simulation for a single tick
                    )
                    break # Process with the first found detector for now
        except Exception as e:
            logger.error("Failed to handle market data message", exc_info=True)
    
    # ===== HELPER METHODS =====
    
    async def _create_feedforward_connections(self, modules: Dict[str, CognitiveModule]) -> List[Tuple[str, str]]:
        """Create feedforward connections between modules"""
        connections = []
        module_list = list(modules.keys())
        
        for i in range(len(module_list) - 1):
            connections.append((module_list[i], module_list[i + 1]))
        
        return connections
    
    async def _create_recurrent_connections(self, modules: Dict[str, CognitiveModule]) -> List[Tuple[str, str]]:
        """Create recurrent connections between modules"""
        connections = []
        module_list = list(modules.keys())
        
        # All-to-all connections
        for i in range(len(module_list)):
            for j in range(len(module_list)):
                if i != j:
                    connections.append((module_list[i], module_list[j]))
        
        return connections
    
    async def _create_hierarchical_connections(self, modules: Dict[str, CognitiveModule]) -> List[Tuple[str, str]]:
        """Create hierarchical connections between modules"""
        connections = []
        module_list = list(modules.keys())
        
        # Tree-like structure
        for i in range(len(module_list)):
            for j in range(i + 1, min(i + 3, len(module_list))):
                connections.append((module_list[i], module_list[j]))
        
        return connections
    
    async def _create_sample_architectures(self):
        """Create sample neuromorphic architectures"""
        # Pattern recognition architecture
        await self.create_cognitive_architecture(
            "Visual Processing",
            ArchitectureType.HIERARCHICAL,
            [CognitiveTask.PATTERN_RECOGNITION, CognitiveTask.ATTENTION]
        )
        
        # Temporal learning architecture
        await self.create_cognitive_architecture(
            "Temporal Processing",
            ArchitectureType.RECURRENT,
            [CognitiveTask.TEMPORAL_LEARNING, CognitiveTask.ASSOCIATIVE_MEMORY]
        )
        
        logger.info("Sample neuromorphic architectures created")
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics"""
        topics = [
            "q.neuromorphic.task.completed",
            "q.neuromorphic.adaptation.triggered",
            "q.neuromorphic.energy.optimized"
        ]
        
        for topic in topics:
            await self.pulsar_service.ensure_topic(topic)

class EnergyOptimizer:
    """Energy optimization for neuromorphic architectures"""
    
    async def optimize_architecture(self, architecture: NeuromorphicArchitecture) -> float:
        """Optimize energy consumption of architecture"""
        energy_saved = 0.0
        
        # Reduce energy budget for overperforming modules
        for module in architecture.modules.values():
            if module.performance_score > 0.8 and module.energy_budget > 50.0:
                reduction = module.energy_budget * 0.1
                module.energy_budget -= reduction
                energy_saved += reduction
        
        return energy_saved

class AdaptationController:
    """Adaptation control for neuromorphic architectures"""
    
    async def update_adaptation(self, architecture: NeuromorphicArchitecture):
        """Update adaptation parameters"""
        for module in architecture.modules.values():
            # Decay adaptation rate over time
            module.adaptation_rate *= 0.99
            module.adaptation_rate = max(0.001, module.adaptation_rate)

# Global service instance
neuromorphic_engine = NeuromorphicEngine() 