"""
Spiking Neural Networks Service

This service provides neuromorphic computing capabilities:
- Spiking neurons with biological realism
- Event-driven computation
- Spike-timing-dependent plasticity (STDP)
- Energy-efficient processing
- Temporal pattern recognition
- Adaptive learning mechanisms
"""

import logging
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import asdict, dataclass
from enum import Enum
import uuid
from collections import deque, defaultdict

# Q Platform imports
from app.services.pulsar_service import PulsarService
from app.services.ignite_service import IgniteService
from app.services.memory_service import MemoryService
from shared.q_memory_schemas.memory_models import AgentMemory, MemoryType

logger = logging.getLogger(__name__)

class NeuronType(Enum):
    """Types of spiking neurons"""
    LEAKY_INTEGRATE_FIRE = "leaky_integrate_fire"
    ADAPTIVE_EXPONENTIAL = "adaptive_exponential"
    IZHIKEVICH = "izhikevich"
    HODGKIN_HUXLEY = "hodgkin_huxley"

class PlasticityType(Enum):
    """Types of synaptic plasticity"""
    STDP = "stdp"                    # Spike-timing-dependent plasticity
    HOMEOSTATIC = "homeostatic"      # Homeostatic plasticity
    METAPLASTICITY = "metaplasticity" # Plasticity of plasticity
    INHIBITORY = "inhibitory"        # Inhibitory plasticity

@dataclass
class SpikeEvent:
    """Represents a spike event"""
    neuron_id: str
    timestamp: float
    intensity: float
    source_id: Optional[str] = None

@dataclass
class SpikingNeuron:
    """Spiking neuron model"""
    neuron_id: str
    neuron_type: NeuronType
    
    # Membrane properties
    membrane_potential: float
    threshold: float
    reset_potential: float
    resting_potential: float
    membrane_resistance: float
    membrane_capacitance: float
    refractory_period: float
    
    # Dynamics
    last_spike_time: Optional[float]
    adaptation_current: float
    
    # Learning
    learning_rate: float
    plasticity_threshold: float
    
    # Energy tracking
    energy_consumption: float
    spike_count: int
    
    created_at: datetime

@dataclass
class Synapse:
    """Synaptic connection between neurons"""
    synapse_id: str
    pre_neuron_id: str
    post_neuron_id: str
    
    # Synaptic properties
    weight: float
    delay: float
    plasticity_type: PlasticityType
    
    # STDP parameters
    tau_plus: float
    tau_minus: float
    a_plus: float
    a_minus: float
    
    # Traces for plasticity
    pre_trace: float
    post_trace: float
    
    # Statistics
    activation_count: int
    last_activation: Optional[float]
    
    created_at: datetime

@dataclass
class SpikingNetwork:
    """Spiking neural network"""
    network_id: str
    name: str
    neurons: Dict[str, SpikingNeuron]
    synapses: Dict[str, Synapse]
    
    # Network properties
    simulation_time: float
    time_step: float
    total_energy: float
    total_spikes: int
    
    # Learning configuration
    plasticity_enabled: bool
    homeostasis_enabled: bool
    
    created_at: datetime

class SpikingNeuralNetworksService:
    """
    Service for spiking neural networks and neuromorphic computing
    """
    
    def __init__(self):
        self.pulsar_service = PulsarService()
        self.ignite_service = IgniteService()
        self.memory_service = MemoryService()
        
        # Networks and components
        self.networks: Dict[str, SpikingNetwork] = {}
        self.spike_queues: Dict[str, deque] = {}  # Queues for delayed spikes
        self.detected_anomalies: Dict[str, List[Dict]] = defaultdict(list) # NEW
        self.network_activity_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200)) # NEW: Store recent firing rates
        
        # Simulation parameters
        self.simulation_config = {
            "default_time_step": 0.1,    # ms
            "max_simulation_time": 1000, # ms
            "energy_tracking": True,
            "spike_recording": True,
            "plasticity_enabled": True
        }
        
        # Performance metrics
        self.metrics = {
            "networks_created": 0,
            "total_spikes_processed": 0,
            "total_energy_consumed": 0.0,
            "average_spike_rate": 0.0,
            "plasticity_events": 0
        }
    
    async def initialize(self):
        """Initialize the spiking neural networks service"""
        logger.info("Initializing Spiking Neural Networks Service")
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        # Start background simulation
        asyncio.create_task(self._simulation_loop())
        asyncio.create_task(self._plasticity_update_loop())
        
        logger.info("Spiking Neural Networks Service initialized successfully")
    
    # ===== NETWORK CREATION =====
    
    async def create_spiking_network(
        self,
        name: str,
        num_input_neurons: int,
        num_hidden_neurons: int,
        num_output_neurons: int,
        neuron_type: NeuronType = NeuronType.LEAKY_INTEGRATE_FIRE
    ) -> str:
        """Create a new spiking neural network"""
        network_id = f"snn_{uuid.uuid4().hex[:12]}"
        
        # Create neurons
        neurons = {}
        
        # Input layer
        for i in range(num_input_neurons):
            neuron = await self._create_neuron(f"input_{i}", neuron_type)
            neurons[neuron.neuron_id] = neuron
        
        # Hidden layer
        for i in range(num_hidden_neurons):
            neuron = await self._create_neuron(f"hidden_{i}", neuron_type)
            neurons[neuron.neuron_id] = neuron
        
        # Output layer
        for i in range(num_output_neurons):
            neuron = await self._create_neuron(f"output_{i}", neuron_type)
            neurons[neuron.neuron_id] = neuron
        
        # Create synapses (fully connected layers)
        synapses = {}
        
        # Input to hidden connections
        for i in range(num_input_neurons):
            for j in range(num_hidden_neurons):
                synapse = await self._create_synapse(
                    f"input_{i}", f"hidden_{j}", PlasticityType.STDP
                )
                synapses[synapse.synapse_id] = synapse
        
        # Hidden to output connections
        for i in range(num_hidden_neurons):
            for j in range(num_output_neurons):
                synapse = await self._create_synapse(
                    f"hidden_{i}", f"output_{j}", PlasticityType.STDP
                )
                synapses[synapse.synapse_id] = synapse
        
        # Create network
        network = SpikingNetwork(
            network_id=network_id,
            name=name,
            neurons=neurons,
            synapses=synapses,
            simulation_time=0.0,
            time_step=self.simulation_config["default_time_step"],
            total_energy=0.0,
            total_spikes=0,
            plasticity_enabled=True,
            homeostasis_enabled=True,
            created_at=datetime.utcnow()
        )
        
        self.networks[network_id] = network
        self.spike_queues[network_id] = deque()
        
        self.metrics["networks_created"] += 1
        
        logger.info(f"Created spiking network: {network_id} with {len(neurons)} neurons")
        return network_id
    
    async def _create_neuron(self, neuron_id: str, neuron_type: NeuronType) -> SpikingNeuron:
        """Create a spiking neuron"""
        if neuron_type == NeuronType.LEAKY_INTEGRATE_FIRE:
            return SpikingNeuron(
                neuron_id=neuron_id,
                neuron_type=neuron_type,
                membrane_potential=-70.0,  # mV
                threshold=-50.0,
                reset_potential=-70.0,
                resting_potential=-70.0,
                membrane_resistance=10.0,  # MΩ
                membrane_capacitance=1.0,  # µF
                refractory_period=2.0,     # ms
                last_spike_time=None,
                adaptation_current=0.0,
                learning_rate=0.01,
                plasticity_threshold=0.1,
                energy_consumption=0.0,
                spike_count=0,
                created_at=datetime.utcnow()
            )
        else:
            # Add other neuron types as needed
            return await self._create_neuron(neuron_id, NeuronType.LEAKY_INTEGRATE_FIRE)
    
    async def _create_synapse(
        self, 
        pre_neuron_id: str, 
        post_neuron_id: str, 
        plasticity_type: PlasticityType
    ) -> Synapse:
        """Create a synapse between neurons"""
        return Synapse(
            synapse_id=f"syn_{pre_neuron_id}_{post_neuron_id}",
            pre_neuron_id=pre_neuron_id,
            post_neuron_id=post_neuron_id,
            weight=np.random.normal(0.5, 0.1),  # Random initial weight
            delay=np.random.uniform(0.5, 2.0),  # ms
            plasticity_type=plasticity_type,
            tau_plus=20.0,   # ms, STDP time constant
            tau_minus=20.0,  # ms
            a_plus=0.01,     # STDP amplitude
            a_minus=0.012,   # Slightly asymmetric
            pre_trace=0.0,
            post_trace=0.0,
            activation_count=0,
            last_activation=None,
            created_at=datetime.utcnow()
        )
    
    # ===== SIMULATION =====
    
    async def simulate_network(
        self,
        network_id: str,
        input_spikes: List[SpikeEvent],
        simulation_time: float = 100.0
    ) -> Dict[str, Any]:
        """Simulate a spiking network"""
        if network_id not in self.networks:
            raise ValueError(f"Network not found: {network_id}")
        
        network = self.networks[network_id]
        output_spikes = []
        
        # Reset network state
        await self._reset_network_state(network)
        
        # Add input spikes to queue
        for spike in input_spikes:
            self.spike_queues[network_id].append(spike)
        
        # Simulation loop
        current_time = 0.0
        while current_time < simulation_time:
            # Process spikes at current time
            spikes_generated = await self._simulate_time_step(network, current_time)
            
            # Record output spikes
            for spike in spikes_generated:
                if spike.neuron_id.startswith("output_"):
                    output_spikes.append(spike)
            
            current_time += network.time_step
        
        # Calculate results
        results = {
            "network_id": network_id,
            "simulation_time": simulation_time,
            "output_spikes": [asdict(spike) for spike in output_spikes],
            "total_spikes": network.total_spikes,
            "total_energy": network.total_energy,
            "spike_rate": network.total_spikes / simulation_time * 1000,  # Hz
            "energy_efficiency": network.total_spikes / max(network.total_energy, 1e-6)
        }
        
        # Update metrics
        self.metrics["total_spikes_processed"] += network.total_spikes
        self.metrics["total_energy_consumed"] += network.total_energy
        
        return results
    
    async def _simulate_time_step(self, network: SpikingNetwork, current_time: float) -> List[SpikeEvent]:
        """Simulate one time step, including adaptive anomaly detection."""
        generated_spikes = []
        
        # Process pending spikes
        pending_spikes = []
        while self.spike_queues[network.network_id]:
            spike = self.spike_queues[network.network_id].popleft()
            if abs(spike.timestamp - current_time) < network.time_step / 2:
                pending_spikes.append(spike)
            else:
                # Put back if not ready
                self.spike_queues[network.network_id].appendleft(spike)
                break
        
        # Update neuron states
        for neuron_id, neuron in network.neurons.items():
            # Collect synaptic inputs
            synaptic_current = 0.0
            for spike in pending_spikes:
                if spike.neuron_id == neuron_id:
                    synaptic_current += spike.intensity
            
            # Update membrane potential
            new_spike = await self._update_neuron(neuron, synaptic_current, current_time, network.time_step)
            
            if new_spike:
                generated_spikes.append(new_spike)
                
                # Generate delayed spikes for postsynaptic neurons
                for synapse in network.synapses.values():
                    if synapse.pre_neuron_id == neuron_id:
                        delayed_spike = SpikeEvent(
                            neuron_id=synapse.post_neuron_id,
                            timestamp=current_time + synapse.delay,
                            intensity=synapse.weight,
                            source_id=neuron_id
                        )
                        self.spike_queues[network.network_id].append(delayed_spike)
        
        # --- Anomaly Detection through Adaptive Surprise ---
        num_spikes_this_step = len(generated_spikes)
        
        # 1. Update the network's activity history
        activity_history = self.network_activity_history[network.network_id]
        activity_history.append(num_spikes_this_step)

        # 2. Calculate the adaptive baseline (expected firing rate)
        # We need enough history to establish a stable baseline.
        expected_spikes = 0.0
        if len(activity_history) > 50:
            expected_spikes = np.mean(list(activity_history))
        
        # 3. Calculate "Surprise" score against the adaptive baseline
        surprise = 0
        if expected_spikes > 0.1: # Avoid division by zero or near-zero
            surprise = (num_spikes_this_step - expected_spikes) / expected_spikes
        elif num_spikes_this_step > 2: # A few spikes out of nowhere is surprising
            surprise = 2.0 

        # 4. Detect anomaly if surprise is high
        if surprise > 7.5: # Higher threshold because baseline is now adaptive
            anomaly_details = {
                "timestamp": current_time,
                "pattern": "Coordinated High-Frequency Burst",
                "details": f"Detected {num_spikes_this_step} spikes, but expected ~{expected_spikes:.1f}. Surprise factor: {surprise:.2f}.",
                "severity": min(1.0, surprise / 10.0)
            }
            self.detected_anomalies[network.network_id].append(anomaly_details)
            logger.warning("SNN Anomaly Detected!", anomaly=anomaly_details)
        # --- End Anomaly Detection ---

        # --- NEW: STDP Learning Rule Implementation ---
        # This loop updates synaptic weights based on spike timing.
        for synapse in network.synapses:
            pre_neuron = network.neurons[synapse.pre_neuron_id]
            post_neuron = network.neurons[synapse.post_neuron_id]

            # Check for a recent pre-synaptic spike
            pre_spike_time = self._get_last_spike_time(pre_neuron, current_time)
            if pre_spike_time is None:
                continue

            # Check for a recent post-synaptic spike
            post_spike_time = self._get_last_spike_time(post_neuron, current_time)
            if post_spike_time is None:
                continue

            time_diff = post_spike_time - pre_spike_time
            
            # STDP rule:
            # If pre-synaptic spike occurs just before post-synaptic spike, potentiate (strengthen).
            # If post-synaptic spike occurs just before pre-synaptic spike, depress (weaken).
            if 0 < time_diff <= network.time_step: # Use network.time_step for tau_plus
                # Potentiation (LTP)
                delta_w = network.learning_rate * np.exp(-time_diff / network.time_step) # Use network.time_step for tau_plus
                synapse.weight = min(2.0, synapse.weight + delta_w) # Bounds
            elif -network.time_step <= time_diff < 0: # Use network.time_step for tau_minus
                # Depression (LTD)
                delta_w = -network.learning_rate * np.exp(time_diff / network.time_step) # Use network.time_step for tau_minus
                synapse.weight = max(0.0, synapse.weight + delta_w) # Bounds
        # --- End STDP ---
        
        # Update network statistics
        network.simulation_time = current_time
        network.total_spikes += len(generated_spikes)
        
        return generated_spikes
        
    def _get_last_spike_time(self, neuron: SpikingNeuron, current_time: float) -> Optional[float]:
        """Helper to get the last spike time for a neuron."""
        # This is a conceptual helper. A real implementation would need to store
        # spike times for each neuron within the simulation window.
        if neuron.last_spike_time and current_time - neuron.last_spike_time < 50.0: # Check within 50ms window
             return neuron.last_spike_time
        return None
    
    async def _update_neuron(
        self, 
        neuron: SpikingNeuron, 
        input_current: float, 
        current_time: float, 
        dt: float
    ) -> Optional[SpikeEvent]:
        """Update neuron state and check for spike"""
        
        # Check refractory period
        if (neuron.last_spike_time and 
            current_time - neuron.last_spike_time < neuron.refractory_period):
            return None
        
        # Leaky integrate-and-fire dynamics
        if neuron.neuron_type == NeuronType.LEAKY_INTEGRATE_FIRE:
            # Membrane equation: C * dV/dt = -(V - V_rest)/R + I
            tau_membrane = neuron.membrane_resistance * neuron.membrane_capacitance
            
            # Exponential decay towards resting potential
            decay_factor = np.exp(-dt / tau_membrane)
            
            # Update membrane potential
            neuron.membrane_potential = (
                neuron.membrane_potential * decay_factor +
                neuron.resting_potential * (1 - decay_factor) +
                input_current * neuron.membrane_resistance * (1 - decay_factor)
            )
            
            # Energy consumption (simplified)
            neuron.energy_consumption += abs(input_current) * dt * 0.001  # pJ
            
            # Check for spike
            if neuron.membrane_potential >= neuron.threshold:
                # Generate spike
                spike = SpikeEvent(
                    neuron_id=neuron.neuron_id,
                    timestamp=current_time,
                    intensity=1.0
                )
                
                # Reset membrane potential
                neuron.membrane_potential = neuron.reset_potential
                neuron.last_spike_time = current_time
                neuron.spike_count += 1
                
                # Additional energy for spike generation
                neuron.energy_consumption += 1.0  # pJ per spike
                
                return spike
        
        return None
    
    # ===== PLASTICITY =====
    
    async def update_plasticity(self, network_id: str, pre_spike: SpikeEvent, post_spike: SpikeEvent):
        """Update synaptic plasticity based on spike timing"""
        if network_id not in self.networks:
            return
        
        network = self.networks[network_id]
        if not network.plasticity_enabled:
            return
        
        # Find synapse between neurons
        synapse_id = f"syn_{pre_spike.neuron_id}_{post_spike.neuron_id}"
        if synapse_id not in network.synapses:
            return
        
        synapse = network.synapses[synapse_id]
        
        # STDP rule
        if synapse.plasticity_type == PlasticityType.STDP:
            dt = post_spike.timestamp - pre_spike.timestamp
            
            if dt > 0:  # Post before pre (depression)
                weight_change = -synapse.a_minus * np.exp(-dt / synapse.tau_minus)
            else:  # Pre before post (potentiation)
                weight_change = synapse.a_plus * np.exp(dt / synapse.tau_plus)
            
            # Update weight
            synapse.weight += weight_change
            synapse.weight = np.clip(synapse.weight, 0.0, 2.0)  # Bounds
            
            # Update traces
            synapse.pre_trace += 1.0
            synapse.post_trace += 1.0
            
            self.metrics["plasticity_events"] += 1
    
    # ===== BACKGROUND TASKS =====
    
    async def _simulation_loop(self):
        """Background simulation processing"""
        while True:
            try:
                await asyncio.sleep(0.01)  # High frequency for real-time processing
                
                # Update spike traces for all networks
                for network in self.networks.values():
                    await self._decay_traces(network)
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
    
    async def _plasticity_update_loop(self):
        """Background plasticity updates"""
        while True:
            try:
                await asyncio.sleep(1.0)  # Every second
                
                # Apply homeostatic plasticity
                for network in self.networks.values():
                    if network.homeostasis_enabled:
                        await self._apply_homeostatic_plasticity(network)
                
            except Exception as e:
                logger.error(f"Error in plasticity update loop: {e}")
    
    async def _decay_traces(self, network: SpikingNetwork):
        """Decay synaptic traces"""
        dt = network.time_step
        
        for synapse in network.synapses.values():
            # Exponential decay
            synapse.pre_trace *= np.exp(-dt / synapse.tau_plus)
            synapse.post_trace *= np.exp(-dt / synapse.tau_minus)
    
    async def _apply_homeostatic_plasticity(self, network: SpikingNetwork):
        """Apply homeostatic plasticity to maintain stable activity"""
        target_rate = 10.0  # Hz
        
        for neuron in network.neurons.values():
            if neuron.spike_count > 0:
                current_rate = neuron.spike_count / max(network.simulation_time / 1000, 1e-6)
                
                # Adjust threshold based on activity
                if current_rate > target_rate:
                    neuron.threshold += 0.1  # Make harder to spike
                elif current_rate < target_rate:
                    neuron.threshold -= 0.1  # Make easier to spike
                
                neuron.threshold = np.clip(neuron.threshold, -60.0, -40.0)
    
    async def _reset_network_state(self, network: SpikingNetwork):
        """Reset network to initial state"""
        for neuron in network.neurons.values():
            neuron.membrane_potential = neuron.resting_potential
            neuron.last_spike_time = None
            neuron.spike_count = 0
            neuron.energy_consumption = 0.0
        
        for synapse in network.synapses.values():
            synapse.pre_trace = 0.0
            synapse.post_trace = 0.0
            synapse.activation_count = 0
        
        network.simulation_time = 0.0
        network.total_spikes = 0
        network.total_energy = 0.0
    
    # ===== ENERGY ANALYSIS =====
    
    async def get_energy_analysis(self, network_id: str) -> Dict[str, Any]:
        """Get energy consumption analysis"""
        if network_id not in self.networks:
            return {}
        
        network = self.networks[network_id]
        
        # Calculate energy metrics
        total_neuron_energy = sum(n.energy_consumption for n in network.neurons.values())
        total_spikes = sum(n.spike_count for n in network.neurons.values())
        
        avg_energy_per_spike = total_neuron_energy / max(total_spikes, 1)
        energy_efficiency = total_spikes / max(total_neuron_energy, 1e-6)
        
        return {
            "network_id": network_id,
            "total_energy": total_neuron_energy,
            "total_spikes": total_spikes,
            "energy_per_spike": avg_energy_per_spike,
            "energy_efficiency": energy_efficiency,
            "neuron_energy_breakdown": {
                neuron_id: neuron.energy_consumption 
                for neuron_id, neuron in network.neurons.items()
            }
        }
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics"""
        topics = [
            "q.neuromorphic.spike.generated",
            "q.neuromorphic.plasticity.updated",
            "q.neuromorphic.network.created"
        ]
        
        for topic in topics:
            await self.pulsar_service.ensure_topic(topic)

# Global service instance  
spiking_neural_networks = SpikingNeuralNetworksService() 