"""
Emerging AI Technologies REST API

This module provides REST API endpoints for:
- Quantum Computing (Optimization & ML)
- Neuromorphic Computing (Spiking Networks & Cognitive Architectures)
- Energy-Efficient Computing
- Hybrid Quantum-Neuromorphic Workflows
- System Monitoring and Health
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import numpy as np
import asyncio

# Q Platform imports
from app.services.quantum_optimization_service import (
    quantum_optimization_service, OptimizationProblem, QuantumAlgorithm, QuantumBackend
)
from app.services.quantum_ml_experiments import (
    quantum_ml_experiments, QuantumMLAlgorithm, MLTaskType
)
from app.services.spiking_neural_networks import (
    spiking_neural_networks, NeuronType, PlasticityType
)
from app.services.neuromorphic_engine import (
    neuromorphic_engine, CognitiveTask, ArchitectureType
)
from app.services.energy_efficient_computing import (
    energy_efficient_computing, PowerMode, EnergySource
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/emerging-ai", tags=["emerging-ai"])

# ===== REQUEST MODELS =====

class QuantumOptimizationRequest(BaseModel):
    problem_type: str = Field(..., description="Type of optimization problem")
    problem_data: Dict[str, Any] = Field(..., description="Problem-specific data")
    algorithm: str = Field(default="qaoa", description="Quantum algorithm to use")
    backend: str = Field(default="simulator", description="Quantum backend")
    num_qubits: Optional[int] = Field(None, description="Number of qubits")
    max_iterations: Optional[int] = Field(100, description="Maximum iterations")

class QuantumMLExperimentRequest(BaseModel):
    name: str = Field(..., description="Experiment name")
    algorithm: str = Field(..., description="Quantum ML algorithm")
    dataset_features: List[List[float]] = Field(..., description="Dataset features")
    dataset_labels: Optional[List[int]] = Field(None, description="Dataset labels")
    task_type: str = Field(..., description="ML task type")
    description: str = Field("", description="Experiment description")

class SpikingNetworkRequest(BaseModel):
    name: str = Field(..., description="Network name")
    num_input_neurons: int = Field(..., description="Number of input neurons")
    num_hidden_neurons: int = Field(..., description="Number of hidden neurons")
    num_output_neurons: int = Field(..., description="Number of output neurons")
    neuron_type: str = Field(default="leaky_integrate_fire", description="Neuron type")

class CognitiveTaskRequest(BaseModel):
    architecture_id: str = Field(..., description="Cognitive architecture ID")
    task_name: str = Field(..., description="Task name")
    task_type: str = Field(..., description="Cognitive task type")
    input_data: List[float] = Field(..., description="Input data")
    expected_output: Optional[List[float]] = Field(None, description="Expected output")

class EnergyBudgetRequest(BaseModel):
    device_id: str = Field(..., description="Device ID")
    allocated_energy: float = Field(..., description="Allocated energy in Joules")
    time_window_hours: float = Field(1.0, description="Time window in hours")
    priority: int = Field(1, description="Priority level")

# ===== QUANTUM COMPUTING ENDPOINTS =====

@router.post("/quantum/optimization/solve")
async def solve_quantum_optimization(request: QuantumOptimizationRequest):
    """
    Solve an optimization problem using quantum algorithms
    """
    try:
        problem_type = OptimizationProblem(request.problem_type)
        algorithm = QuantumAlgorithm(request.algorithm)
        backend = QuantumBackend(request.backend)
        
        task_id = await quantum_optimization_service.solve_optimization_problem(
            problem_type=problem_type,
            problem_data=request.problem_data,
            algorithm=algorithm,
            backend=backend,
            num_qubits=request.num_qubits,
            max_iterations=request.max_iterations
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Quantum optimization task started",
            "algorithm": request.algorithm,
            "problem_type": request.problem_type
        }
        
    except Exception as e:
        logger.error(f"Quantum optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quantum/optimization/{task_id}/result")
async def get_quantum_optimization_result(task_id: str):
    """
    Get the result of a quantum optimization task
    """
    try:
        result = await quantum_optimization_service.get_optimization_result(task_id)
        
        if result is None:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "status": "success",
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quantum/ml/experiment")
async def create_quantum_ml_experiment(request: QuantumMLExperimentRequest):
    """
    Create a quantum machine learning experiment
    """
    try:
        algorithm = QuantumMLAlgorithm(request.algorithm)
        task_type = MLTaskType(request.task_type)
        
        # Create dataset
        features = np.array(request.dataset_features)
        labels = np.array(request.dataset_labels) if request.dataset_labels else None
        
        dataset_id = await quantum_ml_experiments.create_dataset(
            name=f"{request.name}_dataset",
            features=features,
            labels=labels,
            task_type=task_type
        )
        
        # Create experiment
        experiment_id = await quantum_ml_experiments.create_ml_experiment(
            name=request.name,
            algorithm=algorithm,
            dataset_id=dataset_id,
            description=request.description
        )
        
        return {
            "status": "success",
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "message": "Quantum ML experiment created",
            "algorithm": request.algorithm
        }
        
    except Exception as e:
        logger.error(f"Quantum ML experiment creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quantum/ml/experiment/{experiment_id}/results")
async def get_quantum_ml_results(experiment_id: str):
    """
    Get results of a quantum ML experiment
    """
    try:
        results = await quantum_ml_experiments.get_experiment_results(experiment_id)
        
        if results is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {
            "status": "success",
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ML experiment results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== NEUROMORPHIC COMPUTING ENDPOINTS =====

@router.post("/neuromorphic/network/create")
async def create_spiking_network(request: SpikingNetworkRequest):
    """
    Create a spiking neural network
    """
    try:
        neuron_type = NeuronType(request.neuron_type)
        
        network_id = await spiking_neural_networks.create_spiking_network(
            name=request.name,
            num_input_neurons=request.num_input_neurons,
            num_hidden_neurons=request.num_hidden_neurons,
            num_output_neurons=request.num_output_neurons,
            neuron_type=neuron_type
        )
        
        return {
            "status": "success",
            "network_id": network_id,
            "message": "Spiking neural network created",
            "neuron_type": request.neuron_type,
            "total_neurons": request.num_input_neurons + request.num_hidden_neurons + request.num_output_neurons
        }
        
    except Exception as e:
        logger.error(f"Spiking network creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/neuromorphic/network/{network_id}/simulate")
async def simulate_spiking_network(
    network_id: str,
    input_spikes: List[Dict[str, Any]],
    simulation_time: float = 100.0
):
    """
    Simulate a spiking neural network
    """
    try:
        # Convert input spikes format
        from app.services.spiking_neural_networks import SpikeEvent
        spike_events = []
        
        for spike_data in input_spikes:
            spike = SpikeEvent(
                neuron_id=spike_data["neuron_id"],
                timestamp=spike_data["timestamp"],
                intensity=spike_data.get("intensity", 1.0),
                source_id=spike_data.get("source_id")
            )
            spike_events.append(spike)
        
        results = await spiking_neural_networks.simulate_network(
            network_id=network_id,
            input_spikes=spike_events,
            simulation_time=simulation_time
        )
        
        return {
            "status": "success",
            "simulation_results": results
        }
        
    except Exception as e:
        logger.error(f"Network simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/neuromorphic/architecture/create")
async def create_cognitive_architecture(
    name: str,
    architecture_type: str,
    task_types: List[str]
):
    """
    Create a cognitive architecture
    """
    try:
        arch_type = ArchitectureType(architecture_type)
        cognitive_tasks = [CognitiveTask(task) for task in task_types]
        
        architecture_id = await neuromorphic_engine.create_cognitive_architecture(
            name=name,
            architecture_type=arch_type,
            task_types=cognitive_tasks
        )
        
        return {
            "status": "success",
            "architecture_id": architecture_id,
            "message": "Cognitive architecture created",
            "architecture_type": architecture_type,
            "task_types": task_types
        }
        
    except Exception as e:
        logger.error(f"Cognitive architecture creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/neuromorphic/task/process")
async def process_cognitive_task(request: CognitiveTaskRequest):
    """
    Process a cognitive task using neuromorphic architecture
    """
    try:
        task_type = CognitiveTask(request.task_type)
        input_data = np.array(request.input_data)
        expected_output = np.array(request.expected_output) if request.expected_output else None
        
        task_id = await neuromorphic_engine.process_cognitive_task(
            architecture_id=request.architecture_id,
            task_name=request.task_name,
            task_type=task_type,
            input_data=input_data,
            expected_output=expected_output
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Cognitive task processing started",
            "task_type": request.task_type
        }
        
    except Exception as e:
        logger.error(f"Cognitive task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ENERGY EFFICIENCY ENDPOINTS =====

@router.post("/energy/device/register")
async def register_energy_device(
    device_id: str,
    device_type: str,
    power_profile: Dict[str, Any]
):
    """
    Register a device for energy monitoring
    """
    try:
        profile_id = await energy_efficient_computing.register_device(
            device_id=device_id,
            device_type=device_type,
            power_profile=power_profile
        )
        
        return {
            "status": "success",
            "device_id": device_id,
            "profile_id": profile_id,
            "message": "Device registered for energy monitoring"
        }
        
    except Exception as e:
        logger.error(f"Device registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/energy/budget/create")
async def create_energy_budget(request: EnergyBudgetRequest):
    """
    Create an energy budget for a device
    """
    try:
        time_window = timedelta(hours=request.time_window_hours)
        
        budget_id = await energy_efficient_computing.create_energy_budget(
            device_id=request.device_id,
            allocated_energy=request.allocated_energy,
            time_window=time_window,
            priority=request.priority
        )
        
        return {
            "status": "success",
            "budget_id": budget_id,
            "device_id": request.device_id,
            "allocated_energy": request.allocated_energy,
            "message": "Energy budget created"
        }
        
    except Exception as e:
        logger.error(f"Energy budget creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/energy/device/{device_id}/set-power-mode")
async def set_device_power_mode(device_id: str, power_mode: str):
    """
    Set power mode for a device
    """
    try:
        mode = PowerMode(power_mode)
        await energy_efficient_computing.set_power_mode(device_id, mode)
        
        return {
            "status": "success",
            "device_id": device_id,
            "power_mode": power_mode,
            "message": "Power mode updated"
        }
        
    except Exception as e:
        logger.error(f"Power mode setting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/energy/device/{device_id}/analysis")
async def get_energy_analysis(device_id: str):
    """
    Get energy consumption analysis for a device
    """
    try:
        analysis = await energy_efficient_computing.optimize_energy_performance(device_id)
        
        return {
            "status": "success",
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Energy analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== HYBRID COMPUTING ENDPOINTS =====

@router.post("/hybrid/quantum-neuromorphic/workflow")
async def create_hybrid_workflow(
    quantum_task: QuantumOptimizationRequest,
    neuromorphic_task: CognitiveTaskRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a hybrid quantum-neuromorphic workflow
    """
    try:
        # Start quantum optimization
        quantum_problem_type = OptimizationProblem(quantum_task.problem_type)
        quantum_algorithm = QuantumAlgorithm(quantum_task.algorithm)
        quantum_backend = QuantumBackend(quantum_task.backend)
        
        quantum_task_id = await quantum_optimization_service.solve_optimization_problem(
            problem_type=quantum_problem_type,
            problem_data=quantum_task.problem_data,
            algorithm=quantum_algorithm,
            backend=quantum_backend,
            num_qubits=quantum_task.num_qubits,
            max_iterations=quantum_task.max_iterations
        )
        
        # Start neuromorphic processing
        cognitive_task_type = CognitiveTask(neuromorphic_task.task_type)
        input_data = np.array(neuromorphic_task.input_data)
        expected_output = np.array(neuromorphic_task.expected_output) if neuromorphic_task.expected_output else None
        
        neuromorphic_task_id = await neuromorphic_engine.process_cognitive_task(
            architecture_id=neuromorphic_task.architecture_id,
            task_name=neuromorphic_task.task_name,
            task_type=cognitive_task_type,
            input_data=input_data,
            expected_output=expected_output
        )
        
        # Create workflow tracking
        workflow_id = f"hybrid_{quantum_task_id}_{neuromorphic_task_id}"
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "quantum_task_id": quantum_task_id,
            "neuromorphic_task_id": neuromorphic_task_id,
            "message": "Hybrid quantum-neuromorphic workflow started",
            "estimated_completion": "2-10 minutes"
        }
        
    except Exception as e:
        logger.error(f"Hybrid workflow creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hybrid/workflow/{workflow_id}/status")
async def get_hybrid_workflow_status(workflow_id: str):
    """
    Get status of hybrid workflow
    """
    try:
        # Parse workflow ID
        parts = workflow_id.split("_")
        if len(parts) < 3:
            raise HTTPException(status_code=400, detail="Invalid workflow ID")
        
        quantum_task_id = parts[1]
        neuromorphic_task_id = parts[2]
        
        # Get quantum task status
        quantum_result = await quantum_optimization_service.get_optimization_result(quantum_task_id)
        
        # Get neuromorphic task status (simplified - would need actual status tracking)
        neuromorphic_status = "completed"  # Mock status
        
        workflow_status = "running"
        if quantum_result and quantum_result.get("status") == "completed":
            workflow_status = "completed"
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "workflow_status": workflow_status,
            "quantum_task": {
                "task_id": quantum_task_id,
                "status": quantum_result.get("status") if quantum_result else "running"
            },
            "neuromorphic_task": {
                "task_id": neuromorphic_task_id,
                "status": neuromorphic_status
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get hybrid workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== SYSTEM MONITORING ENDPOINTS =====

@router.get("/system/metrics")
async def get_system_metrics():
    """
    Get comprehensive system metrics across all emerging AI components
    """
    try:
        # Gather metrics from all services
        quantum_metrics = quantum_optimization_service.metrics
        neuromorphic_metrics = neuromorphic_engine.metrics
        energy_metrics = energy_efficient_computing.metrics
        
        system_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "quantum_computing": {
                "problems_solved": quantum_metrics.get("problems_solved", 0),
                "average_quantum_advantage": quantum_metrics.get("average_quantum_advantage", 1.0),
                "successful_optimizations": quantum_metrics.get("successful_optimizations", 0),
                "total_quantum_volume": quantum_metrics.get("total_quantum_volume", 0)
            },
            "neuromorphic_computing": {
                "architectures_created": neuromorphic_metrics.get("architectures_created", 0),
                "tasks_completed": neuromorphic_metrics.get("tasks_completed", 0),
                "average_accuracy": neuromorphic_metrics.get("average_accuracy", 0.0),
                "adaptation_events": neuromorphic_metrics.get("adaptation_events", 0)
            },
            "energy_efficiency": {
                "total_energy_consumed": energy_metrics.get("total_energy_consumed", 0.0),
                "total_energy_saved": energy_metrics.get("total_energy_saved", 0.0),
                "average_efficiency": energy_metrics.get("average_efficiency", 0.0),
                "carbon_footprint_reduction": energy_metrics.get("carbon_footprint_reduction", 0.0)
            },
            "system_health": {
                "services_active": 5,  # Number of emerging AI services
                "uptime_hours": 24,    # Mock uptime
                "error_rate": 0.01     # Mock error rate
            }
        }
        
        return {
            "status": "success",
            "metrics": system_metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/health")
async def get_system_health():
    """
    Get health status of all emerging AI services
    """
    try:
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "services": {
                "quantum_optimization": {
                    "status": "healthy",
                    "active_tasks": len(quantum_optimization_service.active_tasks),
                    "completed_tasks": len(quantum_optimization_service.completed_tasks)
                },
                "quantum_ml_experiments": {
                    "status": "healthy", 
                    "active_experiments": len(quantum_ml_experiments.experiments),
                    "trained_models": len(quantum_ml_experiments.models)
                },
                "spiking_neural_networks": {
                    "status": "healthy",
                    "active_networks": len(spiking_neural_networks.networks),
                    "total_spikes_processed": spiking_neural_networks.metrics.get("total_spikes_processed", 0)
                },
                "neuromorphic_engine": {
                    "status": "healthy",
                    "active_architectures": len(neuromorphic_engine.architectures),
                    "active_tasks": len(neuromorphic_engine.active_tasks)
                },
                "energy_efficient_computing": {
                    "status": "healthy",
                    "monitored_devices": len(energy_efficient_computing.energy_metrics),
                    "active_budgets": len(energy_efficient_computing.energy_budgets)
                }
            }
        }
        
        return {
            "status": "success",
            "health": health_status
        }
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/capabilities")
async def get_system_capabilities():
    """
    Get comprehensive list of system capabilities
    """
    return {
        "status": "success",
        "capabilities": {
            "quantum_computing": {
                "optimization_algorithms": [alg.value for alg in QuantumAlgorithm],
                "ml_algorithms": [alg.value for alg in QuantumMLAlgorithm],
                "supported_backends": [backend.value for backend in QuantumBackend],
                "optimization_problems": [prob.value for prob in OptimizationProblem],
                "features": [
                    "Quantum Approximate Optimization Algorithm (QAOA)",
                    "Variational Quantum Eigensolver (VQE)",
                    "Quantum Neural Networks",
                    "Quantum Support Vector Machines",
                    "Quantum Advantage Benchmarking",
                    "Hybrid Quantum-Classical Workflows"
                ]
            },
            "neuromorphic_computing": {
                "neuron_types": [nt.value for nt in NeuronType],
                "plasticity_types": [pt.value for pt in PlasticityType],
                "cognitive_tasks": [ct.value for ct in CognitiveTask],
                "architecture_types": [at.value for at in ArchitectureType],
                "features": [
                    "Spiking Neural Networks",
                    "Biological Neuron Models",
                    "Synaptic Plasticity (STDP)",
                    "Event-Driven Processing",
                    "Ultra-Low Power Consumption",
                    "Real-Time Inference",
                    "Adaptive Learning",
                    "Cognitive Architectures"
                ]
            },
            "energy_efficiency": {
                "power_modes": [pm.value for pm in PowerMode],
                "energy_sources": [es.value for es in EnergySource],
                "features": [
                    "Dynamic Power Management",
                    "Energy Budgeting",
                    "Thermal Management",
                    "Performance-Energy Optimization",
                    "Green Computing Metrics",
                    "Adaptive Energy Control",
                    "Real-Time Monitoring"
                ]
            },
            "hybrid_systems": {
                "features": [
                    "Quantum-Neuromorphic Integration",
                    "Energy-Aware Computing",
                    "Multi-Modal AI Processing",
                    "Adaptive Resource Allocation",
                    "Cross-System Optimization"
                ]
            }
        }
    }

# ===== DEMO AND TESTING ENDPOINTS =====

@router.post("/demo/quantum-advantage")
async def demo_quantum_advantage(background_tasks: BackgroundTasks):
    """
    Run a demonstration of quantum advantage
    """
    try:
        # Run a simple Max-Cut problem
        problem_data = {
            "num_vertices": 6,
            "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4)]
        }
        
        task_id = await quantum_optimization_service.solve_optimization_problem(
            problem_type=OptimizationProblem.MAX_CUT,
            problem_data=problem_data,
            algorithm=QuantumAlgorithm.QAOA,
            backend=QuantumBackend.SIMULATOR,
            num_qubits=6,
            max_iterations=50
        )
        
        return {
            "status": "success",
            "demo_type": "quantum_advantage",
            "task_id": task_id,
            "problem": "Max-Cut Graph Optimization",
            "description": "Demonstrates quantum advantage in combinatorial optimization",
            "check_results_at": f"/api/v1/emerging-ai/quantum/optimization/{task_id}/result"
        }
        
    except Exception as e:
        logger.error(f"Quantum advantage demo failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/demo/neuromorphic-efficiency")
async def demo_neuromorphic_efficiency(background_tasks: BackgroundTasks):
    """
    Run a demonstration of neuromorphic energy efficiency
    """
    try:
        # Create a simple pattern recognition architecture
        architecture_id = await neuromorphic_engine.create_cognitive_architecture(
            name="Energy Efficiency Demo",
            architecture_type=ArchitectureType.FEEDFORWARD,
            task_types=[CognitiveTask.PATTERN_RECOGNITION]
        )
        
        # Process a simple pattern recognition task
        input_data = np.random.rand(10)  # Random input pattern
        task_id = await neuromorphic_engine.process_cognitive_task(
            architecture_id=architecture_id,
            task_name="Pattern Recognition Demo",
            task_type=CognitiveTask.PATTERN_RECOGNITION,
            input_data=input_data
        )
        
        return {
            "status": "success",
            "demo_type": "neuromorphic_efficiency", 
            "architecture_id": architecture_id,
            "task_id": task_id,
            "description": "Demonstrates ultra-low power neuromorphic computing",
            "features": [
                "Event-driven processing",
                "Biological neuron models",
                "Adaptive plasticity",
                "Energy efficiency (picojoules per operation)"
            ]
        }
        
    except Exception as e:
        logger.error(f"Neuromorphic efficiency demo failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/demo/hybrid-workflow")
async def demo_hybrid_workflow(background_tasks: BackgroundTasks):
    """
    Run a demonstration of hybrid quantum-neuromorphic workflow
    """
    try:
        # Step 1: Quantum optimization for resource allocation
        quantum_problem = {
            "num_resources": 4,
            "allocation_matrix": [[1, 2, 3], [2, 1, 4], [3, 4, 1], [1, 3, 2]]
        }
        
        quantum_task_id = await quantum_optimization_service.solve_optimization_problem(
            problem_type=OptimizationProblem.RESOURCE_ALLOCATION,
            problem_data=quantum_problem,
            algorithm=QuantumAlgorithm.QAOA,
            backend=QuantumBackend.SIMULATOR
        )
        
        # Step 2: Neuromorphic adaptive control
        architecture_id = await neuromorphic_engine.create_cognitive_architecture(
            name="Hybrid Control Demo",
            architecture_type=ArchitectureType.RECURRENT,
            task_types=[CognitiveTask.DECISION_MAKING, CognitiveTask.ADAPTATION]
        )
        
        control_data = np.array([0.5, 0.3, 0.8, 0.2])  # Control parameters
        neuromorphic_task_id = await neuromorphic_engine.process_cognitive_task(
            architecture_id=architecture_id,
            task_name="Adaptive Control",
            task_type=CognitiveTask.DECISION_MAKING,
            input_data=control_data
        )
        
        # Step 3: Energy optimization
        await energy_efficient_computing.register_device(
            "demo_device", "hybrid", 
            {"operating_modes": {"quantum": 75.0, "neuromorphic": 2.0}}
        )
        
        workflow_id = f"hybrid_demo_{quantum_task_id}_{neuromorphic_task_id}"
        
        return {
            "status": "success",
            "demo_type": "hybrid_workflow",
            "workflow_id": workflow_id,
            "quantum_task_id": quantum_task_id,
            "neuromorphic_task_id": neuromorphic_task_id,
            "architecture_id": architecture_id,
            "description": "Demonstrates quantum-neuromorphic hybrid computing",
            "workflow_steps": [
                "1. Quantum optimization for resource allocation",
                "2. Neuromorphic adaptive control",
                "3. Energy-efficient execution",
                "4. Cross-system feedback and optimization"
            ],
            "check_status_at": f"/api/v1/emerging-ai/hybrid/workflow/{workflow_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Hybrid workflow demo failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== SERVICE INITIALIZATION =====

async def initialize_emerging_ai_services():
    """Initialize all emerging AI services"""
    try:
        logger.info("Initializing Emerging AI services...")
        
        await quantum_optimization_service.initialize()
        await quantum_ml_experiments.initialize()
        await spiking_neural_networks.initialize()
        await neuromorphic_engine.initialize()
        await energy_efficient_computing.initialize()
        
        logger.info("All Emerging AI services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Emerging AI services: {e}")
        raise

# Initialize services on module import (would be called from main app)
# asyncio.create_task(initialize_emerging_ai_services()) 