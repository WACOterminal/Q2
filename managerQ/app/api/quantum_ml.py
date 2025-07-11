"""
Quantum ML API

This module provides FastAPI endpoints for the quantum ML service:
- Quantum machine learning algorithms
- Quantum model management
- Quantum task execution
- Backend management
- Performance monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import logging
import numpy as np

from ..core.quantum_ml_service import (
    QuantumMLService,
    QuantumTask,
    QuantumModel,
    QuantumAlgorithmType,
    QuantumBackend,
    QuantumTaskStatus,
    OptimizationMethod,
    quantum_ml_service
)
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

# Create the router
router = APIRouter(prefix="/api/v1/quantum-ml", tags=["quantum-ml"])

# ===== PYDANTIC MODELS =====

class QuantumNeuralNetworkRequest(BaseModel):
    """Quantum neural network request"""
    data: List[List[float]]
    labels: List[int]
    num_qubits: int = Field(ge=2, le=32, default=4)
    num_layers: int = Field(ge=1, le=10, default=2)
    backend: Optional[QuantumBackend] = None

class QuantumSVMRequest(BaseModel):
    """Quantum SVM request"""
    data: List[List[float]]
    labels: List[int]
    feature_map_type: str = "ZZFeatureMap"
    backend: Optional[QuantumBackend] = None

class QuantumClusteringRequest(BaseModel):
    """Quantum clustering request"""
    data: List[List[float]]
    num_clusters: int = Field(ge=2, le=10, default=2)
    backend: Optional[QuantumBackend] = None

class QuantumPCARequest(BaseModel):
    """Quantum PCA request"""
    data: List[List[float]]
    num_components: int = Field(ge=1, le=10, default=2)
    backend: Optional[QuantumBackend] = None

class QuantumOptimizationRequest(BaseModel):
    """Quantum optimization request"""
    cost_function: str
    num_variables: int = Field(ge=1, le=20)
    method: OptimizationMethod = OptimizationMethod.COBYLA
    backend: Optional[QuantumBackend] = None

class ModelPredictionRequest(BaseModel):
    """Model prediction request"""
    model_id: str
    data: List[List[float]]

class ModelSaveRequest(BaseModel):
    """Model save request"""
    task_id: str
    model_name: str

# ===== QUANTUM ALGORITHM ENDPOINTS =====

@router.post("/algorithms/quantum-neural-network")
async def quantum_neural_network(
    request: QuantumNeuralNetworkRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Train a quantum neural network"""
    try:
        data = np.array(request.data)
        labels = np.array(request.labels)
        
        task_id = await quantum_ml_service.quantum_neural_network(
            data=data,
            labels=labels,
            num_qubits=request.num_qubits,
            num_layers=request.num_layers
        )
        
        logger.info(f"Quantum neural network task created: {task_id} by {user.get('username', 'unknown')}")
        
        return JSONResponse(
            content={"message": "Quantum neural network task created", "task_id": task_id},
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error creating quantum neural network: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/algorithms/quantum-svm")
async def quantum_support_vector_machine(
    request: QuantumSVMRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Train a quantum support vector machine"""
    try:
        data = np.array(request.data)
        labels = np.array(request.labels)
        
        task_id = await quantum_ml_service.quantum_support_vector_machine(
            data=data,
            labels=labels,
            feature_map_type=request.feature_map_type
        )
        
        logger.info(f"Quantum SVM task created: {task_id} by {user.get('username', 'unknown')}")
        
        return JSONResponse(
            content={"message": "Quantum SVM task created", "task_id": task_id},
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error creating quantum SVM: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/algorithms/quantum-clustering")
async def quantum_clustering(
    request: QuantumClusteringRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Perform quantum clustering"""
    try:
        data = np.array(request.data)
        
        task_id = await quantum_ml_service.quantum_clustering(
            data=data,
            num_clusters=request.num_clusters
        )
        
        logger.info(f"Quantum clustering task created: {task_id} by {user.get('username', 'unknown')}")
        
        return JSONResponse(
            content={"message": "Quantum clustering task created", "task_id": task_id},
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error creating quantum clustering: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/algorithms/quantum-pca")
async def quantum_principal_component_analysis(
    request: QuantumPCARequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Perform quantum PCA"""
    try:
        data = np.array(request.data)
        
        task_id = await quantum_ml_service.quantum_pca(
            data=data,
            num_components=request.num_components
        )
        
        logger.info(f"Quantum PCA task created: {task_id} by {user.get('username', 'unknown')}")
        
        return JSONResponse(
            content={"message": "Quantum PCA task created", "task_id": task_id},
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error creating quantum PCA: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/algorithms/quantum-optimization")
async def quantum_optimization(
    request: QuantumOptimizationRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Perform quantum optimization"""
    try:
        task_id = await quantum_ml_service.quantum_optimization(
            cost_function=request.cost_function,
            num_variables=request.num_variables,
            method=request.method
        )
        
        logger.info(f"Quantum optimization task created: {task_id} by {user.get('username', 'unknown')}")
        
        return JSONResponse(
            content={"message": "Quantum optimization task created", "task_id": task_id},
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error creating quantum optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== TASK MANAGEMENT ENDPOINTS =====

@router.get("/tasks")
async def get_tasks(
    algorithm_type: Optional[QuantumAlgorithmType] = None,
    status: Optional[QuantumTaskStatus] = None,
    limit: int = Query(100, ge=1, le=1000),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get quantum tasks"""
    try:
        tasks = await quantum_ml_service.list_tasks(algorithm_type, status)
        
        # Apply limit
        tasks = tasks[-limit:]
        
        # Convert to dict
        result = []
        for task in tasks:
            task_dict = {
                "task_id": task.task_id,
                "algorithm_type": task.algorithm_type.value,
                "backend": task.backend.value,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "execution_time": task.execution_time,
                "error_message": task.error_message
            }
            result.append(task_dict)
        
        return JSONResponse(content={"tasks": result})
        
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get task details"""
    try:
        task = await quantum_ml_service.get_task_status(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_dict = {
            "task_id": task.task_id,
            "algorithm_type": task.algorithm_type.value,
            "backend": task.backend.value,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "execution_time": task.execution_time,
            "result": task.result,
            "error_message": task.error_message,
            "circuit_config": {
                "num_qubits": task.circuit_config.num_qubits,
                "num_layers": task.circuit_config.num_layers,
                "entanglement": task.circuit_config.entanglement
            }
        }
        
        return JSONResponse(content=task_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Cancel a task"""
    try:
        success = await quantum_ml_service.cancel_task(task_id)
        
        if success:
            logger.info(f"Task {task_id} cancelled by {user.get('username', 'unknown')}")
            return JSONResponse(content={"message": "Task cancelled successfully"})
        else:
            raise HTTPException(status_code=404, detail="Task not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== MODEL MANAGEMENT ENDPOINTS =====

@router.post("/models")
async def save_model(
    request: ModelSaveRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Save a trained quantum model"""
    try:
        model_id = await quantum_ml_service.save_model(
            request.task_id,
            request.model_name
        )
        
        logger.info(f"Model {model_id} saved by {user.get('username', 'unknown')}")
        
        return JSONResponse(
            content={"message": "Model saved successfully", "model_id": model_id},
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/models")
async def get_models(
    algorithm_type: Optional[QuantumAlgorithmType] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get quantum models"""
    try:
        models = list(quantum_ml_service.models.values())
        
        # Apply filter
        if algorithm_type:
            models = [m for m in models if m.algorithm_type == algorithm_type]
        
        # Convert to dict
        result = []
        for model in models:
            model_dict = {
                "model_id": model.model_id,
                "name": model.name,
                "algorithm_type": model.algorithm_type.value,
                "backend": model.backend.value,
                "created_at": model.created_at.isoformat(),
                "performance_metrics": model.performance_metrics,
                "circuit_config": {
                    "num_qubits": model.circuit_config.num_qubits,
                    "num_layers": model.circuit_config.num_layers
                }
            }
            result.append(model_dict)
        
        return JSONResponse(content={"models": result})
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/models/{model_id}")
async def get_model(
    model_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get model details"""
    try:
        model = await quantum_ml_service.load_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_dict = {
            "model_id": model.model_id,
            "name": model.name,
            "algorithm_type": model.algorithm_type.value,
            "backend": model.backend.value,
            "created_at": model.created_at.isoformat(),
            "performance_metrics": model.performance_metrics,
            "training_history": model.training_history,
            "circuit_config": {
                "num_qubits": model.circuit_config.num_qubits,
                "num_layers": model.circuit_config.num_layers,
                "entanglement": model.circuit_config.entanglement
            }
        }
        
        return JSONResponse(content=model_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/models/{model_id}/predict")
async def predict(
    model_id: str,
    request: ModelPredictionRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Make predictions with a quantum model"""
    try:
        data = np.array(request.data)
        
        result = await quantum_ml_service.predict(model_id, data)
        
        return JSONResponse(content={"predictions": result})
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== SYSTEM STATUS ENDPOINTS =====

@router.get("/system/status")
async def get_system_status(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get system status"""
    try:
        status = await quantum_ml_service.get_system_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/system/algorithms")
async def get_available_algorithms(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get available quantum algorithms"""
    try:
        algorithms = await quantum_ml_service.get_available_algorithms()
        return JSONResponse(content={"algorithms": algorithms})
        
    except Exception as e:
        logger.error(f"Error getting available algorithms: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/system/backends")
async def get_backend_info(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get backend information"""
    try:
        backend_info = await quantum_ml_service.get_backend_info()
        return JSONResponse(content=backend_info)
        
    except Exception as e:
        logger.error(f"Error getting backend info: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={"status": "healthy", "service": "quantum-ml"})

# ===== INITIALIZATION FUNCTIONS =====

async def initialize_quantum_ml_api():
    """Initialize the quantum ML API"""
    try:
        await quantum_ml_service.initialize()
        logger.info("Quantum ML API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Quantum ML API: {e}")
        raise

async def shutdown_quantum_ml_api():
    """Shutdown the quantum ML API"""
    try:
        await quantum_ml_service.shutdown()
        logger.info("Quantum ML API shutdown successfully")
    except Exception as e:
        logger.error(f"Error shutting down Quantum ML API: {e}") 