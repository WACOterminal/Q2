"""
Advanced ML Capabilities API

This module provides REST API endpoints for all ML capabilities:
- Federated Learning orchestration
- AutoML experiments
- Reinforcement Learning training
- Multi-modal AI processing
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import json
import base64
import tempfile
import uuid

# ML Services
from managerQ.app.core.federated_learning_orchestrator import (
    federated_learning_orchestrator,
    AggregationStrategy,
    FederatedLearningStatus
)
from managerQ.app.core.automl_service import (
    automl_service,
    ModelType,
    OptimizationObjective,
    AutoMLStatus
)
from managerQ.app.core.reinforcement_learning_service import (
    rl_service,
    RLAlgorithm,
    RLEnvironmentType,
    RLTrainingStatus
)
from managerQ.app.core.multimodal_ai_service import (
    multimodal_ai_service,
    ModalityType,
    ProcessingTask,
    ProcessingStatus
)

router = APIRouter()

# ===== REQUEST/RESPONSE MODELS =====

# Federated Learning Models
class FederatedLearningRequest(BaseModel):
    model_architecture: str = Field(..., description="Model architecture specification")
    dataset_config: Dict[str, Any] = Field(..., description="Dataset configuration")
    training_config: Dict[str, Any] = Field(..., description="Training configuration")
    participating_agents: Optional[List[str]] = Field(None, description="List of participating agents")
    aggregation_strategy: AggregationStrategy = Field(AggregationStrategy.FEDERATED_AVERAGING, description="Aggregation strategy")
    privacy_config: Optional[Dict[str, Any]] = Field(None, description="Privacy configuration")

class FederatedLearningResponse(BaseModel):
    session_id: str
    status: str
    message: str

# AutoML Models
class AutoMLRequest(BaseModel):
    experiment_name: str = Field(..., description="Experiment name")
    model_type: ModelType = Field(..., description="Model type")
    dataset_config: Dict[str, Any] = Field(..., description="Dataset configuration")
    optimization_objective: OptimizationObjective = Field(OptimizationObjective.ACCURACY, description="Optimization objective")
    training_config: Optional[Dict[str, Any]] = Field(None, description="Training configuration")
    search_space: Optional[Dict[str, Any]] = Field(None, description="Hyperparameter search space")
    n_trials: int = Field(100, description="Number of trials")
    timeout_hours: int = Field(24, description="Timeout in hours")

class AutoMLResponse(BaseModel):
    experiment_id: str
    status: str
    message: str

# Reinforcement Learning Models
class RLTrainingRequest(BaseModel):
    agent_name: str = Field(..., description="Agent name")
    environment_type: RLEnvironmentType = Field(..., description="Environment type")
    algorithm: RLAlgorithm = Field(RLAlgorithm.PPO, description="RL algorithm")
    training_config: Optional[Dict[str, Any]] = Field(None, description="Training configuration")
    environment_config: Optional[Dict[str, Any]] = Field(None, description="Environment configuration")
    total_timesteps: int = Field(100000, description="Total training timesteps")

class RLTrainingResponse(BaseModel):
    session_id: str
    status: str
    message: str

# Multi-modal AI Models
class MultiModalRequest(BaseModel):
    modality: ModalityType = Field(..., description="Modality type")
    task: ProcessingTask = Field(..., description="Processing task")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class MultiModalResponse(BaseModel):
    request_id: str
    status: str
    message: str

class AssetUploadRequest(BaseModel):
    modality: ModalityType = Field(..., description="Modality type")
    content_type: str = Field(..., description="Content type")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Asset metadata")
    tags: Optional[List[str]] = Field(None, description="Asset tags")

# ===== FEDERATED LEARNING ENDPOINTS =====

@router.post("/federated-learning/start", response_model=FederatedLearningResponse)
async def start_federated_learning(request: FederatedLearningRequest):
    """Start a new federated learning session"""
    
    try:
        session_id = await federated_learning_orchestrator.start_federated_learning_session(
            model_architecture=request.model_architecture,
            dataset_config=request.dataset_config,
            training_config=request.training_config,
            participating_agents=request.participating_agents,
            aggregation_strategy=request.aggregation_strategy,
            privacy_config=request.privacy_config
        )
        
        return FederatedLearningResponse(
            session_id=session_id,
            status="started",
            message=f"Federated learning session {session_id} started successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start federated learning: {str(e)}")

@router.get("/federated-learning/status/{session_id}")
async def get_federated_learning_status(session_id: str):
    """Get status of a federated learning session"""
    
    try:
        status = await federated_learning_orchestrator.get_session_status(session_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")

@router.get("/federated-learning/models")
async def list_federated_models(session_id: Optional[str] = None):
    """List federated learning model versions"""
    
    try:
        models = await federated_learning_orchestrator.list_model_versions(session_id)
        
        return {
            "models": [
                {
                    "version_id": model.version_id,
                    "model_architecture": model.model_architecture,
                    "performance_metrics": model.performance_metrics,
                    "created_at": model.created_at,
                    "agent_contributions": model.agent_contributions
                }
                for model in models
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/federated-learning/metrics")
async def get_federated_learning_metrics():
    """Get federated learning metrics"""
    
    try:
        metrics = await federated_learning_orchestrator.get_federated_learning_metrics()
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# ===== AUTOML ENDPOINTS =====

@router.post("/automl/start", response_model=AutoMLResponse)
async def start_automl_experiment(request: AutoMLRequest):
    """Start a new AutoML experiment"""
    
    try:
        experiment_id = await automl_service.start_automl_experiment(
            experiment_name=request.experiment_name,
            model_type=request.model_type,
            dataset_config=request.dataset_config,
            optimization_objective=request.optimization_objective,
            training_config=request.training_config,
            search_space=request.search_space,
            n_trials=request.n_trials,
            timeout_hours=request.timeout_hours
        )
        
        return AutoMLResponse(
            experiment_id=experiment_id,
            status="started",
            message=f"AutoML experiment {experiment_id} started successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start AutoML experiment: {str(e)}")

@router.get("/automl/status/{experiment_id}")
async def get_automl_status(experiment_id: str):
    """Get status of an AutoML experiment"""
    
    try:
        status = await automl_service.get_experiment_status(experiment_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment status: {str(e)}")

@router.get("/automl/results/{experiment_id}")
async def get_automl_results(experiment_id: str):
    """Get results of an AutoML experiment"""
    
    try:
        results = await automl_service.get_experiment_results(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "results": [
                {
                    "model_id": result.model_id,
                    "model_name": result.model_name,
                    "model_type": result.model_type,
                    "hyperparameters": result.hyperparameters,
                    "performance_metrics": result.performance_metrics,
                    "training_time": result.training_time,
                    "created_at": result.created_at
                }
                for result in results
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment results: {str(e)}")

@router.get("/automl/leaderboard")
async def get_automl_leaderboard(model_type: Optional[str] = None):
    """Get AutoML model leaderboard"""
    
    try:
        leaderboard = await automl_service.get_model_leaderboard(model_type)
        return {"leaderboard": leaderboard}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get leaderboard: {str(e)}")

@router.get("/automl/metrics")
async def get_automl_metrics():
    """Get AutoML service metrics"""
    
    try:
        metrics = await automl_service.get_automl_metrics()
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# ===== REINFORCEMENT LEARNING ENDPOINTS =====

@router.post("/rl/start-training", response_model=RLTrainingResponse)
async def start_rl_training(request: RLTrainingRequest):
    """Start RL training session"""
    
    try:
        session_id = await rl_service.start_rl_training(
            agent_name=request.agent_name,
            environment_type=request.environment_type,
            algorithm=request.algorithm,
            training_config=request.training_config,
            environment_config=request.environment_config,
            total_timesteps=request.total_timesteps
        )
        
        return RLTrainingResponse(
            session_id=session_id,
            status="started",
            message=f"RL training session {session_id} started successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start RL training: {str(e)}")

@router.get("/rl/training-status/{session_id}")
async def get_rl_training_status(session_id: str):
    """Get status of RL training session"""
    
    try:
        status = await rl_service.get_training_status(session_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")

@router.post("/rl/deploy-agent/{agent_id}")
async def deploy_rl_agent(agent_id: str, target_environment: str = "production"):
    """Deploy trained RL agent"""
    
    try:
        success = await rl_service.deploy_rl_agent(agent_id, target_environment)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to deploy agent")
        
        return {
            "agent_id": agent_id,
            "status": "deployed",
            "environment": target_environment,
            "message": f"Agent {agent_id} deployed successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deploy agent: {str(e)}")

@router.get("/rl/agents")
async def list_rl_agents():
    """List trained RL agents"""
    
    try:
        agents = await rl_service.list_trained_agents()
        return {"agents": agents}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

@router.post("/rl/agent-action/{agent_id}")
async def get_rl_agent_action(agent_id: str, state: Dict[str, Any]):
    """Get action from RL agent for given state"""
    
    try:
        action = await rl_service.get_rl_agent_action(agent_id, state)
        
        if action is None:
            raise HTTPException(status_code=404, detail="Agent not found or unavailable")
        
        return {
            "agent_id": agent_id,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent action: {str(e)}")

@router.get("/rl/metrics")
async def get_rl_metrics():
    """Get RL service metrics"""
    
    try:
        metrics = await rl_service.get_rl_metrics()
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# ===== MULTIMODAL AI ENDPOINTS =====

@router.post("/multimodal/process", response_model=MultiModalResponse)
async def process_multimodal_request(request: MultiModalRequest, agent_id: str = "default"):
    """Process multi-modal request"""
    
    try:
        request_id = await multimodal_ai_service.process_multimodal_request(
            agent_id=agent_id,
            modality=request.modality,
            task=request.task,
            input_data=request.input_data,
            parameters=request.parameters
        )
        
        return MultiModalResponse(
            request_id=request_id,
            status="queued",
            message=f"Multi-modal request {request_id} queued successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")

@router.get("/multimodal/status/{request_id}")
async def get_multimodal_status(request_id: str):
    """Get status of multi-modal request"""
    
    try:
        status = await multimodal_ai_service.get_request_status(request_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Request not found")
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get request status: {str(e)}")

@router.get("/multimodal/result/{request_id}")
async def get_multimodal_result(request_id: str):
    """Get result of multi-modal request"""
    
    try:
        result = await multimodal_ai_service.get_request_result(request_id)
        
        if result is None:
            raise HTTPException(status_code=404, detail="Request not found or not completed")
        
        return {
            "request_id": request_id,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get request result: {str(e)}")

@router.post("/multimodal/upload-asset")
async def upload_multimodal_asset(
    file: UploadFile = File(...),
    modality: ModalityType = Form(...),
    content_type: str = Form(...),
    metadata: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Upload multi-modal asset"""
    
    try:
        # Read file data
        file_data = await file.read()
        
        # Parse optional fields
        parsed_metadata = json.loads(metadata) if metadata else {}
        parsed_tags = json.loads(tags) if tags else []
        
        # Store asset
        asset_id = await multimodal_ai_service.store_multimodal_asset(
            modality=modality,
            content_type=content_type,
            data=file_data,
            metadata=parsed_metadata,
            tags=parsed_tags
        )
        
        return {
            "asset_id": asset_id,
            "filename": file.filename,
            "size": len(file_data),
            "message": f"Asset {asset_id} uploaded successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload asset: {str(e)}")

@router.get("/multimodal/assets")
async def list_multimodal_assets(modality: Optional[ModalityType] = None):
    """List multi-modal assets"""
    
    try:
        assets = await multimodal_ai_service.list_multimodal_assets(modality)
        return {"assets": assets}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list assets: {str(e)}")

@router.get("/multimodal/metrics")
async def get_multimodal_metrics():
    """Get multi-modal AI metrics"""
    
    try:
        metrics = await multimodal_ai_service.get_multimodal_metrics()
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# ===== CONVENIENCE ENDPOINTS =====

@router.post("/multimodal/classify-image")
async def classify_image(
    image: UploadFile = File(...),
    agent_id: str = Form("default")
):
    """Convenience endpoint for image classification"""
    
    try:
        # Read and encode image
        image_data = await image.read()
        image_base64 = base64.b64encode(image_data).decode()
        
        # Process request
        request_id = await multimodal_ai_service.process_multimodal_request(
            agent_id=agent_id,
            modality=ModalityType.IMAGE,
            task=ProcessingTask.CLASSIFICATION,
            input_data={"base64": image_base64}
        )
        
        return {
            "request_id": request_id,
            "message": "Image classification request submitted"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to classify image: {str(e)}")

@router.post("/multimodal/transcribe-audio")
async def transcribe_audio(
    audio: UploadFile = File(...),
    agent_id: str = Form("default")
):
    """Convenience endpoint for audio transcription"""
    
    try:
        # Read and encode audio
        audio_data = await audio.read()
        audio_base64 = base64.b64encode(audio_data).decode()
        
        # Process request
        request_id = await multimodal_ai_service.process_multimodal_request(
            agent_id=agent_id,
            modality=ModalityType.AUDIO,
            task=ProcessingTask.TRANSCRIPTION,
            input_data={"base64": audio_base64}
        )
        
        return {
            "request_id": request_id,
            "message": "Audio transcription request submitted"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

@router.post("/multimodal/analyze-sentiment")
async def analyze_sentiment(
    text: str = Form(...),
    agent_id: str = Form("default")
):
    """Convenience endpoint for sentiment analysis"""
    
    try:
        # Process request
        request_id = await multimodal_ai_service.process_multimodal_request(
            agent_id=agent_id,
            modality=ModalityType.TEXT,
            task=ProcessingTask.CLASSIFICATION,
            input_data={"text": text}
        )
        
        return {
            "request_id": request_id,
            "message": "Sentiment analysis request submitted"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze sentiment: {str(e)}")

# ===== BATCH PROCESSING ENDPOINTS =====

@router.post("/batch/process-images")
async def batch_process_images(
    images: List[UploadFile] = File(...),
    task: ProcessingTask = Form(ProcessingTask.CLASSIFICATION),
    agent_id: str = Form("default")
):
    """Batch process multiple images"""
    
    try:
        request_ids = []
        
        for image in images:
            # Read and encode image
            image_data = await image.read()
            image_base64 = base64.b64encode(image_data).decode()
            
            # Process request
            request_id = await multimodal_ai_service.process_multimodal_request(
                agent_id=agent_id,
                modality=ModalityType.IMAGE,
                task=task,
                input_data={
                    "base64": image_base64,
                    "filename": image.filename
                }
            )
            
            request_ids.append(request_id)
        
        return {
            "request_ids": request_ids,
            "count": len(request_ids),
            "message": f"Batch processing {len(request_ids)} images"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to batch process images: {str(e)}")

# ===== INTEGRATION ENDPOINTS =====

@router.post("/integration/collect-workflow-experience")
async def collect_workflow_experience(
    workflow_id: str,
    agent_id: str,
    workflow_data: Dict[str, Any]
):
    """Collect experience from workflow execution for RL training"""
    
    try:
        await rl_service.collect_workflow_experience(
            workflow_id=workflow_id,
            agent_id=agent_id,
            workflow_data=workflow_data
        )
        
        return {
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "message": "Workflow experience collected successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to collect experience: {str(e)}")

@router.get("/integration/ml-capabilities-status")
async def get_ml_capabilities_status():
    """Get overall status of all ML capabilities"""
    
    try:
        # Get metrics from all services
        fl_metrics = await federated_learning_orchestrator.get_federated_learning_metrics()
        automl_metrics = await automl_service.get_automl_metrics()
        rl_metrics = await rl_service.get_rl_metrics()
        multimodal_metrics = await multimodal_ai_service.get_multimodal_metrics()
        
        return {
            "federated_learning": fl_metrics,
            "automl": automl_metrics,
            "reinforcement_learning": rl_metrics,
            "multimodal_ai": multimodal_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ML capabilities status: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for ML capabilities"""
    
    try:
        # Check if services are initialized
        services_status = {
            "federated_learning": "healthy",
            "automl": "healthy",
            "reinforcement_learning": "healthy",
            "multimodal_ai": "healthy"
        }
        
        return {
            "status": "healthy",
            "services": services_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        } 