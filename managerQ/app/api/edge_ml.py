"""
Edge ML API

This module provides FastAPI endpoints for the edge ML service:
- Edge device registration and management
- Model deployment and versioning
- Inference request handling
- Performance monitoring and analytics
- Device and model synchronization
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import logging
import tempfile
from pathlib import Path

from ..core.edge_ml_service import (
    EdgeMLService,
    EdgeDevice,
    EdgeModel,
    ModelDeployment,
    InferenceRequest,
    InferenceResult,
    DeviceMetrics,
    EdgeDeviceType,
    DeviceStatus,
    ModelFormat,
    DeploymentStatus,
    InferenceMode,
    OptimizationLevel,
    edge_ml_service
)
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

# Create the router
router = APIRouter(prefix="/api/v1/edge-ml", tags=["edge-ml"])

# ===== PYDANTIC MODELS =====

class EdgeDeviceModel(BaseModel):
    """Edge device model for API"""
    device_id: str
    name: str
    device_type: EdgeDeviceType
    status: DeviceStatus
    ip_address: str
    port: int
    capabilities: Dict[str, Any]
    resources: Dict[str, Any]
    last_seen: datetime
    registered_at: datetime
    metadata: Optional[Dict[str, Any]] = None

class DeviceRegistrationModel(BaseModel):
    """Device registration model"""
    device_id: str
    name: str
    device_type: EdgeDeviceType
    ip_address: str
    port: int = 8080
    capabilities: Dict[str, Any]
    resources: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class EdgeModelModel(BaseModel):
    """Edge model model for API"""
    model_id: str
    name: str
    version: str
    format: ModelFormat
    size_bytes: int
    checksum: str
    input_shape: List[int]
    output_shape: List[int]
    inference_time_ms: float
    memory_mb: float
    optimization_level: OptimizationLevel
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

class ModelRegistrationModel(BaseModel):
    """Model registration model"""
    model_id: str
    name: str
    version: str
    format: ModelFormat
    input_shape: List[int]
    output_shape: List[int]
    inference_time_ms: float = 0.0
    memory_mb: float = 0.0
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    metadata: Optional[Dict[str, Any]] = None

class ModelDeploymentModel(BaseModel):
    """Model deployment model for API"""
    deployment_id: str
    device_id: str
    model_id: str
    status: DeploymentStatus
    deployment_config: Dict[str, Any]
    deployed_at: Optional[datetime] = None
    last_update: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class DeploymentRequestModel(BaseModel):
    """Deployment request model"""
    device_id: str
    model_id: str
    deployment_config: Optional[Dict[str, Any]] = None

class InferenceRequestModel(BaseModel):
    """Inference request model for API"""
    request_id: str
    device_id: str
    model_id: str
    input_data: Any
    inference_mode: InferenceMode = InferenceMode.REAL_TIME
    priority: int = Field(ge=1, le=10, default=5)
    timeout: int = Field(ge=1, le=300, default=30)
    metadata: Optional[Dict[str, Any]] = None

class InferenceResultModel(BaseModel):
    """Inference result model for API"""
    request_id: str
    device_id: str
    model_id: str
    result: Any
    inference_time_ms: float
    confidence: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime

class DeviceMetricsModel(BaseModel):
    """Device metrics model for API"""
    device_id: str
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    temperature: Optional[float] = None
    battery_level: Optional[float] = None
    inference_count: int = 0
    error_count: int = 0

class DeviceUpdateModel(BaseModel):
    """Device update model"""
    status: Optional[DeviceStatus] = None
    capabilities: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

# ===== DEVICE MANAGEMENT ENDPOINTS =====

@router.post("/devices/register")
async def register_device(
    device: DeviceRegistrationModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Register a new edge device"""
    try:
        # Create device object
        edge_device = EdgeDevice(
            device_id=device.device_id,
            name=device.name,
            device_type=device.device_type,
            status=DeviceStatus.ONLINE,
            ip_address=device.ip_address,
            port=device.port,
            capabilities=device.capabilities,
            resources=device.resources,
            last_seen=datetime.utcnow(),
            registered_at=datetime.utcnow(),
            metadata=device.metadata or {}
        )
        
        success = await edge_ml_service.register_device(edge_device)
        
        if success:
            logger.info(f"Device {device.device_id} registered by {user.get('username', 'unknown')}")
            return JSONResponse(
                content={"message": "Device registered successfully", "device_id": device.device_id},
                status_code=201
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to register device")
            
    except Exception as e:
        logger.error(f"Error registering device: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/devices")
async def get_devices(
    device_type: Optional[EdgeDeviceType] = None,
    status: Optional[DeviceStatus] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all registered devices with optional filtering"""
    try:
        devices = await edge_ml_service.get_device_list()
        
        # Apply filters
        if device_type:
            devices = [d for d in devices if d.get('device_type') == device_type.value]
        
        if status:
            devices = [d for d in devices if d.get('status') == status.value]
        
        return JSONResponse(content={"devices": devices})
        
    except Exception as e:
        logger.error(f"Error getting devices: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/devices/{device_id}")
async def get_device(
    device_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get device details"""
    try:
        if device_id not in edge_ml_service.devices:
            raise HTTPException(status_code=404, detail="Device not found")
        
        device = edge_ml_service.devices[device_id]
        
        device_model = EdgeDeviceModel(
            device_id=device.device_id,
            name=device.name,
            device_type=device.device_type,
            status=device.status,
            ip_address=device.ip_address,
            port=device.port,
            capabilities=device.capabilities,
            resources=device.resources,
            last_seen=device.last_seen,
            registered_at=device.registered_at,
            metadata=device.metadata
        )
        
        return JSONResponse(content=device_model.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting device: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.put("/devices/{device_id}")
async def update_device(
    device_id: str,
    update: DeviceUpdateModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Update device information"""
    try:
        if device_id not in edge_ml_service.devices:
            raise HTTPException(status_code=404, detail="Device not found")
        
        device = edge_ml_service.devices[device_id]
        
        # Update fields
        if update.status:
            await edge_ml_service.update_device_status(device_id, update.status)
        
        if update.capabilities:
            device.capabilities.update(update.capabilities)
        
        if update.resources:
            device.resources.update(update.resources)
        
        if update.metadata:
            device.metadata.update(update.metadata)
        
        logger.info(f"Device {device_id} updated by {user.get('username', 'unknown')}")
        
        return JSONResponse(content={"message": "Device updated successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating device: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/devices/{device_id}")
async def unregister_device(
    device_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Unregister an edge device"""
    try:
        success = await edge_ml_service.unregister_device(device_id)
        
        if success:
            logger.info(f"Device {device_id} unregistered by {user.get('username', 'unknown')}")
            return JSONResponse(content={"message": "Device unregistered successfully"})
        else:
            raise HTTPException(status_code=404, detail="Device not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering device: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/devices/{device_id}/metrics")
async def get_device_metrics(
    device_id: str,
    limit: int = Query(100, ge=1, le=1000),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get device performance metrics"""
    try:
        metrics = await edge_ml_service.get_device_metrics(device_id)
        
        # Apply limit
        if limit:
            metrics = metrics[-limit:]
        
        # Convert to API models
        result = []
        for metric in metrics:
            result.append(DeviceMetricsModel(
                device_id=metric.device_id,
                timestamp=metric.timestamp,
                cpu_usage=metric.cpu_usage,
                memory_usage=metric.memory_usage,
                disk_usage=metric.disk_usage,
                network_latency=metric.network_latency,
                temperature=metric.temperature,
                battery_level=metric.battery_level,
                inference_count=metric.inference_count,
                error_count=metric.error_count
            ))
        
        return JSONResponse(content={"metrics": [m.dict() for m in result]})
        
    except Exception as e:
        logger.error(f"Error getting device metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== MODEL MANAGEMENT ENDPOINTS =====

@router.post("/models/register")
async def register_model(
    model: ModelRegistrationModel,
    model_file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Register a new edge model"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{model.format.value}") as tmp_file:
            content = await model_file.read()
            tmp_file.write(content)
            tmp_file_path = Path(tmp_file.name)
        
        # Create model object
        edge_model = EdgeModel(
            model_id=model.model_id,
            name=model.name,
            version=model.version,
            format=model.format,
            size_bytes=len(content),
            checksum="",  # Will be calculated in service
            input_shape=model.input_shape,
            output_shape=model.output_shape,
            inference_time_ms=model.inference_time_ms,
            memory_mb=model.memory_mb,
            optimization_level=model.optimization_level,
            created_at=datetime.utcnow(),
            metadata=model.metadata or {}
        )
        
        success = await edge_ml_service.register_model(edge_model, tmp_file_path)
        
        # Clean up temporary file
        tmp_file_path.unlink()
        
        if success:
            logger.info(f"Model {model.model_id} registered by {user.get('username', 'unknown')}")
            return JSONResponse(
                content={"message": "Model registered successfully", "model_id": model.model_id},
                status_code=201
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to register model")
            
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/models")
async def get_models(
    format: Optional[ModelFormat] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all registered models with optional filtering"""
    try:
        models = await edge_ml_service.get_model_list()
        
        # Apply filters
        if format:
            models = [m for m in models if m.get('format') == format.value]
        
        return JSONResponse(content={"models": models})
        
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
        if model_id not in edge_ml_service.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = edge_ml_service.models[model_id]
        
        model_dict = EdgeModelModel(
            model_id=model.model_id,
            name=model.name,
            version=model.version,
            format=model.format,
            size_bytes=model.size_bytes,
            checksum=model.checksum,
            input_shape=model.input_shape,
            output_shape=model.output_shape,
            inference_time_ms=model.inference_time_ms,
            memory_mb=model.memory_mb,
            optimization_level=model.optimization_level,
            created_at=model.created_at,
            metadata=model.metadata
        )
        
        return JSONResponse(content=model_dict.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/models/{model_id}")
async def remove_model(
    model_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Remove a model"""
    try:
        success = await edge_ml_service.remove_model(model_id)
        
        if success:
            logger.info(f"Model {model_id} removed by {user.get('username', 'unknown')}")
            return JSONResponse(content={"message": "Model removed successfully"})
        else:
            raise HTTPException(status_code=404, detail="Model not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== DEPLOYMENT MANAGEMENT ENDPOINTS =====

@router.post("/deployments")
async def deploy_model(
    deployment: DeploymentRequestModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Deploy a model to an edge device"""
    try:
        deployment_id = await edge_ml_service.deploy_model(
            deployment.device_id,
            deployment.model_id,
            deployment.deployment_config
        )
        
        logger.info(f"Model deployment {deployment_id} created by {user.get('username', 'unknown')}")
        
        return JSONResponse(
            content={"message": "Model deployment initiated", "deployment_id": deployment_id},
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/deployments")
async def get_deployments(
    device_id: Optional[str] = None,
    model_id: Optional[str] = None,
    status: Optional[DeploymentStatus] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get deployments with optional filtering"""
    try:
        deployments = await edge_ml_service.list_deployments(device_id, model_id)
        
        # Apply status filter
        if status:
            deployments = [d for d in deployments if d.status == status]
        
        # Convert to API models
        result = []
        for deployment in deployments:
            result.append(ModelDeploymentModel(
                deployment_id=deployment.deployment_id,
                device_id=deployment.device_id,
                model_id=deployment.model_id,
                status=deployment.status,
                deployment_config=deployment.deployment_config,
                deployed_at=deployment.deployed_at,
                last_update=deployment.last_update,
                performance_metrics=deployment.performance_metrics,
                error_message=deployment.error_message
            ))
        
        return JSONResponse(content={"deployments": [d.dict() for d in result]})
        
    except Exception as e:
        logger.error(f"Error getting deployments: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/deployments/{deployment_id}")
async def get_deployment(
    deployment_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get deployment details"""
    try:
        deployment = await edge_ml_service.get_deployment_status(deployment_id)
        
        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")
        
        deployment_model = ModelDeploymentModel(
            deployment_id=deployment.deployment_id,
            device_id=deployment.device_id,
            model_id=deployment.model_id,
            status=deployment.status,
            deployment_config=deployment.deployment_config,
            deployed_at=deployment.deployed_at,
            last_update=deployment.last_update,
            performance_metrics=deployment.performance_metrics,
            error_message=deployment.error_message
        )
        
        return JSONResponse(content=deployment_model.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deployment: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/deployments/{deployment_id}")
async def undeploy_model(
    deployment_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Undeploy a model from an edge device"""
    try:
        success = await edge_ml_service.undeploy_model(deployment_id)
        
        if success:
            logger.info(f"Model undeployed {deployment_id} by {user.get('username', 'unknown')}")
            return JSONResponse(content={"message": "Model undeployed successfully"})
        else:
            raise HTTPException(status_code=404, detail="Deployment not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error undeploying model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== INFERENCE ENDPOINTS =====

@router.post("/inference")
async def submit_inference(
    request: InferenceRequestModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Submit an inference request"""
    try:
        # Create inference request
        inference_request = InferenceRequest(
            request_id=request.request_id,
            device_id=request.device_id,
            model_id=request.model_id,
            input_data=request.input_data,
            inference_mode=request.inference_mode,
            priority=request.priority,
            timeout=request.timeout,
            created_at=datetime.utcnow(),
            metadata=request.metadata
        )
        
        request_id = await edge_ml_service.submit_inference_request(inference_request)
        
        logger.info(f"Inference request {request_id} submitted by {user.get('username', 'unknown')}")
        
        return JSONResponse(
            content={"message": "Inference request submitted", "request_id": request_id},
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error submitting inference: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/inference/{request_id}")
async def get_inference_result(
    request_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get inference result"""
    try:
        result = await edge_ml_service.get_inference_result(request_id)
        
        if not result:
            # Check if request exists
            if request_id not in edge_ml_service.inference_requests:
                raise HTTPException(status_code=404, detail="Inference request not found")
            
            # Request is still pending
            return JSONResponse(content={"status": "pending", "message": "Inference still processing"})
        
        result_model = InferenceResultModel(
            request_id=result.request_id,
            device_id=result.device_id,
            model_id=result.model_id,
            result=result.result,
            inference_time_ms=result.inference_time_ms,
            confidence=result.confidence,
            error_message=result.error_message,
            created_at=result.created_at
        )
        
        return JSONResponse(content=result_model.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting inference result: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/inference/{request_id}")
async def cancel_inference(
    request_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Cancel an inference request"""
    try:
        success = await edge_ml_service.cancel_inference_request(request_id)
        
        if success:
            logger.info(f"Inference request {request_id} cancelled by {user.get('username', 'unknown')}")
            return JSONResponse(content={"message": "Inference request cancelled"})
        else:
            raise HTTPException(status_code=404, detail="Inference request not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling inference: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== MONITORING ENDPOINTS =====

@router.get("/system/status")
async def get_system_status(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get system status"""
    try:
        status = await edge_ml_service.get_system_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/analytics/dashboard")
async def get_analytics_dashboard(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get analytics dashboard data"""
    try:
        status = await edge_ml_service.get_system_status()
        
        # Calculate additional analytics
        total_inference_time = sum(
            result.inference_time_ms for result in edge_ml_service.inference_results.values()
        )
        
        dashboard_data = {
            "system_status": status,
            "performance_summary": {
                "total_inference_time": total_inference_time,
                "average_inference_time": total_inference_time / max(len(edge_ml_service.inference_results), 1),
                "success_rate": len([r for r in edge_ml_service.inference_results.values() if r.error_message is None]) / max(len(edge_ml_service.inference_results), 1) * 100,
                "deployment_success_rate": len([d for d in edge_ml_service.deployments.values() if d.status == DeploymentStatus.DEPLOYED]) / max(len(edge_ml_service.deployments), 1) * 100
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting analytics dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/analytics/trends")
async def get_analytics_trends(
    device_id: Optional[str] = None,
    model_id: Optional[str] = None,
    time_range: int = Query(24, description="Time range in hours"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get analytics trends"""
    try:
        # Get metrics from the last time_range hours
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_range)
        
        trends = {
            "device_metrics": {},
            "inference_trends": {},
            "deployment_trends": {}
        }
        
        # Device metrics trends
        for device_id_key, metrics in edge_ml_service.device_metrics.items():
            if device_id and device_id_key != device_id:
                continue
            
            filtered_metrics = [m for m in metrics if m.timestamp >= start_time]
            
            if filtered_metrics:
                trends["device_metrics"][device_id_key] = {
                    "cpu_usage": [m.cpu_usage for m in filtered_metrics],
                    "memory_usage": [m.memory_usage for m in filtered_metrics],
                    "inference_count": [m.inference_count for m in filtered_metrics],
                    "timestamps": [m.timestamp.isoformat() for m in filtered_metrics]
                }
        
        # Inference trends
        inference_results = [r for r in edge_ml_service.inference_results.values() if r.created_at >= start_time]
        
        if model_id:
            inference_results = [r for r in inference_results if r.model_id == model_id]
        
        if inference_results:
            trends["inference_trends"] = {
                "total_requests": len(inference_results),
                "success_rate": len([r for r in inference_results if r.error_message is None]) / len(inference_results) * 100,
                "average_inference_time": sum(r.inference_time_ms for r in inference_results) / len(inference_results),
                "inference_times": [r.inference_time_ms for r in inference_results],
                "timestamps": [r.created_at.isoformat() for r in inference_results]
            }
        
        return JSONResponse(content=trends)
        
    except Exception as e:
        logger.error(f"Error getting analytics trends: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== SYSTEM MANAGEMENT ENDPOINTS =====

@router.post("/system/initialize")
async def initialize_system(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Initialize the edge ML system"""
    try:
        await edge_ml_service.initialize()
        
        logger.info(f"Edge ML system initialized by {user.get('username', 'unknown')}")
        
        return JSONResponse(content={"message": "System initialized successfully"})
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/system/config")
async def get_system_config(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get system configuration"""
    try:
        config = edge_ml_service.config
        return JSONResponse(content={"config": config})
        
    except Exception as e:
        logger.error(f"Error getting system config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.put("/system/config")
async def update_system_config(
    config: Dict[str, Any],
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Update system configuration"""
    try:
        # Validate and update configuration
        valid_keys = set(edge_ml_service.config.keys())
        for key, value in config.items():
            if key in valid_keys:
                edge_ml_service.config[key] = value
        
        logger.info(f"System configuration updated by {user.get('username', 'unknown')}")
        
        return JSONResponse(content={"message": "Configuration updated successfully"})
        
    except Exception as e:
        logger.error(f"Error updating system config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== HELPER ENDPOINTS =====

@router.get("/compatibility/{device_id}/{model_id}")
async def check_compatibility(
    device_id: str,
    model_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Check device-model compatibility"""
    try:
        if device_id not in edge_ml_service.devices:
            raise HTTPException(status_code=404, detail="Device not found")
        
        if model_id not in edge_ml_service.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        device = edge_ml_service.devices[device_id]
        model = edge_ml_service.models[model_id]
        
        # Check compatibility
        compatible = await edge_ml_service._check_device_compatibility(device, model)
        
        compatibility_info = {
            "compatible": compatible,
            "device_id": device_id,
            "model_id": model_id,
            "checks": {
                "memory_sufficient": device.resources.get("memory_mb", 0) >= model.memory_mb,
                "format_supported": model.format.value in device.capabilities.get("model_formats", []),
                "gpu_available": device.capabilities.get("has_gpu", False) if model.metadata.get("requires_gpu", False) else True
            }
        }
        
        return JSONResponse(content=compatibility_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking compatibility: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={"status": "healthy", "service": "edge-ml"})

# ===== INITIALIZATION FUNCTIONS =====

async def initialize_edge_ml_api():
    """Initialize the edge ML API"""
    try:
        await edge_ml_service.initialize()
        logger.info("Edge ML API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Edge ML API: {e}")
        raise

async def shutdown_edge_ml_api():
    """Shutdown the edge ML API"""
    try:
        await edge_ml_service.shutdown()
        logger.info("Edge ML API shutdown successfully")
    except Exception as e:
        logger.error(f"Error shutting down Edge ML API: {e}") 