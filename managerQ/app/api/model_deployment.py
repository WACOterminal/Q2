"""
Model Deployment API

This API provides endpoints for managing model deployments, including:
- Creating and managing deployments
- Monitoring deployment health and metrics
- Traffic management and routing
- Rollback and recovery operations
- A/B testing and canary deployments
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from ..core.model_deployment_service import (
    ModelDeploymentService,
    ModelDeploymentSpec,
    DeploymentStrategy,
    ServiceStatus,
    TrafficSplit
)
from ..core.auth_service import auth_service
from ..core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Initialize components
router = APIRouter(prefix="/api/v1/deployment", tags=["Model Deployment"])
security = HTTPBearer()
rate_limiter = RateLimiter()

# Global service instance
model_deployment_service: Optional[ModelDeploymentService] = None

# ===== REQUEST/RESPONSE MODELS =====

class DeploymentRequest(BaseModel):
    """Request model for creating a deployment"""
    model_id: str = Field(..., description="Model ID to deploy")
    model_version: str = Field(..., description="Model version")
    deployment_name: str = Field(..., description="Deployment name")
    strategy: DeploymentStrategy = Field(default=DeploymentStrategy.BLUE_GREEN, description="Deployment strategy")
    container_image: str = Field(..., description="Container image")
    container_tag: str = Field(default="latest", description="Container tag")
    resources: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    config: Dict[str, Any] = Field(default_factory=dict, description="Deployment configuration")
    health_check: Dict[str, Any] = Field(default_factory=dict, description="Health check configuration")
    scaling: Dict[str, Any] = Field(default_factory=dict, description="Scaling configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    approval_required: bool = Field(default=False, description="Require approval before deployment")
    pipeline_id: Optional[str] = Field(None, description="Pipeline ID to use")

class DeploymentResponse(BaseModel):
    """Response model for deployment operations"""
    deployment_id: str
    status: str
    message: str
    created_at: datetime

class DeploymentStatusResponse(BaseModel):
    """Response model for deployment status"""
    deployment_id: str
    status: str
    strategy: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    current_traffic_split: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

class ServiceStatusResponse(BaseModel):
    """Response model for service status"""
    service_name: str
    endpoint_url: str
    version: str
    status: str
    health_score: float
    last_health_check: datetime
    metrics: Dict[str, Any] = Field(default_factory=dict)
    traffic_percentage: float = 0.0

class TrafficSplitRequest(BaseModel):
    """Request model for traffic splitting"""
    traffic_config: Dict[str, float] = Field(..., description="Traffic distribution configuration")

class RollbackRequest(BaseModel):
    """Request model for rollback operations"""
    target_version: Optional[str] = Field(None, description="Target version to rollback to")
    reason: str = Field(..., description="Reason for rollback")

class MetricsResponse(BaseModel):
    """Response model for deployment metrics"""
    total_deployments: int
    successful_deployments: int
    failed_deployments: int
    rollbacks: int
    average_deployment_time: float
    active_services: int
    total_requests_served: int
    average_response_time: float

# ===== DEPENDENCY INJECTION =====

def get_model_deployment_service() -> ModelDeploymentService:
    """Get the model deployment service instance"""
    global model_deployment_service
    if model_deployment_service is None:
        raise HTTPException(status_code=503, detail="Model deployment service not available")
    return model_deployment_service

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    return auth_service.verify_token(credentials.credentials)

# ===== DEPLOYMENT MANAGEMENT =====

@router.post("/deploy", response_model=DeploymentResponse)
async def deploy_model(
    request: DeploymentRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user),
    service: ModelDeploymentService = Depends(get_model_deployment_service)
):
    """
    Deploy a model using the specified strategy
    
    This endpoint creates a new model deployment with the specified configuration.
    Supports Blue/Green, Canary, Rolling, and A/B testing strategies.
    """
    try:
        # Rate limiting
        await rate_limiter.check_rate_limit(f"deployment:{user['user_id']}", max_requests=5, window_seconds=300)
        
        # Create deployment specification
        spec = ModelDeploymentSpec(
            model_id=request.model_id,
            model_version=request.model_version,
            deployment_name=request.deployment_name,
            strategy=request.strategy,
            container_image=request.container_image,
            container_tag=request.container_tag,
            resources=request.resources,
            environment=request.environment,
            config=request.config,
            health_check=request.health_check,
            scaling=request.scaling,
            metadata={
                **request.metadata,
                "created_by": user["user_id"],
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        # Start deployment
        deployment_id = await service.deploy_model(
            spec=spec,
            pipeline_id=request.pipeline_id,
            approval_required=request.approval_required
        )
        
        logger.info(f"Deployment created: {deployment_id} by user {user['user_id']}")
        
        return DeploymentResponse(
            deployment_id=deployment_id,
            status="created",
            message="Deployment created successfully",
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error creating deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deployments", response_model=List[Dict[str, Any]])
async def list_deployments(
    limit: int = Query(50, description="Number of deployments to return"),
    user=Depends(get_current_user),
    service: ModelDeploymentService = Depends(get_model_deployment_service)
):
    """
    List recent deployments
    
    Returns a paginated list of deployments with their current status.
    """
    try:
        await rate_limiter.check_rate_limit(f"list_deployments:{user['user_id']}", max_requests=10, window_seconds=60)
        
        deployments = await service.list_deployments(limit=limit)
        
        return deployments
        
    except Exception as e:
        logger.error(f"Error listing deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deployments/{deployment_id}", response_model=DeploymentStatusResponse)
async def get_deployment_status(
    deployment_id: str,
    user=Depends(get_current_user),
    service: ModelDeploymentService = Depends(get_model_deployment_service)
):
    """
    Get deployment status
    
    Returns detailed status information for a specific deployment.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_deployment:{user['user_id']}", max_requests=20, window_seconds=60)
        
        deployment = await service.get_deployment_status(deployment_id)
        
        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")
        
        return DeploymentStatusResponse(**deployment)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deployment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deployments/{deployment_id}/rollback", response_model=DeploymentResponse)
async def rollback_deployment(
    deployment_id: str,
    request: RollbackRequest,
    user=Depends(get_current_user),
    service: ModelDeploymentService = Depends(get_model_deployment_service)
):
    """
    Rollback a deployment
    
    Rolls back a deployment to a previous version or stable state.
    """
    try:
        await rate_limiter.check_rate_limit(f"rollback:{user['user_id']}", max_requests=3, window_seconds=300)
        
        success = await service.rollback_deployment(
            deployment_id=deployment_id,
            target_version=request.target_version
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Rollback failed")
        
        logger.info(f"Deployment rolled back: {deployment_id} by user {user['user_id']}, reason: {request.reason}")
        
        return DeploymentResponse(
            deployment_id=deployment_id,
            status="rolled_back",
            message="Deployment rolled back successfully",
            created_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== SERVICE MANAGEMENT =====

@router.get("/services", response_model=List[ServiceStatusResponse])
async def list_services(
    user=Depends(get_current_user),
    service: ModelDeploymentService = Depends(get_model_deployment_service)
):
    """
    List active services
    
    Returns a list of all active services and their health status.
    """
    try:
        await rate_limiter.check_rate_limit(f"list_services:{user['user_id']}", max_requests=10, window_seconds=60)
        
        services = []
        for service_name, endpoint in service.active_services.items():
            services.append(ServiceStatusResponse(
                service_name=service_name,
                endpoint_url=endpoint.endpoint_url,
                version=endpoint.version,
                status=endpoint.status.value,
                health_score=endpoint.health_score,
                last_health_check=endpoint.last_health_check,
                metrics=endpoint.metrics,
                traffic_percentage=endpoint.traffic_percentage
            ))
        
        return services
        
    except Exception as e:
        logger.error(f"Error listing services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/services/{service_name}", response_model=ServiceStatusResponse)
async def get_service_status(
    service_name: str,
    user=Depends(get_current_user),
    service: ModelDeploymentService = Depends(get_model_deployment_service)
):
    """
    Get service status
    
    Returns detailed status information for a specific service.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_service:{user['user_id']}", max_requests=20, window_seconds=60)
        
        service_status = await service.get_service_status(service_name)
        
        if not service_status:
            raise HTTPException(status_code=404, detail="Service not found")
        
        return ServiceStatusResponse(**service_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== TRAFFIC MANAGEMENT =====

@router.post("/deployments/{deployment_name}/traffic", response_model=Dict[str, Any])
async def update_traffic_split(
    deployment_name: str,
    request: TrafficSplitRequest,
    user=Depends(get_current_user),
    service: ModelDeploymentService = Depends(get_model_deployment_service)
):
    """
    Update traffic split for a deployment
    
    Adjusts traffic distribution between different versions of a service.
    """
    try:
        await rate_limiter.check_rate_limit(f"traffic:{user['user_id']}", max_requests=5, window_seconds=300)
        
        # Validate traffic percentages sum to 100
        total_traffic = sum(request.traffic_config.values())
        if abs(total_traffic - 100.0) > 0.01:
            raise HTTPException(status_code=400, detail="Traffic percentages must sum to 100")
        
        await service.update_traffic_split(deployment_name, request.traffic_config)
        
        logger.info(f"Traffic split updated for {deployment_name} by user {user['user_id']}")
        
        return {
            "deployment_name": deployment_name,
            "traffic_config": request.traffic_config,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating traffic split: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deployments/{deployment_name}/traffic", response_model=Dict[str, Any])
async def get_traffic_split(
    deployment_name: str,
    user=Depends(get_current_user),
    service: ModelDeploymentService = Depends(get_model_deployment_service)
):
    """
    Get current traffic split for a deployment
    
    Returns the current traffic distribution configuration.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_traffic:{user['user_id']}", max_requests=20, window_seconds=60)
        
        traffic_config = service.traffic_routes.get(deployment_name, {})
        
        return {
            "deployment_name": deployment_name,
            "traffic_config": traffic_config,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting traffic split: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== METRICS AND MONITORING =====

@router.get("/metrics", response_model=MetricsResponse)
async def get_deployment_metrics(
    user=Depends(get_current_user),
    service: ModelDeploymentService = Depends(get_model_deployment_service)
):
    """
    Get deployment metrics
    
    Returns overall deployment statistics and performance metrics.
    """
    try:
        await rate_limiter.check_rate_limit(f"metrics:{user['user_id']}", max_requests=10, window_seconds=60)
        
        metrics = await service.get_deployment_metrics()
        
        return MetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting deployment metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint
    
    Returns the health status of the deployment service.
    """
    try:
        global model_deployment_service
        
        if model_deployment_service is None:
            return {"status": "unhealthy", "message": "Service not initialized"}
        
        # Check service health
        active_services = len(model_deployment_service.active_services)
        healthy_services = sum(
            1 for service in model_deployment_service.active_services.values()
            if service.status == ServiceStatus.HEALTHY
        )
        
        health_score = healthy_services / active_services if active_services > 0 else 1.0
        
        return {
            "status": "healthy" if health_score >= 0.8 else "degraded",
            "health_score": health_score,
            "active_services": active_services,
            "healthy_services": healthy_services,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {"status": "unhealthy", "message": str(e)}

# ===== PIPELINE MANAGEMENT =====

@router.get("/pipelines", response_model=List[Dict[str, Any]])
async def list_pipelines(
    user=Depends(get_current_user),
    service: ModelDeploymentService = Depends(get_model_deployment_service)
):
    """
    List available deployment pipelines
    
    Returns a list of configured deployment pipelines.
    """
    try:
        await rate_limiter.check_rate_limit(f"list_pipelines:{user['user_id']}", max_requests=10, window_seconds=60)
        
        pipelines = []
        for pipeline_id, pipeline in service.deployment_pipelines.items():
            pipelines.append({
                "pipeline_id": pipeline.pipeline_id,
                "name": pipeline.name,
                "stages": pipeline.stages,
                "approval_required": pipeline.approval_required,
                "automated_rollback": pipeline.automated_rollback,
                "created_at": pipeline.created_at.isoformat()
            })
        
        return pipelines
        
    except Exception as e:
        logger.error(f"Error listing pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== INITIALIZATION =====

async def initialize_deployment_api():
    """Initialize the deployment API with service instance"""
    global model_deployment_service
    
    try:
        model_deployment_service = ModelDeploymentService()
        await model_deployment_service.initialize()
        
        logger.info("Model deployment API initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing deployment API: {e}")
        raise

async def shutdown_deployment_api():
    """Shutdown the deployment API"""
    global model_deployment_service
    
    try:
        if model_deployment_service:
            await model_deployment_service.shutdown()
            model_deployment_service = None
        
        logger.info("Model deployment API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error shutting down deployment API: {e}") 