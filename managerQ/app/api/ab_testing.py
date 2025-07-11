"""
A/B Testing API

This API provides endpoints for managing A/B testing experiments:
- Experiment creation and management
- Traffic allocation and user assignment
- Result tracking and metrics collection
- Statistical analysis and reporting
- Real-time monitoring and alerts
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from ..core.ab_testing_service import (
    ABTestingService,
    ab_testing_service,
    ExperimentConfig,
    ExperimentVariant,
    ExperimentMetric,
    AudienceSegment,
    ExperimentStatus,
    VariantType,
    MetricType,
    StatisticalTest,
    TrafficAllocation
)
from ..core.auth_service import auth_service
from ..core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Initialize components
router = APIRouter(prefix="/api/v1/ab-testing", tags=["A/B Testing"])
security = HTTPBearer()
rate_limiter = RateLimiter()

# ===== REQUEST/RESPONSE MODELS =====

class ExperimentVariantRequest(BaseModel):
    """Request model for experiment variant"""
    variant_id: str = Field(..., description="Unique variant identifier")
    name: str = Field(..., description="Variant name")
    description: str = Field(..., description="Variant description")
    variant_type: VariantType = Field(..., description="Variant type")
    traffic_allocation: float = Field(..., description="Traffic allocation percentage")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Variant configuration")
    model_id: Optional[str] = Field(None, description="Model ID for this variant")
    model_version: Optional[str] = Field(None, description="Model version for this variant")

class ExperimentMetricRequest(BaseModel):
    """Request model for experiment metric"""
    metric_id: str = Field(..., description="Unique metric identifier")
    name: str = Field(..., description="Metric name")
    description: str = Field(..., description="Metric description")
    metric_type: MetricType = Field(..., description="Metric type")
    primary: bool = Field(default=False, description="Whether this is the primary metric")
    higher_is_better: bool = Field(default=True, description="Whether higher values are better")
    minimum_detectable_effect: float = Field(default=0.05, description="Minimum detectable effect size")
    statistical_test: StatisticalTest = Field(default=StatisticalTest.T_TEST, description="Statistical test to use")
    confidence_level: float = Field(default=0.95, description="Confidence level for analysis")

class AudienceSegmentRequest(BaseModel):
    """Request model for audience segment"""
    segment_id: str = Field(..., description="Unique segment identifier")
    name: str = Field(..., description="Segment name")
    description: str = Field(..., description="Segment description")
    criteria: Dict[str, Any] = Field(..., description="Segmentation criteria")
    size_estimate: Optional[int] = Field(None, description="Estimated segment size")

class ExperimentRequest(BaseModel):
    """Request model for creating an experiment"""
    experiment_id: Optional[str] = Field(None, description="Unique experiment identifier")
    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    hypothesis: str = Field(..., description="Experiment hypothesis")
    objective: str = Field(..., description="Experiment objective")
    variants: List[ExperimentVariantRequest] = Field(..., description="Experiment variants")
    metrics: List[ExperimentMetricRequest] = Field(..., description="Experiment metrics")
    audience_segments: List[AudienceSegmentRequest] = Field(default_factory=list, description="Audience segments")
    traffic_allocation: TrafficAllocation = Field(default=TrafficAllocation.EQUAL, description="Traffic allocation strategy")
    start_date: datetime = Field(..., description="Experiment start date")
    end_date: datetime = Field(..., description="Experiment end date")
    sample_size: int = Field(default=0, description="Required sample size (0 for auto-calculation)")
    confidence_level: float = Field(default=0.95, description="Statistical confidence level")
    statistical_power: float = Field(default=0.8, description="Statistical power")
    early_stopping_enabled: bool = Field(default=True, description="Enable early stopping")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ExperimentResponse(BaseModel):
    """Response model for experiment operations"""
    experiment_id: str
    status: str
    message: str
    created_at: datetime

class VariantAssignmentRequest(BaseModel):
    """Request model for variant assignment"""
    user_id: str = Field(..., description="User ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="Assignment context")

class VariantAssignmentResponse(BaseModel):
    """Response model for variant assignment"""
    experiment_id: str
    user_id: str
    variant_id: Optional[str]
    assigned_at: datetime

class EventTrackingRequest(BaseModel):
    """Request model for event tracking"""
    user_id: str = Field(..., description="User ID")
    metric_id: str = Field(..., description="Metric ID")
    value: float = Field(..., description="Metric value")
    context: Dict[str, Any] = Field(default_factory=dict, description="Event context")

class EventTrackingResponse(BaseModel):
    """Response model for event tracking"""
    success: bool
    message: str
    tracked_at: datetime

class ExperimentSummaryResponse(BaseModel):
    """Response model for experiment summary"""
    experiment_id: str
    status: str
    name: str
    description: str
    start_date: datetime
    end_date: Optional[datetime]
    total_participants: int
    variant_performance: Dict[str, Dict[str, float]]
    statistical_significance: bool
    winning_variant: Optional[str]
    confidence_score: float
    recommendations: List[str]

class StatisticalAnalysisResponse(BaseModel):
    """Response model for statistical analysis"""
    metric_id: str
    statistical_significance: bool
    p_value: float
    confidence_interval: List[float]
    effect_size: float
    power: float
    sample_size: int
    variant_comparisons: List[Dict[str, Any]]

class ExperimentListResponse(BaseModel):
    """Response model for experiment listing"""
    experiments: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int

# ===== DEPENDENCY INJECTION =====

def get_ab_testing_service() -> ABTestingService:
    """Get the A/B testing service instance"""
    return ab_testing_service

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    return auth_service.verify_token(credentials.credentials)

# ===== EXPERIMENT MANAGEMENT =====

@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentRequest,
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    Create a new A/B test experiment
    
    Creates a new experiment with the specified configuration including variants,
    metrics, and audience segments. The experiment will be in DRAFT status initially.
    """
    try:
        # Rate limiting
        await rate_limiter.check_rate_limit(f"create_experiment:{user['user_id']}", max_requests=10, window_seconds=3600)
        
        # Convert request to experiment config
        variants = [
            ExperimentVariant(
                variant_id=v.variant_id,
                name=v.name,
                description=v.description,
                variant_type=v.variant_type,
                traffic_allocation=v.traffic_allocation,
                configuration=v.configuration,
                model_id=v.model_id,
                model_version=v.model_version
            ) for v in request.variants
        ]
        
        metrics = [
            ExperimentMetric(
                metric_id=m.metric_id,
                name=m.name,
                description=m.description,
                metric_type=m.metric_type,
                primary=m.primary,
                higher_is_better=m.higher_is_better,
                minimum_detectable_effect=m.minimum_detectable_effect,
                statistical_test=m.statistical_test,
                confidence_level=m.confidence_level
            ) for m in request.metrics
        ]
        
        audience_segments = [
            AudienceSegment(
                segment_id=s.segment_id,
                name=s.name,
                description=s.description,
                criteria=s.criteria,
                size_estimate=s.size_estimate
            ) for s in request.audience_segments
        ]
        
        config = ExperimentConfig(
            experiment_id=request.experiment_id or "",
            name=request.name,
            description=request.description,
            hypothesis=request.hypothesis,
            objective=request.objective,
            variants=variants,
            metrics=metrics,
            audience_segments=audience_segments,
            traffic_allocation=request.traffic_allocation,
            start_date=request.start_date,
            end_date=request.end_date,
            sample_size=request.sample_size,
            confidence_level=request.confidence_level,
            statistical_power=request.statistical_power,
            early_stopping_enabled=request.early_stopping_enabled,
            metadata={
                **request.metadata,
                "created_by": user["user_id"],
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        # Create experiment
        experiment_id = await service.create_experiment(config)
        
        logger.info(f"Created experiment: {experiment_id} by user {user['user_id']}")
        
        return ExperimentResponse(
            experiment_id=experiment_id,
            status="created",
            message="Experiment created successfully",
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments", response_model=ExperimentListResponse)
async def list_experiments(
    status: Optional[ExperimentStatus] = Query(None, description="Filter by experiment status"),
    page: int = Query(1, description="Page number"),
    page_size: int = Query(20, description="Number of experiments per page"),
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    List experiments with optional filtering
    
    Returns a paginated list of experiments with their basic information.
    """
    try:
        await rate_limiter.check_rate_limit(f"list_experiments:{user['user_id']}", max_requests=20, window_seconds=60)
        
        # Get all experiments
        experiments = []
        for experiment_id, config in service.experiments.items():
            summary = await service.get_experiment_summary(experiment_id)
            
            # Apply status filter
            if status and summary and summary.status != status:
                continue
            
            experiments.append({
                "experiment_id": experiment_id,
                "name": config.name,
                "description": config.description,
                "status": summary.status.value if summary else "unknown",
                "start_date": config.start_date.isoformat(),
                "end_date": config.end_date.isoformat(),
                "total_participants": summary.total_participants if summary else 0,
                "variants_count": len(config.variants),
                "metrics_count": len(config.metrics),
                "created_at": config.metadata.get("created_at", "")
            })
        
        # Apply pagination
        total_count = len(experiments)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_experiments = experiments[start_idx:end_idx]
        
        return ExperimentListResponse(
            experiments=paginated_experiments,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}", response_model=ExperimentSummaryResponse)
async def get_experiment(
    experiment_id: str,
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    Get detailed experiment information
    
    Returns comprehensive information about an experiment including current status,
    performance metrics, and statistical analysis results.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_experiment:{user['user_id']}", max_requests=30, window_seconds=60)
        
        if experiment_id not in service.experiments:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        config = service.experiments[experiment_id]
        summary = await service.get_experiment_summary(experiment_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Experiment summary not found")
        
        # Check for statistical significance
        statistical_significance = any(
            analysis.statistical_significance 
            for analysis in summary.statistical_analyses
        )
        
        return ExperimentSummaryResponse(
            experiment_id=experiment_id,
            status=summary.status.value,
            name=config.name,
            description=config.description,
            start_date=summary.start_date,
            end_date=summary.end_date,
            total_participants=summary.total_participants,
            variant_performance=summary.variant_performance,
            statistical_significance=statistical_significance,
            winning_variant=summary.winning_variant,
            confidence_score=summary.confidence_score,
            recommendations=summary.recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{experiment_id}/start", response_model=ExperimentResponse)
async def start_experiment(
    experiment_id: str,
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    Start an experiment
    
    Starts the experiment and begins traffic allocation to variants.
    """
    try:
        await rate_limiter.check_rate_limit(f"start_experiment:{user['user_id']}", max_requests=5, window_seconds=300)
        
        success = await service.start_experiment(experiment_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start experiment")
        
        logger.info(f"Started experiment: {experiment_id} by user {user['user_id']}")
        
        return ExperimentResponse(
            experiment_id=experiment_id,
            status="started",
            message="Experiment started successfully",
            created_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{experiment_id}/stop", response_model=ExperimentResponse)
async def stop_experiment(
    experiment_id: str,
    reason: str = Query("Manual stop", description="Reason for stopping"),
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    Stop an experiment
    
    Stops the experiment and performs final analysis.
    """
    try:
        await rate_limiter.check_rate_limit(f"stop_experiment:{user['user_id']}", max_requests=5, window_seconds=300)
        
        success = await service.stop_experiment(experiment_id, reason)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to stop experiment")
        
        logger.info(f"Stopped experiment: {experiment_id} by user {user['user_id']}, reason: {reason}")
        
        return ExperimentResponse(
            experiment_id=experiment_id,
            status="stopped",
            message="Experiment stopped successfully",
            created_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== TRAFFIC ALLOCATION =====

@router.post("/experiments/{experiment_id}/assign", response_model=VariantAssignmentResponse)
async def assign_variant(
    experiment_id: str,
    request: VariantAssignmentRequest,
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    Assign a user to an experiment variant
    
    Assigns a user to a variant based on the experiment's traffic allocation strategy.
    """
    try:
        await rate_limiter.check_rate_limit(f"assign_variant:{user['user_id']}", max_requests=100, window_seconds=60)
        
        variant_id = await service.assign_user_to_variant(
            experiment_id=experiment_id,
            user_id=request.user_id,
            context=request.context
        )
        
        return VariantAssignmentResponse(
            experiment_id=experiment_id,
            user_id=request.user_id,
            variant_id=variant_id,
            assigned_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error assigning variant: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}/assignment/{user_id}")
async def get_user_assignment(
    experiment_id: str,
    user_id: str,
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    Get user's variant assignment for an experiment
    
    Returns the variant assignment for a specific user in an experiment.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_assignment:{user['user_id']}", max_requests=50, window_seconds=60)
        
        if experiment_id not in service.user_assignments:
            raise HTTPException(status_code=404, detail="No assignments found for experiment")
        
        variant_id = service.user_assignments[experiment_id].get(user_id)
        
        if not variant_id:
            raise HTTPException(status_code=404, detail="User not assigned to experiment")
        
        return {
            "experiment_id": experiment_id,
            "user_id": user_id,
            "variant_id": variant_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user assignment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== EVENT TRACKING =====

@router.post("/experiments/{experiment_id}/track", response_model=EventTrackingResponse)
async def track_event(
    experiment_id: str,
    request: EventTrackingRequest,
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    Track an event/metric for an experiment
    
    Records a metric value for a user in an experiment.
    """
    try:
        await rate_limiter.check_rate_limit(f"track_event:{user['user_id']}", max_requests=1000, window_seconds=60)
        
        success = await service.track_event(
            experiment_id=experiment_id,
            user_id=request.user_id,
            metric_id=request.metric_id,
            value=request.value,
            context=request.context
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to track event")
        
        return EventTrackingResponse(
            success=True,
            message="Event tracked successfully",
            tracked_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== STATISTICAL ANALYSIS =====

@router.get("/experiments/{experiment_id}/analysis", response_model=List[StatisticalAnalysisResponse])
async def get_experiment_analysis(
    experiment_id: str,
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    Get statistical analysis for an experiment
    
    Returns detailed statistical analysis including significance testing,
    confidence intervals, and effect sizes.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_analysis:{user['user_id']}", max_requests=20, window_seconds=60)
        
        analyses = await service.analyze_experiment(experiment_id)
        
        if not analyses:
            raise HTTPException(status_code=404, detail="No analysis available for experiment")
        
        response = []
        for analysis in analyses:
            response.append(StatisticalAnalysisResponse(
                metric_id=analysis.metric_id,
                statistical_significance=analysis.statistical_significance,
                p_value=analysis.p_value,
                confidence_interval=list(analysis.confidence_interval),
                effect_size=analysis.effect_size,
                power=analysis.power,
                sample_size=analysis.sample_size,
                variant_comparisons=analysis.variant_comparisons
            ))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str,
    limit: int = Query(1000, description="Maximum number of results to return"),
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    Get raw experiment results
    
    Returns raw result data for an experiment.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_results:{user['user_id']}", max_requests=10, window_seconds=60)
        
        results = await service.get_experiment_results(experiment_id)
        
        if not results:
            return {"results": [], "total_count": 0}
        
        # Limit results
        limited_results = results[-limit:] if len(results) > limit else results
        
        # Convert to response format
        response_results = []
        for result in limited_results:
            response_results.append({
                "experiment_id": result.experiment_id,
                "variant_id": result.variant_id,
                "metric_id": result.metric_id,
                "timestamp": result.timestamp.isoformat(),
                "value": result.value,
                "user_id": result.user_id,
                "context": result.context
            })
        
        return {
            "results": response_results,
            "total_count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error getting experiment results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== MONITORING =====

@router.get("/experiments/active")
async def get_active_experiments(
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    Get list of active experiments
    
    Returns a list of currently running experiments.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_active:{user['user_id']}", max_requests=20, window_seconds=60)
        
        active_experiments = await service.get_active_experiments()
        
        response = []
        for experiment_id in active_experiments:
            if experiment_id in service.experiments:
                config = service.experiments[experiment_id]
                summary = await service.get_experiment_summary(experiment_id)
                
                response.append({
                    "experiment_id": experiment_id,
                    "name": config.name,
                    "status": summary.status.value if summary else "unknown",
                    "participants": summary.total_participants if summary else 0,
                    "start_date": config.start_date.isoformat(),
                    "end_date": config.end_date.isoformat()
                })
        
        return {"active_experiments": response}
        
    except Exception as e:
        logger.error(f"Error getting active experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_performance_metrics(
    user=Depends(get_current_user),
    service: ABTestingService = Depends(get_ab_testing_service)
):
    """
    Get A/B testing performance metrics
    
    Returns overall performance metrics for the A/B testing system.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_metrics:{user['user_id']}", max_requests=20, window_seconds=60)
        
        metrics = await service.get_performance_metrics()
        
        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns the health status of the A/B testing service.
    """
    try:
        # Check service health
        active_count = len(ab_testing_service.active_experiments)
        
        return {
            "status": "healthy",
            "active_experiments": active_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {"status": "unhealthy", "message": str(e)}

# ===== INITIALIZATION =====

async def initialize_ab_testing_api():
    """Initialize the A/B testing API"""
    
    try:
        await ab_testing_service.initialize()
        logger.info("A/B testing API initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing A/B testing API: {e}")
        raise

async def shutdown_ab_testing_api():
    """Shutdown the A/B testing API"""
    
    try:
        await ab_testing_service.shutdown()
        logger.info("A/B testing API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error shutting down A/B testing API: {e}") 