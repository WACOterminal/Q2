"""
Context-Aware Decision API

This API provides endpoints for intelligent decision making:
- Decision request processing
- Rule management
- Feedback collection
- Analytics and insights
- Context analysis
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from ..core.context_aware_decision_service import (
    ContextAwareDecisionService,
    context_aware_decision_service,
    DecisionRequest,
    DecisionResult,
    DecisionFeedback,
    DecisionRule,
    DecisionType,
    DecisionStrategy,
    DecisionOutcome,
    ConfidenceLevel,
    ContextType,
    DecisionOption,
    DecisionCriteria
)
from ..core.auth_service import auth_service
from ..core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Initialize components
router = APIRouter(prefix="/api/v1/context-decision", tags=["Context-Aware Decision"])
security = HTTPBearer()
rate_limiter = RateLimiter()

# ===== REQUEST/RESPONSE MODELS =====

class DecisionOptionRequest(BaseModel):
    """Request model for decision option"""
    option_id: str = Field(..., description="Option ID")
    name: str = Field(..., description="Option name")
    description: str = Field(..., description="Option description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Option parameters")
    estimated_outcomes: Dict[str, Any] = Field(default_factory=dict, description="Estimated outcomes")
    feasibility_score: float = Field(..., description="Feasibility score (0-1)")
    risk_score: float = Field(..., description="Risk score (0-1)")
    cost: float = Field(..., description="Cost")
    benefit: float = Field(..., description="Benefit")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class DecisionCriteriaRequest(BaseModel):
    """Request model for decision criteria"""
    criteria_id: str = Field(..., description="Criteria ID")
    name: str = Field(..., description="Criteria name")
    description: str = Field(..., description="Criteria description")
    weight: float = Field(..., description="Criteria weight")
    target_value: Any = Field(..., description="Target value")
    threshold: Optional[float] = Field(None, description="Threshold value")
    direction: str = Field(default="maximize", description="Direction (maximize, minimize, target)")
    mandatory: bool = Field(default=False, description="Whether criteria is mandatory")

class DecisionRequestModel(BaseModel):
    """Request model for decision making"""
    request_id: Optional[str] = Field(None, description="Request ID")
    decision_type: DecisionType = Field(..., description="Decision type")
    context: Dict[str, Any] = Field(default_factory=dict, description="Decision context")
    criteria: List[DecisionCriteriaRequest] = Field(..., description="Decision criteria")
    options: List[DecisionOptionRequest] = Field(..., description="Decision options")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Constraints")
    strategy: DecisionStrategy = Field(default=DecisionStrategy.HYBRID, description="Decision strategy")
    priority: int = Field(default=0, description="Priority")
    deadline: Optional[datetime] = Field(None, description="Decision deadline")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class DecisionResultResponse(BaseModel):
    """Response model for decision result"""
    result_id: str
    request_id: str
    selected_option: str
    confidence_score: float
    confidence_level: str
    reasoning: str
    criteria_scores: Dict[str, float]
    alternative_options: List[List[Any]]
    decision_time: datetime
    expiry_time: Optional[datetime] = None

class DecisionFeedbackRequest(BaseModel):
    """Request model for decision feedback"""
    result_id: str = Field(..., description="Result ID")
    outcome: DecisionOutcome = Field(..., description="Decision outcome")
    actual_metrics: Dict[str, Any] = Field(default_factory=dict, description="Actual metrics")
    user_satisfaction: Optional[float] = Field(None, description="User satisfaction score")
    lessons_learned: List[str] = Field(default_factory=list, description="Lessons learned")

class DecisionRuleRequest(BaseModel):
    """Request model for decision rule"""
    rule_id: Optional[str] = Field(None, description="Rule ID")
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    conditions: List[Dict[str, Any]] = Field(..., description="Rule conditions")
    actions: List[Dict[str, Any]] = Field(..., description="Rule actions")
    priority: int = Field(default=0, description="Rule priority")
    enabled: bool = Field(default=True, description="Whether rule is enabled")

class DecisionMetricsResponse(BaseModel):
    """Response model for decision metrics"""
    total_decisions: int
    successful_decisions: int
    success_rate: float
    average_decision_time: float
    average_confidence: float
    rule_hits: int
    ml_predictions: int
    feedback_collected: int
    total_rules: int
    active_models: int

# ===== DEPENDENCY INJECTION =====

def get_decision_service() -> ContextAwareDecisionService:
    """Get the context-aware decision service instance"""
    return context_aware_decision_service

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    return auth_service.verify_token(credentials.credentials)

# ===== DECISION MAKING =====

@router.post("/decisions", response_model=DecisionResultResponse)
async def make_decision(
    request: DecisionRequestModel,
    user=Depends(get_current_user),
    service: ContextAwareDecisionService = Depends(get_decision_service)
):
    """
    Make a context-aware decision
    
    Processes a decision request using the specified strategy and context.
    """
    try:
        # Rate limiting
        await rate_limiter.check_rate_limit(f"decision:{user['user_id']}", max_requests=20, window_seconds=60)
        
        # Convert request to service model
        criteria = [
            DecisionCriteria(
                criteria_id=c.criteria_id,
                name=c.name,
                description=c.description,
                weight=c.weight,
                target_value=c.target_value,
                threshold=c.threshold,
                direction=c.direction,
                mandatory=c.mandatory
            ) for c in request.criteria
        ]
        
        options = [
            DecisionOption(
                option_id=o.option_id,
                name=o.name,
                description=o.description,
                parameters=o.parameters,
                estimated_outcomes=o.estimated_outcomes,
                feasibility_score=o.feasibility_score,
                risk_score=o.risk_score,
                cost=o.cost,
                benefit=o.benefit,
                metadata=o.metadata
            ) for o in request.options
        ]
        
        decision_request = DecisionRequest(
            request_id=request.request_id or "",
            decision_type=request.decision_type,
            context={
                **request.context,
                "user_id": user["user_id"],
                "submitted_at": datetime.utcnow().isoformat()
            },
            criteria=criteria,
            options=options,
            constraints=request.constraints,
            strategy=request.strategy,
            user_id=user["user_id"],
            priority=request.priority,
            deadline=request.deadline,
            metadata=request.metadata
        )
        
        # Make decision
        result = await service.make_decision(decision_request)
        
        logger.info(f"Decision made: {result.result_id} by user {user['user_id']}")
        
        return DecisionResultResponse(
            result_id=result.result_id,
            request_id=result.request_id,
            selected_option=result.selected_option,
            confidence_score=result.confidence_score,
            confidence_level=result.confidence_level.value,
            reasoning=result.reasoning,
            criteria_scores=result.criteria_scores,
            alternative_options=result.alternative_options,
            decision_time=result.decision_time,
            expiry_time=result.expiry_time
        )
        
    except Exception as e:
        logger.error(f"Error making decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/decisions/{result_id}")
async def get_decision_result(
    result_id: str,
    user=Depends(get_current_user),
    service: ContextAwareDecisionService = Depends(get_decision_service)
):
    """
    Get decision result details
    
    Returns detailed information about a specific decision result.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_decision:{user['user_id']}", max_requests=50, window_seconds=60)
        
        if result_id not in service.decision_results:
            raise HTTPException(status_code=404, detail="Decision result not found")
        
        result = service.decision_results[result_id]
        
        return {
            "result_id": result.result_id,
            "request_id": result.request_id,
            "selected_option": result.selected_option,
            "confidence_score": result.confidence_score,
            "confidence_level": result.confidence_level.value,
            "reasoning": result.reasoning,
            "criteria_scores": result.criteria_scores,
            "alternative_options": result.alternative_options,
            "decision_time": result.decision_time.isoformat(),
            "expiry_time": result.expiry_time.isoformat() if result.expiry_time else None,
            "context_factors": len(result.context_factors),
            "metadata": result.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting decision result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decisions/{result_id}/feedback")
async def provide_feedback(
    result_id: str,
    feedback: DecisionFeedbackRequest,
    user=Depends(get_current_user),
    service: ContextAwareDecisionService = Depends(get_decision_service)
):
    """
    Provide feedback for a decision
    
    Records feedback about a decision outcome for learning purposes.
    """
    try:
        await rate_limiter.check_rate_limit(f"feedback:{user['user_id']}", max_requests=30, window_seconds=60)
        
        if result_id not in service.decision_results:
            raise HTTPException(status_code=404, detail="Decision result not found")
        
        # Create feedback
        decision_feedback = DecisionFeedback(
            feedback_id=f"fb_{result_id}_{datetime.utcnow().timestamp()}",
            result_id=result_id,
            outcome=feedback.outcome,
            actual_metrics=feedback.actual_metrics,
            user_satisfaction=feedback.user_satisfaction,
            lessons_learned=feedback.lessons_learned,
            timestamp=datetime.utcnow()
        )
        
        # Record feedback
        success = await service.record_feedback(decision_feedback)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to record feedback")
        
        logger.info(f"Feedback recorded: {decision_feedback.feedback_id} by user {user['user_id']}")
        
        return {
            "feedback_id": decision_feedback.feedback_id,
            "result_id": result_id,
            "outcome": feedback.outcome.value,
            "recorded_at": datetime.utcnow().isoformat(),
            "message": "Feedback recorded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error providing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== RULE MANAGEMENT =====

@router.post("/rules")
async def create_rule(
    rule: DecisionRuleRequest,
    user=Depends(get_current_user),
    service: ContextAwareDecisionService = Depends(get_decision_service)
):
    """
    Create a decision rule
    
    Creates a new rule for automated decision making.
    """
    try:
        await rate_limiter.check_rate_limit(f"create_rule:{user['user_id']}", max_requests=10, window_seconds=3600)
        
        # Create rule
        decision_rule = DecisionRule(
            rule_id=rule.rule_id or f"rule_{datetime.utcnow().timestamp()}",
            name=rule.name,
            description=rule.description,
            conditions=rule.conditions,
            actions=rule.actions,
            priority=rule.priority,
            enabled=rule.enabled,
            created_at=datetime.utcnow()
        )
        
        # Save rule
        success = await service.create_decision_rule(decision_rule)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to create rule")
        
        logger.info(f"Rule created: {decision_rule.rule_id} by user {user['user_id']}")
        
        return {
            "rule_id": decision_rule.rule_id,
            "name": decision_rule.name,
            "description": decision_rule.description,
            "priority": decision_rule.priority,
            "enabled": decision_rule.enabled,
            "created_at": decision_rule.created_at.isoformat(),
            "message": "Rule created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rules")
async def list_rules(
    user=Depends(get_current_user),
    service: ContextAwareDecisionService = Depends(get_decision_service)
):
    """
    List all decision rules
    
    Returns a list of all configured decision rules.
    """
    try:
        await rate_limiter.check_rate_limit(f"list_rules:{user['user_id']}", max_requests=20, window_seconds=60)
        
        rules = []
        for rule in service.decision_rules.values():
            rules.append({
                "rule_id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "priority": rule.priority,
                "enabled": rule.enabled,
                "created_at": rule.created_at.isoformat(),
                "conditions_count": len(rule.conditions),
                "actions_count": len(rule.actions)
            })
        
        return {
            "rules": rules,
            "total_count": len(rules)
        }
        
    except Exception as e:
        logger.error(f"Error listing rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/rules/{rule_id}")
async def update_rule(
    rule_id: str,
    updates: Dict[str, Any],
    user=Depends(get_current_user),
    service: ContextAwareDecisionService = Depends(get_decision_service)
):
    """
    Update a decision rule
    
    Updates an existing decision rule with new configuration.
    """
    try:
        await rate_limiter.check_rate_limit(f"update_rule:{user['user_id']}", max_requests=10, window_seconds=3600)
        
        # Update rule
        success = await service.update_decision_rule(rule_id, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        logger.info(f"Rule updated: {rule_id} by user {user['user_id']}")
        
        return {
            "rule_id": rule_id,
            "updated_at": datetime.utcnow().isoformat(),
            "message": "Rule updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/rules/{rule_id}")
async def delete_rule(
    rule_id: str,
    user=Depends(get_current_user),
    service: ContextAwareDecisionService = Depends(get_decision_service)
):
    """
    Delete a decision rule
    
    Removes a decision rule from the system.
    """
    try:
        await rate_limiter.check_rate_limit(f"delete_rule:{user['user_id']}", max_requests=10, window_seconds=3600)
        
        # Delete rule
        success = await service.delete_decision_rule(rule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        logger.info(f"Rule deleted: {rule_id} by user {user['user_id']}")
        
        return {
            "rule_id": rule_id,
            "deleted_at": datetime.utcnow().isoformat(),
            "message": "Rule deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ANALYTICS =====

@router.get("/metrics", response_model=DecisionMetricsResponse)
async def get_metrics(
    user=Depends(get_current_user),
    service: ContextAwareDecisionService = Depends(get_decision_service)
):
    """
    Get decision metrics
    
    Returns comprehensive metrics about decision making performance.
    """
    try:
        await rate_limiter.check_rate_limit(f"metrics:{user['user_id']}", max_requests=10, window_seconds=60)
        
        metrics = await service.get_decision_metrics()
        
        return DecisionMetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_decision_history(
    limit: int = Query(50, description="Number of decisions to return"),
    user=Depends(get_current_user),
    service: ContextAwareDecisionService = Depends(get_decision_service)
):
    """
    Get decision history
    
    Returns recent decision history with outcomes.
    """
    try:
        await rate_limiter.check_rate_limit(f"history:{user['user_id']}", max_requests=20, window_seconds=60)
        
        history = await service.get_decision_history(limit)
        
        return {
            "history": history,
            "total_count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting decision history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/context")
async def get_context_analytics(
    user=Depends(get_current_user),
    service: ContextAwareDecisionService = Depends(get_decision_service)
):
    """
    Get context analytics
    
    Returns analytics about context factors and their impact on decisions.
    """
    try:
        await rate_limiter.check_rate_limit(f"context_analytics:{user['user_id']}", max_requests=10, window_seconds=60)
        
        # Analyze context factors
        context_stats = {}
        for context_type in ContextType:
            context_stats[context_type.value] = {
                "usage_count": 0,
                "average_weight": 0.0,
                "impact_score": 0.0
            }
        
        # Calculate context factor statistics
        total_factors = 0
        for result in service.decision_results.values():
            for factor in result.context_factors:
                context_stats[factor.context_type.value]["usage_count"] += 1
                context_stats[factor.context_type.value]["average_weight"] += factor.weight
                total_factors += 1
        
        # Calculate averages
        for stats in context_stats.values():
            if stats["usage_count"] > 0:
                stats["average_weight"] /= stats["usage_count"]
                stats["impact_score"] = stats["usage_count"] / max(total_factors, 1)
        
        return {
            "context_statistics": context_stats,
            "total_context_factors": total_factors,
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting context analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== HEALTH AND STATUS =====

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns the health status of the context-aware decision service.
    """
    try:
        # Check service health
        metrics = await context_aware_decision_service.get_decision_metrics()
        
        return {
            "status": "healthy",
            "total_decisions": metrics["total_decisions"],
            "success_rate": metrics["success_rate"],
            "active_rules": metrics["total_rules"],
            "active_models": metrics["active_models"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {"status": "unhealthy", "message": str(e)}

# ===== INITIALIZATION =====

async def initialize_context_decision_api():
    """Initialize the context-aware decision API"""
    
    try:
        await context_aware_decision_service.initialize()
        logger.info("Context-aware decision API initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing context-aware decision API: {e}")
        raise

async def shutdown_context_decision_api():
    """Shutdown the context-aware decision API"""
    
    try:
        await context_aware_decision_service.shutdown()
        logger.info("Context-aware decision API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error shutting down context-aware decision API: {e}") 