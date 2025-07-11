"""
Adaptive Workflow API

This module provides FastAPI endpoints for the adaptive workflow service:
- Workflow definition and management
- Execution control and monitoring
- Optimization and adaptation
- Performance analytics
- Resource management
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import logging

from ..core.adaptive_workflow_service import (
    AdaptiveWorkflowService,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowExecution,
    OptimizationResult,
    PerformanceMetrics,
    AdaptationEvent,
    WorkflowPriority,
    ExecutionStatus,
    OptimizationStrategy,
    AdaptationTrigger,
    ResourceType,
    adaptive_workflow_service
)
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

# Create the router
router = APIRouter(prefix="/api/v1/adaptive-workflow", tags=["adaptive-workflow"])

# ===== PYDANTIC MODELS =====

class WorkflowStepModel(BaseModel):
    """Workflow step model for API"""
    step_id: str
    name: str
    function: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    resources: Dict[ResourceType, float]
    timeout: int = 3600
    retry_count: int = 3
    retry_delay: int = 5
    priority: WorkflowPriority = WorkflowPriority.NORMAL
    metadata: Optional[Dict[str, Any]] = None

class WorkflowDefinitionModel(BaseModel):
    """Workflow definition model for API"""
    workflow_id: str
    name: str
    description: str
    version: str
    steps: List[WorkflowStepModel]
    triggers: List[str]
    schedule: Optional[str] = None
    max_concurrent: int = 1
    timeout: int = 3600
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    metadata: Optional[Dict[str, Any]] = None

class WorkflowExecutionModel(BaseModel):
    """Workflow execution model for API"""
    execution_id: str
    workflow_id: str
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time: float = 0.0
    resource_usage: Optional[Dict[ResourceType, float]] = None
    cost: float = 0.0
    step_results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    adaptation_applied: bool = False

class ExecutionRequestModel(BaseModel):
    """Execution request model"""
    workflow_id: str
    parameters: Optional[Dict[str, Any]] = None
    priority: WorkflowPriority = WorkflowPriority.NORMAL

class OptimizationRequestModel(BaseModel):
    """Optimization request model"""
    workflow_id: str
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    target_metrics: Optional[Dict[str, float]] = None

class OptimizationResultModel(BaseModel):
    """Optimization result model for API"""
    optimization_id: str
    workflow_id: str
    strategy: OptimizationStrategy
    improvements: Dict[str, float]
    new_configuration: Dict[str, Any]
    confidence: float
    created_at: datetime
    applied: bool = False

class PerformanceMetricsModel(BaseModel):
    """Performance metrics model for API"""
    execution_id: str
    workflow_id: str
    timestamp: datetime
    execution_time: float
    resource_efficiency: float
    cost_efficiency: float
    success_rate: float
    throughput: float
    latency: float
    error_rate: float
    adaptation_score: float

class AdaptationEventModel(BaseModel):
    """Adaptation event model for API"""
    event_id: str
    workflow_id: str
    trigger: AdaptationTrigger
    description: str
    action_taken: str
    impact: Dict[str, float]
    timestamp: datetime

class ResourceScalingModel(BaseModel):
    """Resource scaling model"""
    resource_type: ResourceType
    scale_factor: float = Field(gt=0, description="Scaling factor (1.0 = no change)")

class AdaptationRequestModel(BaseModel):
    """Adaptation request model"""
    workflow_id: str
    trigger: AdaptationTrigger
    description: str
    force: bool = False

# ===== WORKFLOW MANAGEMENT ENDPOINTS =====

@router.post("/workflows")
async def create_workflow(
    workflow: WorkflowDefinitionModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new workflow"""
    try:
        # Convert API model to domain model
        workflow_steps = []
        for step_data in workflow.steps:
            step = WorkflowStep(
                step_id=step_data.step_id,
                name=step_data.name,
                function=step_data.function,
                parameters=step_data.parameters,
                dependencies=step_data.dependencies,
                resources=step_data.resources,
                timeout=step_data.timeout,
                retry_count=step_data.retry_count,
                retry_delay=step_data.retry_delay,
                priority=step_data.priority,
                metadata=step_data.metadata or {}
            )
            workflow_steps.append(step)
        
        workflow_definition = WorkflowDefinition(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            description=workflow.description,
            version=workflow.version,
            steps=workflow_steps,
            triggers=workflow.triggers,
            schedule=workflow.schedule,
            max_concurrent=workflow.max_concurrent,
            timeout=workflow.timeout,
            optimization_strategy=workflow.optimization_strategy,
            metadata=workflow.metadata or {}
        )
        
        success = await adaptive_workflow_service.register_workflow(workflow_definition)
        
        if success:
            logger.info(f"Workflow {workflow.workflow_id} created by {user.get('username', 'unknown')}")
            return JSONResponse(
                content={"message": "Workflow created successfully", "workflow_id": workflow.workflow_id},
                status_code=201
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to create workflow")
            
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/workflows")
async def get_workflows(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all workflows"""
    try:
        workflows = await adaptive_workflow_service.get_workflow_list()
        return JSONResponse(content={"workflows": workflows})
        
    except Exception as e:
        logger.error(f"Error getting workflows: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get workflow details"""
    try:
        if workflow_id not in adaptive_workflow_service.workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow = adaptive_workflow_service.workflows[workflow_id]
        
        # Convert to API model
        workflow_steps = []
        for step in workflow.steps:
            step_model = WorkflowStepModel(
                step_id=step.step_id,
                name=step.name,
                function=step.function,
                parameters=step.parameters,
                dependencies=step.dependencies,
                resources=step.resources,
                timeout=step.timeout,
                retry_count=step.retry_count,
                retry_delay=step.retry_delay,
                priority=step.priority,
                metadata=step.metadata
            )
            workflow_steps.append(step_model)
        
        workflow_model = WorkflowDefinitionModel(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            description=workflow.description,
            version=workflow.version,
            steps=workflow_steps,
            triggers=workflow.triggers,
            schedule=workflow.schedule,
            max_concurrent=workflow.max_concurrent,
            timeout=workflow.timeout,
            optimization_strategy=workflow.optimization_strategy,
            metadata=workflow.metadata
        )
        
        return JSONResponse(content=workflow_model.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a workflow"""
    try:
        if workflow_id not in adaptive_workflow_service.workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check for running executions
        running_executions = [
            e for e in adaptive_workflow_service.executions.values()
            if e.workflow_id == workflow_id and e.status == ExecutionStatus.RUNNING
        ]
        
        if running_executions:
            raise HTTPException(status_code=400, detail="Cannot delete workflow with running executions")
        
        # Delete workflow
        del adaptive_workflow_service.workflows[workflow_id]
        
        # Clean up related data
        if workflow_id in adaptive_workflow_service.workflow_graph:
            del adaptive_workflow_service.workflow_graph[workflow_id]
        
        if workflow_id in adaptive_workflow_service.performance_metrics:
            del adaptive_workflow_service.performance_metrics[workflow_id]
        
        logger.info(f"Workflow {workflow_id} deleted by {user.get('username', 'unknown')}")
        
        return JSONResponse(content={"message": "Workflow deleted successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== EXECUTION MANAGEMENT ENDPOINTS =====

@router.post("/executions")
async def execute_workflow(
    request: ExecutionRequestModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Execute a workflow"""
    try:
        execution_id = await adaptive_workflow_service.execute_workflow(
            request.workflow_id,
            request.parameters
        )
        
        logger.info(f"Workflow execution {execution_id} started by {user.get('username', 'unknown')}")
        
        return JSONResponse(
            content={"message": "Workflow execution started", "execution_id": execution_id},
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/executions")
async def get_executions(
    workflow_id: Optional[str] = None,
    status: Optional[ExecutionStatus] = None,
    limit: int = Query(100, ge=1, le=1000),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get executions with optional filtering"""
    try:
        executions = await adaptive_workflow_service.list_executions(workflow_id, status)
        
        # Apply limit
        executions = executions[-limit:]
        
        # Convert to API models
        result = []
        for execution in executions:
            execution_model = WorkflowExecutionModel(
                execution_id=execution.execution_id,
                workflow_id=execution.workflow_id,
                status=execution.status,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                execution_time=execution.execution_time,
                resource_usage=execution.resource_usage,
                cost=execution.cost,
                step_results=execution.step_results,
                error_message=execution.error_message,
                retry_count=execution.retry_count,
                adaptation_applied=execution.adaptation_applied
            )
            result.append(execution_model)
        
        return JSONResponse(content={"executions": [e.dict() for e in result]})
        
    except Exception as e:
        logger.error(f"Error getting executions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/executions/{execution_id}")
async def get_execution(
    execution_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get execution details"""
    try:
        execution = await adaptive_workflow_service.get_execution_status(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        execution_model = WorkflowExecutionModel(
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id,
            status=execution.status,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            execution_time=execution.execution_time,
            resource_usage=execution.resource_usage,
            cost=execution.cost,
            step_results=execution.step_results,
            error_message=execution.error_message,
            retry_count=execution.retry_count,
            adaptation_applied=execution.adaptation_applied
        )
        
        return JSONResponse(content=execution_model.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/executions/{execution_id}")
async def cancel_execution(
    execution_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Cancel an execution"""
    try:
        success = await adaptive_workflow_service.cancel_execution(execution_id)
        
        if success:
            logger.info(f"Execution {execution_id} cancelled by {user.get('username', 'unknown')}")
            return JSONResponse(content={"message": "Execution cancelled successfully"})
        else:
            raise HTTPException(status_code=404, detail="Execution not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling execution: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/executions/{execution_id}/logs")
async def get_execution_logs(
    execution_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get execution logs"""
    try:
        execution = await adaptive_workflow_service.get_execution_status(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        # Return step results as logs
        logs = []
        for step_id, result in execution.step_results.items():
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "step_id": step_id,
                "level": "INFO" if result.get("status") == "completed" else "ERROR",
                "message": result.get("result", result.get("error", "Unknown")),
                "details": result
            }
            logs.append(log_entry)
        
        return JSONResponse(content={"logs": logs})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution logs: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== OPTIMIZATION ENDPOINTS =====

@router.post("/optimizations")
async def optimize_workflow(
    request: OptimizationRequestModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Optimize a workflow"""
    try:
        optimization_result = await adaptive_workflow_service.optimize_workflow(
            request.workflow_id,
            request.strategy
        )
        
        result_model = OptimizationResultModel(
            optimization_id=optimization_result.optimization_id,
            workflow_id=optimization_result.workflow_id,
            strategy=optimization_result.strategy,
            improvements=optimization_result.improvements,
            new_configuration=optimization_result.new_configuration,
            confidence=optimization_result.confidence,
            created_at=optimization_result.created_at,
            applied=optimization_result.applied
        )
        
        logger.info(f"Workflow optimization {optimization_result.optimization_id} created by {user.get('username', 'unknown')}")
        
        return JSONResponse(
            content={"message": "Workflow optimization completed", "optimization": result_model.dict()},
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error optimizing workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/optimizations")
async def get_optimizations(
    workflow_id: Optional[str] = None,
    applied: Optional[bool] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get optimizations with optional filtering"""
    try:
        optimizations = list(adaptive_workflow_service.optimization_results.values())
        
        # Apply filters
        if workflow_id:
            optimizations = [o for o in optimizations if o.workflow_id == workflow_id]
        
        if applied is not None:
            optimizations = [o for o in optimizations if o.applied == applied]
        
        # Convert to API models
        result = []
        for optimization in optimizations:
            result_model = OptimizationResultModel(
                optimization_id=optimization.optimization_id,
                workflow_id=optimization.workflow_id,
                strategy=optimization.strategy,
                improvements=optimization.improvements,
                new_configuration=optimization.new_configuration,
                confidence=optimization.confidence,
                created_at=optimization.created_at,
                applied=optimization.applied
            )
            result.append(result_model)
        
        return JSONResponse(content={"optimizations": [o.dict() for o in result]})
        
    except Exception as e:
        logger.error(f"Error getting optimizations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/optimizations/{optimization_id}/apply")
async def apply_optimization(
    optimization_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Apply an optimization"""
    try:
        if optimization_id not in adaptive_workflow_service.optimization_results:
            raise HTTPException(status_code=404, detail="Optimization not found")
        
        optimization = adaptive_workflow_service.optimization_results[optimization_id]
        
        success = await adaptive_workflow_service.apply_optimization(
            optimization.workflow_id,
            optimization_id
        )
        
        if success:
            logger.info(f"Optimization {optimization_id} applied by {user.get('username', 'unknown')}")
            return JSONResponse(content={"message": "Optimization applied successfully"})
        else:
            raise HTTPException(status_code=400, detail="Failed to apply optimization")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== ADAPTATION ENDPOINTS =====

@router.post("/adaptations")
async def trigger_adaptation(
    request: AdaptationRequestModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Trigger workflow adaptation"""
    try:
        success = await adaptive_workflow_service.auto_adapt_workflow(
            request.workflow_id,
            request.trigger
        )
        
        if success:
            logger.info(f"Adaptation triggered for workflow {request.workflow_id} by {user.get('username', 'unknown')}")
            return JSONResponse(content={"message": "Adaptation triggered successfully"})
        else:
            raise HTTPException(status_code=400, detail="Failed to trigger adaptation")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering adaptation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/adaptations")
async def get_adaptations(
    workflow_id: Optional[str] = None,
    trigger: Optional[AdaptationTrigger] = None,
    limit: int = Query(100, ge=1, le=1000),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get adaptation events with optional filtering"""
    try:
        adaptations = list(adaptive_workflow_service.adaptation_events)
        
        # Apply filters
        if workflow_id:
            adaptations = [a for a in adaptations if a.workflow_id == workflow_id]
        
        if trigger:
            adaptations = [a for a in adaptations if a.trigger == trigger]
        
        # Apply limit
        adaptations = adaptations[-limit:]
        
        # Convert to API models
        result = []
        for adaptation in adaptations:
            adaptation_model = AdaptationEventModel(
                event_id=adaptation.event_id,
                workflow_id=adaptation.workflow_id,
                trigger=adaptation.trigger,
                description=adaptation.description,
                action_taken=adaptation.action_taken,
                impact=adaptation.impact,
                timestamp=adaptation.timestamp
            )
            result.append(adaptation_model)
        
        return JSONResponse(content={"adaptations": [a.dict() for a in result]})
        
    except Exception as e:
        logger.error(f"Error getting adaptations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== RESOURCE MANAGEMENT ENDPOINTS =====

@router.get("/resources")
async def get_resources(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get resource information"""
    try:
        resource_info = {
            "pools": {},
            "usage": adaptive_workflow_service.resource_usage,
            "allocations": {}
        }
        
        # Resource pools
        for resource_type, pool in adaptive_workflow_service.resource_pools.items():
            resource_info["pools"][resource_type.value] = {
                "capacity": pool["capacity"],
                "allocated": pool["allocated"],
                "available": pool["available"],
                "utilization": (pool["allocated"] / pool["capacity"]) * 100 if pool["capacity"] > 0 else 0
            }
        
        # Current allocations
        for execution_id, allocations in adaptive_workflow_service.resource_allocations.items():
            resource_info["allocations"][execution_id] = [
                {
                    "resource_type": alloc.resource_type.value,
                    "allocated": alloc.allocated,
                    "used": alloc.used,
                    "efficiency": alloc.efficiency
                } for alloc in allocations
            ]
        
        return JSONResponse(content=resource_info)
        
    except Exception as e:
        logger.error(f"Error getting resources: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/resources/scale")
async def scale_resources(
    scaling: ResourceScalingModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Scale resources"""
    try:
        success = await adaptive_workflow_service.scale_resources(
            scaling.resource_type,
            scaling.scale_factor
        )
        
        if success:
            logger.info(f"Resource {scaling.resource_type.value} scaled by {scaling.scale_factor} by {user.get('username', 'unknown')}")
            return JSONResponse(content={"message": "Resources scaled successfully"})
        else:
            raise HTTPException(status_code=400, detail="Failed to scale resources")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scaling resources: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== ANALYTICS ENDPOINTS =====

@router.get("/analytics/dashboard")
async def get_analytics_dashboard(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get analytics dashboard"""
    try:
        # Get system status
        system_status = await adaptive_workflow_service.get_system_status()
        
        # Get recent performance metrics
        recent_metrics = []
        for workflow_id, metrics in adaptive_workflow_service.performance_metrics.items():
            if metrics:
                recent_metrics.extend(list(metrics)[-10:])  # Last 10 metrics per workflow
        
        # Calculate dashboard metrics
        dashboard_data = {
            "system_status": system_status,
            "recent_performance": {
                "total_executions": len(recent_metrics),
                "average_execution_time": sum(m.execution_time for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                "average_success_rate": sum(m.success_rate for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                "average_resource_efficiency": sum(m.resource_efficiency for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
            },
            "resource_utilization": {
                resource_type.value: usage for resource_type, usage in adaptive_workflow_service.resource_usage.items()
            },
            "recent_adaptations": len([e for e in adaptive_workflow_service.adaptation_events if e.timestamp > datetime.utcnow() - timedelta(hours=24)]),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting analytics dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/analytics/performance")
async def get_performance_analytics(
    workflow_id: Optional[str] = None,
    time_range: int = Query(24, description="Time range in hours"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get performance analytics"""
    try:
        analytics = await adaptive_workflow_service.get_performance_analytics(workflow_id)
        
        # Add time-based filtering
        if workflow_id and workflow_id in adaptive_workflow_service.performance_metrics:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_range)
            
            metrics = [
                m for m in adaptive_workflow_service.performance_metrics[workflow_id]
                if m.timestamp >= start_time
            ]
            
            analytics["time_series"] = {
                "execution_times": [m.execution_time for m in metrics],
                "success_rates": [m.success_rate for m in metrics],
                "resource_efficiencies": [m.resource_efficiency for m in metrics],
                "timestamps": [m.timestamp.isoformat() for m in metrics]
            }
        
        return JSONResponse(content=analytics)
        
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/analytics/trends")
async def get_trends(
    workflow_id: Optional[str] = None,
    metric: str = Query("execution_time", description="Metric to analyze"),
    time_range: int = Query(168, description="Time range in hours"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get trend analysis"""
    try:
        trends = {
            "metric": metric,
            "time_range": time_range,
            "workflows": {}
        }
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_range)
        
        # Get metrics for analysis
        workflows_to_analyze = [workflow_id] if workflow_id else list(adaptive_workflow_service.performance_metrics.keys())
        
        for wf_id in workflows_to_analyze:
            if wf_id in adaptive_workflow_service.performance_metrics:
                metrics = [
                    m for m in adaptive_workflow_service.performance_metrics[wf_id]
                    if m.timestamp >= start_time
                ]
                
                if metrics:
                    if metric == "execution_time":
                        values = [m.execution_time for m in metrics]
                    elif metric == "success_rate":
                        values = [m.success_rate for m in metrics]
                    elif metric == "resource_efficiency":
                        values = [m.resource_efficiency for m in metrics]
                    else:
                        values = [m.execution_time for m in metrics]  # Default
                    
                    # Calculate trend
                    if len(values) >= 2:
                        # Simple linear trend
                        x = list(range(len(values)))
                        y = values
                        
                        # Calculate slope
                        n = len(values)
                        sum_x = sum(x)
                        sum_y = sum(y)
                        sum_xy = sum(x[i] * y[i] for i in range(n))
                        sum_x2 = sum(x[i] * x[i] for i in range(n))
                        
                        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
                        
                        trends["workflows"][wf_id] = {
                            "trend": "improving" if slope < 0 else "declining" if slope > 0 else "stable",
                            "slope": slope,
                            "values": values,
                            "timestamps": [m.timestamp.isoformat() for m in metrics]
                        }
        
        return JSONResponse(content=trends)
        
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== SYSTEM MANAGEMENT ENDPOINTS =====

@router.get("/system/status")
async def get_system_status(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get system status"""
    try:
        status = await adaptive_workflow_service.get_system_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/system/initialize")
async def initialize_system(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Initialize the adaptive workflow system"""
    try:
        await adaptive_workflow_service.initialize()
        
        logger.info(f"Adaptive workflow system initialized by {user.get('username', 'unknown')}")
        
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
        config = adaptive_workflow_service.config
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
        valid_keys = set(adaptive_workflow_service.config.keys())
        for key, value in config.items():
            if key in valid_keys:
                adaptive_workflow_service.config[key] = value
        
        logger.info(f"System configuration updated by {user.get('username', 'unknown')}")
        
        return JSONResponse(content={"message": "Configuration updated successfully"})
        
    except Exception as e:
        logger.error(f"Error updating system config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={"status": "healthy", "service": "adaptive-workflow"})

# ===== INITIALIZATION FUNCTIONS =====

async def initialize_adaptive_workflow_api():
    """Initialize the adaptive workflow API"""
    try:
        await adaptive_workflow_service.initialize()
        logger.info("Adaptive Workflow API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Adaptive Workflow API: {e}")
        raise

async def shutdown_adaptive_workflow_api():
    """Shutdown the adaptive workflow API"""
    try:
        await adaptive_workflow_service.shutdown()
        logger.info("Adaptive Workflow API shutdown successfully")
    except Exception as e:
        logger.error(f"Error shutting down Adaptive Workflow API: {e}") 