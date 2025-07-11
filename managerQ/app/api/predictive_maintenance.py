"""
Predictive Maintenance API

This module provides FastAPI endpoints for the predictive maintenance service:
- Component registration and health monitoring
- Anomaly detection and alerting
- Maintenance scheduling and management
- Performance metrics and analytics
- Failure prediction and prevention
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import logging

from ..core.predictive_maintenance_service import (
    PredictiveMaintenanceService,
    ComponentHealth,
    SystemMetric,
    MaintenanceTask,
    MaintenanceWindow,
    Anomaly,
    FailurePrediction,
    ComponentType,
    HealthStatus,
    MaintenanceType,
    MaintenanceStatus,
    AlertSeverity,
    AnomalyType,
    predictive_maintenance_service
)
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

# Create the router
router = APIRouter(prefix="/api/v1/predictive-maintenance", tags=["predictive-maintenance"])

# ===== PYDANTIC MODELS =====

class SystemMetricModel(BaseModel):
    """System metric model for API"""
    metric_id: str
    component_id: str
    component_type: ComponentType
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class ComponentHealthModel(BaseModel):
    """Component health model for API"""
    component_id: str
    component_type: ComponentType
    name: str
    status: HealthStatus
    health_score: float
    last_check: datetime
    metrics: List[SystemMetricModel]
    issues: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ComponentRegistrationModel(BaseModel):
    """Component registration model"""
    component_id: str
    component_type: ComponentType
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MaintenanceTaskModel(BaseModel):
    """Maintenance task model for API"""
    task_id: str
    component_id: str
    maintenance_type: MaintenanceType
    title: str
    description: str
    priority: int = Field(ge=1, le=10)
    estimated_duration: int  # minutes
    scheduled_time: datetime
    status: MaintenanceStatus
    assigned_to: Optional[str] = None
    dependencies: Optional[List[str]] = None
    resources_required: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class MaintenanceWindowModel(BaseModel):
    """Maintenance window model for API"""
    window_id: str
    name: str
    start_time: datetime
    end_time: datetime
    components: List[str]
    maintenance_type: MaintenanceType
    impact_assessment: str
    approval_required: bool = True
    created_by: Optional[str] = None

class AnomalyModel(BaseModel):
    """Anomaly model for API"""
    anomaly_id: str
    component_id: str
    anomaly_type: AnomalyType
    description: str
    severity: AlertSeverity
    detected_at: datetime
    metric_values: Dict[str, float]
    confidence_score: float
    predicted_impact: str
    recommended_actions: Optional[List[str]] = None

class FailurePredictionModel(BaseModel):
    """Failure prediction model for API"""
    prediction_id: str
    component_id: str
    predicted_failure_time: datetime
    confidence: float
    failure_type: str
    contributing_factors: List[str]
    recommended_actions: List[str]
    created_at: datetime

class MetricUpdateModel(BaseModel):
    """Metric update model"""
    metrics: List[SystemMetricModel]

class MaintenanceScheduleRequest(BaseModel):
    """Maintenance schedule request model"""
    component_id: str
    maintenance_type: MaintenanceType
    title: str
    description: str
    priority: int = Field(ge=1, le=10)
    estimated_duration: int  # minutes
    preferred_time: Optional[datetime] = None
    dependencies: Optional[List[str]] = None
    resources_required: Optional[List[str]] = None

class AlertConfigModel(BaseModel):
    """Alert configuration model"""
    component_id: str
    metric_name: str
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==, !=
    alert_severity: AlertSeverity
    cooldown_period: int = 1800  # seconds

# ===== COMPONENT MANAGEMENT ENDPOINTS =====

@router.post("/components/register")
async def register_component(
    component: ComponentRegistrationModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Register a new component for monitoring"""
    try:
        # Create component health object
        component_health = ComponentHealth(
            component_id=component.component_id,
            component_type=component.component_type,
            name=component.name,
            status=HealthStatus.UNKNOWN,
            health_score=0.5,
            last_check=datetime.utcnow(),
            metrics=[],
            metadata=component.metadata or {}
        )
        
        success = await predictive_maintenance_service.register_component(component_health)
        
        if success:
            logger.info(f"Component {component.component_id} registered by {user.get('username', 'unknown')}")
            return JSONResponse(
                content={"message": "Component registered successfully", "component_id": component.component_id},
                status_code=201
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to register component")
            
    except Exception as e:
        logger.error(f"Error registering component: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/components")
async def get_components(
    component_type: Optional[ComponentType] = None,
    status: Optional[HealthStatus] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all registered components with optional filtering"""
    try:
        components = predictive_maintenance_service.components
        
        # Apply filters
        filtered_components = {}
        for comp_id, comp in components.items():
            if component_type and comp.component_type != component_type:
                continue
            if status and comp.status != status:
                continue
            filtered_components[comp_id] = comp
        
        # Convert to API models
        result = []
        for comp_id, comp in filtered_components.items():
            metrics = [
                SystemMetricModel(
                    metric_id=metric.metric_id,
                    component_id=metric.component_id,
                    component_type=metric.component_type,
                    metric_name=metric.metric_name,
                    value=metric.value,
                    unit=metric.unit,
                    timestamp=metric.timestamp,
                    metadata=metric.metadata
                ) for metric in comp.metrics
            ]
            
            result.append(ComponentHealthModel(
                component_id=comp.component_id,
                component_type=comp.component_type,
                name=comp.name,
                status=comp.status,
                health_score=comp.health_score,
                last_check=comp.last_check,
                metrics=metrics,
                issues=comp.issues,
                recommendations=comp.recommendations,
                metadata=comp.metadata
            ))
        
        return JSONResponse(content={"components": [comp.dict() for comp in result]})
        
    except Exception as e:
        logger.error(f"Error getting components: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/components/{component_id}")
async def get_component_details(
    component_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get detailed information about a specific component"""
    try:
        details = await predictive_maintenance_service.get_component_details(component_id)
        
        if not details:
            raise HTTPException(status_code=404, detail="Component not found")
        
        return JSONResponse(content=details)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting component details: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.put("/components/{component_id}/metrics")
async def update_component_metrics(
    component_id: str,
    metrics: MetricUpdateModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Update component metrics"""
    try:
        # Convert API models to domain models
        domain_metrics = []
        for metric in metrics.metrics:
            domain_metric = SystemMetric(
                metric_id=metric.metric_id,
                component_id=metric.component_id,
                component_type=metric.component_type,
                metric_name=metric.metric_name,
                value=metric.value,
                unit=metric.unit,
                timestamp=metric.timestamp,
                metadata=metric.metadata
            )
            domain_metrics.append(domain_metric)
        
        success = await predictive_maintenance_service.update_component_health(component_id, domain_metrics)
        
        if success:
            return JSONResponse(content={"message": "Component metrics updated successfully"})
        else:
            raise HTTPException(status_code=400, detail="Failed to update component metrics")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating component metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== HEALTH MONITORING ENDPOINTS =====

@router.get("/health/system")
async def get_system_health(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get overall system health status"""
    try:
        health_status = await predictive_maintenance_service.get_system_health()
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health/metrics")
async def get_health_metrics(
    component_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    metric_names: Optional[List[str]] = Query(None),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get health metrics for components"""
    try:
        # Get performance metrics
        performance_metrics = await predictive_maintenance_service.get_performance_metrics()
        
        # Filter by component if specified
        if component_id:
            if component_id not in predictive_maintenance_service.components:
                raise HTTPException(status_code=404, detail="Component not found")
            
            component_metrics = {}
            for key, values in predictive_maintenance_service.metrics_history.items():
                if key.startswith(component_id):
                    component_metrics[key] = list(values)
            
            performance_metrics["component_metrics"] = component_metrics
        
        return JSONResponse(content=performance_metrics)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting health metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== ANOMALY DETECTION ENDPOINTS =====

@router.get("/anomalies")
async def get_anomalies(
    component_id: Optional[str] = None,
    severity: Optional[AlertSeverity] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get detected anomalies with optional filtering"""
    try:
        anomalies = predictive_maintenance_service.anomalies
        
        # Apply filters
        filtered_anomalies = {}
        for anomaly_id, anomaly in anomalies.items():
            if component_id and anomaly.component_id != component_id:
                continue
            if severity and anomaly.severity != severity:
                continue
            if start_time and anomaly.detected_at < start_time:
                continue
            if end_time and anomaly.detected_at > end_time:
                continue
            filtered_anomalies[anomaly_id] = anomaly
        
        # Convert to API models
        result = []
        for anomaly_id, anomaly in filtered_anomalies.items():
            result.append(AnomalyModel(
                anomaly_id=anomaly.anomaly_id,
                component_id=anomaly.component_id,
                anomaly_type=anomaly.anomaly_type,
                description=anomaly.description,
                severity=anomaly.severity,
                detected_at=anomaly.detected_at,
                metric_values=anomaly.metric_values,
                confidence_score=anomaly.confidence_score,
                predicted_impact=anomaly.predicted_impact,
                recommended_actions=anomaly.recommended_actions
            ))
        
        return JSONResponse(content={"anomalies": [anomaly.dict() for anomaly in result]})
        
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/anomalies/{anomaly_id}")
async def get_anomaly_details(
    anomaly_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get detailed information about a specific anomaly"""
    try:
        if anomaly_id not in predictive_maintenance_service.anomalies:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        
        anomaly = predictive_maintenance_service.anomalies[anomaly_id]
        
        anomaly_model = AnomalyModel(
            anomaly_id=anomaly.anomaly_id,
            component_id=anomaly.component_id,
            anomaly_type=anomaly.anomaly_type,
            description=anomaly.description,
            severity=anomaly.severity,
            detected_at=anomaly.detected_at,
            metric_values=anomaly.metric_values,
            confidence_score=anomaly.confidence_score,
            predicted_impact=anomaly.predicted_impact,
            recommended_actions=anomaly.recommended_actions
        )
        
        return JSONResponse(content=anomaly_model.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting anomaly details: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== MAINTENANCE SCHEDULING ENDPOINTS =====

@router.post("/maintenance/schedule")
async def schedule_maintenance(
    request: MaintenanceScheduleRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Schedule a maintenance task"""
    try:
        # Create maintenance task
        task = MaintenanceTask(
            task_id=f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request.component_id}",
            component_id=request.component_id,
            maintenance_type=request.maintenance_type,
            title=request.title,
            description=request.description,
            priority=request.priority,
            estimated_duration=timedelta(minutes=request.estimated_duration),
            scheduled_time=request.preferred_time or datetime.utcnow() + timedelta(hours=1),
            status=MaintenanceStatus.SCHEDULED,
            dependencies=request.dependencies,
            resources_required=request.resources_required
        )
        
        success = await predictive_maintenance_service.schedule_maintenance(task)
        
        if success:
            logger.info(f"Maintenance task {task.task_id} scheduled by {user.get('username', 'unknown')}")
            return JSONResponse(
                content={"message": "Maintenance task scheduled successfully", "task_id": task.task_id},
                status_code=201
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to schedule maintenance task")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling maintenance: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/maintenance/schedule")
async def get_maintenance_schedule(
    component_id: Optional[str] = None,
    maintenance_type: Optional[MaintenanceType] = None,
    status: Optional[MaintenanceStatus] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get maintenance schedule with optional filtering"""
    try:
        schedule = await predictive_maintenance_service.get_maintenance_schedule()
        
        # Apply filters
        filtered_schedule = []
        for task_dict in schedule:
            if component_id and task_dict.get('component_id') != component_id:
                continue
            if maintenance_type and task_dict.get('maintenance_type') != maintenance_type.value:
                continue
            if status and task_dict.get('status') != status.value:
                continue
            filtered_schedule.append(task_dict)
        
        return JSONResponse(content={"schedule": filtered_schedule})
        
    except Exception as e:
        logger.error(f"Error getting maintenance schedule: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/maintenance/tasks/{task_id}")
async def get_maintenance_task(
    task_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get details of a specific maintenance task"""
    try:
        if task_id not in predictive_maintenance_service.maintenance_tasks:
            raise HTTPException(status_code=404, detail="Maintenance task not found")
        
        task = predictive_maintenance_service.maintenance_tasks[task_id]
        
        task_model = MaintenanceTaskModel(
            task_id=task.task_id,
            component_id=task.component_id,
            maintenance_type=task.maintenance_type,
            title=task.title,
            description=task.description,
            priority=task.priority,
            estimated_duration=int(task.estimated_duration.total_seconds() / 60),
            scheduled_time=task.scheduled_time,
            status=task.status,
            assigned_to=task.assigned_to,
            dependencies=task.dependencies,
            resources_required=task.resources_required,
            metadata=task.metadata
        )
        
        return JSONResponse(content=task_model.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting maintenance task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.put("/maintenance/tasks/{task_id}/status")
async def update_maintenance_task_status(
    task_id: str,
    status: MaintenanceStatus,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Update maintenance task status"""
    try:
        if task_id not in predictive_maintenance_service.maintenance_tasks:
            raise HTTPException(status_code=404, detail="Maintenance task not found")
        
        task = predictive_maintenance_service.maintenance_tasks[task_id]
        task.status = status
        
        logger.info(f"Maintenance task {task_id} status updated to {status.value} by {user.get('username', 'unknown')}")
        
        return JSONResponse(content={"message": "Maintenance task status updated successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating maintenance task status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== FAILURE PREDICTION ENDPOINTS =====

@router.get("/predictions")
async def get_failure_predictions(
    component_id: Optional[str] = None,
    confidence_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get failure predictions with optional filtering"""
    try:
        predictions = predictive_maintenance_service.failure_predictions
        
        # Apply filters
        filtered_predictions = {}
        for pred_id, prediction in predictions.items():
            if component_id and prediction.component_id != component_id:
                continue
            if confidence_threshold and prediction.confidence < confidence_threshold:
                continue
            filtered_predictions[pred_id] = prediction
        
        # Convert to API models
        result = []
        for pred_id, prediction in filtered_predictions.items():
            result.append(FailurePredictionModel(
                prediction_id=prediction.prediction_id,
                component_id=prediction.component_id,
                predicted_failure_time=prediction.predicted_failure_time,
                confidence=prediction.confidence,
                failure_type=prediction.failure_type,
                contributing_factors=prediction.contributing_factors,
                recommended_actions=prediction.recommended_actions,
                created_at=prediction.created_at
            ))
        
        return JSONResponse(content={"predictions": [pred.dict() for pred in result]})
        
    except Exception as e:
        logger.error(f"Error getting failure predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/predictions/{prediction_id}")
async def get_prediction_details(
    prediction_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get detailed information about a specific failure prediction"""
    try:
        if prediction_id not in predictive_maintenance_service.failure_predictions:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        prediction = predictive_maintenance_service.failure_predictions[prediction_id]
        
        prediction_model = FailurePredictionModel(
            prediction_id=prediction.prediction_id,
            component_id=prediction.component_id,
            predicted_failure_time=prediction.predicted_failure_time,
            confidence=prediction.confidence,
            failure_type=prediction.failure_type,
            contributing_factors=prediction.contributing_factors,
            recommended_actions=prediction.recommended_actions,
            created_at=prediction.created_at
        )
        
        return JSONResponse(content=prediction_model.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction details: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== ALERT MANAGEMENT ENDPOINTS =====

@router.get("/alerts")
async def get_alerts(
    component_id: Optional[str] = None,
    severity: Optional[AlertSeverity] = None,
    active_only: bool = Query(True),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get alerts with optional filtering"""
    try:
        alerts = predictive_maintenance_service.active_alerts if active_only else predictive_maintenance_service.alert_history
        
        # Apply filters
        filtered_alerts = {}
        if isinstance(alerts, dict):
            for alert_id, alert in alerts.items():
                if component_id and alert.get('component_id') != component_id:
                    continue
                if severity and alert.get('severity') != severity.value:
                    continue
                filtered_alerts[alert_id] = alert
        else:
            # Handle deque for alert history
            filtered_alerts = []
            for alert in alerts:
                if component_id and alert.get('component_id') != component_id:
                    continue
                if severity and alert.get('severity') != severity.value:
                    continue
                filtered_alerts.append(alert)
        
        return JSONResponse(content={"alerts": filtered_alerts})
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/alerts/configure")
async def configure_alert(
    config: AlertConfigModel,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Configure alert thresholds for a component metric"""
    try:
        # Store alert configuration (this would typically be persisted)
        alert_config = {
            "component_id": config.component_id,
            "metric_name": config.metric_name,
            "threshold_value": config.threshold_value,
            "comparison_operator": config.comparison_operator,
            "alert_severity": config.alert_severity,
            "cooldown_period": config.cooldown_period,
            "created_by": user.get("username", "unknown"),
            "created_at": datetime.utcnow()
        }
        
        # This would typically be stored in a database
        logger.info(f"Alert configuration created for {config.component_id}:{config.metric_name}")
        
        return JSONResponse(
            content={"message": "Alert configuration created successfully", "config": alert_config},
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error configuring alert: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== ANALYTICS ENDPOINTS =====

@router.get("/analytics/dashboard")
async def get_analytics_dashboard(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get analytics dashboard data"""
    try:
        # Get system health
        system_health = await predictive_maintenance_service.get_system_health()
        
        # Get performance metrics
        performance_metrics = await predictive_maintenance_service.get_performance_metrics()
        
        # Get recent anomalies
        recent_anomalies = []
        for anomaly in list(predictive_maintenance_service.anomalies.values())[-10:]:
            recent_anomalies.append({
                "anomaly_id": anomaly.anomaly_id,
                "component_id": anomaly.component_id,
                "severity": anomaly.severity.value,
                "detected_at": anomaly.detected_at.isoformat(),
                "description": anomaly.description
            })
        
        # Get maintenance schedule summary
        maintenance_schedule = await predictive_maintenance_service.get_maintenance_schedule()
        maintenance_summary = {
            "total_tasks": len(maintenance_schedule),
            "scheduled_tasks": len([t for t in maintenance_schedule if t.get('status') == 'scheduled']),
            "in_progress_tasks": len([t for t in maintenance_schedule if t.get('status') == 'in_progress']),
            "completed_tasks": len([t for t in maintenance_schedule if t.get('status') == 'completed'])
        }
        
        dashboard_data = {
            "system_health": system_health,
            "performance_metrics": performance_metrics,
            "recent_anomalies": recent_anomalies,
            "maintenance_summary": maintenance_summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting analytics dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/analytics/trends")
async def get_trends(
    component_id: Optional[str] = None,
    metric_name: Optional[str] = None,
    time_range: int = Query(24, description="Time range in hours"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get trend analysis for metrics"""
    try:
        trends = {}
        
        # Get metrics history
        for key, values in predictive_maintenance_service.metrics_history.items():
            if component_id and not key.startswith(component_id):
                continue
            if metric_name and metric_name not in key:
                continue
            
            # Convert to time series data
            time_series = []
            for metric in list(values)[-100:]:  # Last 100 data points
                time_series.append({
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.value
                })
            
            trends[key] = {
                "data": time_series,
                "trend": "stable",  # Would implement actual trend analysis
                "correlation": 0.0  # Would implement correlation analysis
            }
        
        return JSONResponse(content={"trends": trends})
        
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== SYSTEM MANAGEMENT ENDPOINTS =====

@router.post("/system/initialize")
async def initialize_system(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Initialize the predictive maintenance system"""
    try:
        await predictive_maintenance_service.initialize()
        
        logger.info(f"Predictive maintenance system initialized by {user.get('username', 'unknown')}")
        
        return JSONResponse(content={"message": "System initialized successfully"})
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/system/status")
async def get_system_status(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get system status and configuration"""
    try:
        status = {
            "service_status": "running",
            "configuration": predictive_maintenance_service.config,
            "metrics": predictive_maintenance_service.metrics,
            "registered_components": len(predictive_maintenance_service.components),
            "active_alerts": len(predictive_maintenance_service.active_alerts),
            "maintenance_tasks": len(predictive_maintenance_service.maintenance_tasks),
            "failure_predictions": len(predictive_maintenance_service.failure_predictions),
            "background_tasks": len(predictive_maintenance_service.background_tasks)
        }
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/system/config")
async def update_system_config(
    config: Dict[str, Any],
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Update system configuration"""
    try:
        # Validate and update configuration
        valid_keys = set(predictive_maintenance_service.config.keys())
        for key, value in config.items():
            if key in valid_keys:
                predictive_maintenance_service.config[key] = value
        
        logger.info(f"System configuration updated by {user.get('username', 'unknown')}")
        
        return JSONResponse(content={"message": "Configuration updated successfully"})
        
    except Exception as e:
        logger.error(f"Error updating system config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 