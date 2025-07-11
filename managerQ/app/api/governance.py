from fastapi import APIRouter, HTTPException, Depends, Body, Query
from typing import Optional, List, Any, Dict, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from managerQ.app.core.workflow_manager import workflow_manager
from managerQ.app.models import Workflow
from shared.q_auth_parser import get_current_user_is_admin
from managerQ.app.core.ai_governance_service import (
    ai_governance_service,
    PolicyType,
    PolicyStatus,
    ViolationSeverity,
    ComplianceStatus,
    AuditEventType
)
import yaml
import os
import json

router = APIRouter()

CONSTITUTION_PATH = "governance/platform_constitution.yaml"

@router.get("/constitution", response_model=Any)
async def get_constitution(is_admin: bool = Depends(get_current_user_is_admin)):
    """Retrieves the current platform constitution."""
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized.")
    try:
        with open(CONSTITUTION_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Constitution file not found.")

@router.post("/constitution", status_code=204)
async def update_constitution(
    constitution_data: Dict[str, Any] = Body(...),
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Updates the platform constitution file."""
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized.")
    try:
        with open(CONSTITUTION_PATH, 'w') as f:
            yaml.dump(constitution_data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write constitution file: {e}")


@router.get("/vetoed-workflows", response_model=List[Workflow])
async def get_vetoed_workflows(is_admin: bool = Depends(get_current_user_is_admin)):
    """
    Retrieves a list of all workflows that have been halted with a VETOED status.
    This endpoint is protected and only accessible by admins.
    """
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to view vetoed workflows.")
    
    # This relies on a new method in the WorkflowManager
    return workflow_manager.get_workflows_by_status("VETOED")


@router.get("/vetoed-workflows/{workflow_id}", response_model=Workflow)
async def get_vetoed_workflow_details(workflow_id: str, is_admin: bool = Depends(get_current_user_is_admin)):
    """
    Retrieves the full details for a single vetoed workflow.
    """
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to view vetoed workflows.")

    workflow = workflow_manager.get_workflow(workflow_id)
    if not workflow or workflow.status.value != "VETOED":
        raise HTTPException(status_code=404, detail="Vetoed workflow not found.")
    
    return workflow 

@router.get("/events/history")
async def get_event_history(
    timestamp: float,
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """
    Retrieves the persisted event tick closest to a given timestamp.
    """
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized.")
    
    try:
        # This is a conceptual implementation. A real Ignite query would be more complex,
        # likely using a continuous query or a scan query with a custom filter.
        # For now, we simulate fetching a single record.
        # This assumes the ObservabilityManager's Ignite cache is accessible here.
        # A better design would be a dedicated service.
        
        # from managerQ.app.core.observability_manager import observability_manager
        # record = observability_manager._event_history_cache.get(timestamp, with_-binary=True)
        
        # MOCK IMPLEMENTATION:
        mock_record = {
            "type": "TICK",
            "payload": [
                {"event_type": "NODE_CREATED", "data": {"id": "agent_abc", "label": "Agent-DevOps", "type": "agent"}}
            ],
            "timestamp": timestamp
        }
        return json.loads(json.dumps(mock_record))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch event history: {e}")

# ===== NEW AI GOVERNANCE ENDPOINTS =====

# Request Models
class PolicyCreateRequest(BaseModel):
    """Request for creating a new policy"""
    name: str = Field(..., description="Policy name")
    description: str = Field(..., description="Policy description") 
    policy_type: str = Field(..., description="Policy type")
    rules: List[Dict[str, Any]] = Field(..., description="Policy rules")
    enforcement_mode: str = Field("blocking", description="Enforcement mode")
    scope: Optional[Dict[str, Any]] = Field(None, description="Policy scope")

class ComplianceReportRequest(BaseModel):
    """Request for generating compliance report"""
    assessment_type: str = Field(..., description="Assessment type")
    scope: Dict[str, Any] = Field(..., description="Assessment scope")
    framework: Optional[str] = Field(None, description="Regulatory framework")

class RiskAssessmentRequest(BaseModel):
    """Request for risk assessment"""
    target_id: str = Field(..., description="Target resource ID")
    target_type: str = Field(..., description="Target resource type")

@router.post("/policies", response_model=Dict[str, str])
async def create_policy(
    request: PolicyCreateRequest,
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Create a new governance policy"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        policy_id = await ai_governance_service.create_policy(
            name=request.name,
            description=request.description,
            policy_type=PolicyType(request.policy_type),
            rules=request.rules,
            enforcement_mode=request.enforcement_mode,
            scope=request.scope,
            created_by="admin"
        )
        
        return {"policy_id": policy_id, "status": "created"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/policies", response_model=List[Dict[str, Any]])
async def list_policies(
    policy_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """List governance policies"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        policies = []
        for policy in ai_governance_service.policies.values():
            # Apply filters
            if policy_type and policy.policy_type.value != policy_type:
                continue
            if status and policy.status.value != status:
                continue
            
            policy_dict = {
                "policy_id": policy.policy_id,
                "name": policy.name,
                "description": policy.description,
                "policy_type": policy.policy_type.value,
                "status": policy.status.value,
                "enforcement_mode": policy.enforcement_mode,
                "created_at": policy.created_at.isoformat(),
                "updated_at": policy.updated_at.isoformat(),
                "version": policy.version
            }
            policies.append(policy_dict)
        
        return policies
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/policies/{policy_id}/activate")
async def activate_policy(
    policy_id: str,
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Activate a policy"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        success = await ai_governance_service.activate_policy(policy_id, "admin")
        if not success:
            raise HTTPException(status_code=404, detail="Policy not found")
        
        return {"status": "activated"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/policies/{policy_id}/deactivate")
async def deactivate_policy(
    policy_id: str,
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Deactivate a policy"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        success = await ai_governance_service.deactivate_policy(policy_id, "admin")
        if not success:
            raise HTTPException(status_code=404, detail="Policy not found")
        
        return {"status": "deactivated"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate")
async def evaluate_policies(
    resource_type: str,
    resource_id: str,
    context: Dict[str, Any] = Body(...),
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Evaluate policies for a resource"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        result = await ai_governance_service.evaluate_policies(
            resource_type=resource_type,
            resource_id=resource_id,
            context=context,
            user_id="admin"
        )
        
        # Convert violations to serializable format
        if "violations" in result:
            violations = []
            for violation in result["violations"]:
                violations.append({
                    "violation_id": violation.violation_id,
                    "policy_id": violation.policy_id,
                    "severity": violation.severity.value,
                    "description": violation.description,
                    "detected_at": violation.detected_at.isoformat()
                })
            result["violations"] = violations
        
        if "warnings" in result:
            warnings = []
            for warning in result["warnings"]:
                warnings.append({
                    "violation_id": warning.violation_id,
                    "policy_id": warning.policy_id,
                    "severity": warning.severity.value,
                    "description": warning.description,
                    "detected_at": warning.detected_at.isoformat()
                })
            result["warnings"] = warnings
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/violations", response_model=List[Dict[str, Any]])
async def list_violations(
    severity: Optional[str] = Query(None),
    resolved: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """List policy violations"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        violations = []
        count = 0
        
        for violation in ai_governance_service.policy_violations.values():
            # Apply filters
            if severity and violation.severity.value != severity:
                continue
            if resolved is not None:
                is_resolved = violation.resolved_at is not None
                if resolved != is_resolved:
                    continue
            
            violation_dict = {
                "violation_id": violation.violation_id,
                "policy_id": violation.policy_id,
                "resource_id": violation.resource_id,
                "resource_type": violation.resource_type,
                "severity": violation.severity.value,
                "description": violation.description,
                "detected_at": violation.detected_at.isoformat(),
                "resolved_at": violation.resolved_at.isoformat() if violation.resolved_at else None,
                "resolution_action": violation.resolution_action
            }
            violations.append(violation_dict)
            
            count += 1
            if count >= limit:
                break
        
        return violations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compliance/report", response_model=Dict[str, str])
async def generate_compliance_report(
    request: ComplianceReportRequest,
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Generate compliance assessment report"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        report_id = await ai_governance_service.generate_compliance_report(
            assessment_type=request.assessment_type,
            scope=request.scope,
            framework=request.framework
        )
        
        return {"report_id": report_id, "status": "generated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/compliance/reports/{report_id}", response_model=Dict[str, Any])
async def get_compliance_report(
    report_id: str,
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Get compliance report"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        report = ai_governance_service.compliance_reports.get(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "report_id": report.report_id,
            "assessment_type": report.assessment_type,
            "scope": report.scope,
            "compliance_status": report.compliance_status.value,
            "compliance_score": report.compliance_score,
            "findings": report.findings,
            "recommendations": report.recommendations,
            "risk_assessment": report.risk_assessment,
            "generated_at": report.generated_at.isoformat(),
            "next_assessment_due": report.next_assessment_due.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk/assess", response_model=Dict[str, str])
async def conduct_risk_assessment(
    request: RiskAssessmentRequest,
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Conduct risk assessment"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        assessment_id = await ai_governance_service.conduct_risk_assessment(
            target_id=request.target_id,
            target_type=request.target_type,
            assessed_by="admin"
        )
        
        return {"assessment_id": assessment_id, "status": "completed"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/assessments/{assessment_id}", response_model=Dict[str, Any])
async def get_risk_assessment(
    assessment_id: str,
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Get risk assessment"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        assessment = ai_governance_service.risk_assessments.get(assessment_id)
        if not assessment:
            raise HTTPException(status_code=404, detail="Assessment not found")
        
        return {
            "assessment_id": assessment.assessment_id,
            "target_id": assessment.target_id,
            "target_type": assessment.target_type,
            "risk_categories": assessment.risk_categories,
            "overall_risk_score": assessment.overall_risk_score,
            "risk_factors": assessment.risk_factors,
            "mitigation_strategies": assessment.mitigation_strategies,
            "assessed_at": assessment.assessed_at.isoformat(),
            "assessed_by": assessment.assessed_by,
            "next_assessment_due": assessment.next_assessment_due.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/audit/events", response_model=List[Dict[str, Any]])
async def search_audit_events(
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Search audit events"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        # Parse datetime strings
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None
        event_type_enum = AuditEventType(event_type) if event_type else None
        
        events = await ai_governance_service.search_audit_events(
            start_time=start_dt,
            end_time=end_dt,
            event_type=event_type_enum,
            user_id=user_id,
            resource_id=resource_id,
            action=action,
            limit=limit
        )
        
        # Convert to serializable format
        events_list = []
        for event in events:
            events_list.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "resource_id": event.resource_id,
                "resource_type": event.resource_type,
                "action": event.action,
                "outcome": event.outcome,
                "details": event.details,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "session_id": event.session_id
            })
        
        return events_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=Dict[str, Any])
async def get_governance_metrics(
    is_admin: bool = Depends(get_current_user_is_admin)
):
    """Get governance metrics"""
    
    if not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        return ai_governance_service.governance_metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=Dict[str, Any])
async def governance_health_check():
    """Health check for governance service"""
    
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "active_policies": ai_governance_service.governance_metrics["active_policies"],
            "total_policies": ai_governance_service.governance_metrics["total_policies"],
            "service_version": "1.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        } 