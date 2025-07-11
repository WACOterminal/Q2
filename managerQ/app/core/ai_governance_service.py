"""
AI Governance & Compliance Service

This service provides comprehensive AI governance and compliance capabilities:
- Automated policy enforcement with real-time monitoring
- Comprehensive audit trails and compliance reporting
- Model governance and lifecycle management
- Data governance and privacy compliance
- Ethical AI guidelines enforcement
- Risk assessment and mitigation
- Regulatory compliance (GDPR, AI Act, etc.)
- Automated remediation and incident response
- Continuous compliance monitoring
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
from pathlib import Path
import pickle
from collections import defaultdict, deque
import re

# Data validation and schema libraries
from pydantic import BaseModel, ValidationError
import jsonschema
from jsonschema import validate

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.q_vectorstore_client.client import VectorStoreClient
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.vault_client import VaultClient

# Import other services for integration
from .model_registry_service import ModelRegistryService
from .data_versioning_service import DataVersioningService
from .explainable_ai_service import explainable_ai_service
from .feature_store_service import FeatureStoreService

logger = logging.getLogger(__name__)

class PolicyType(Enum):
    """Types of governance policies"""
    DATA_PRIVACY = "data_privacy"
    MODEL_FAIRNESS = "model_fairness"
    ETHICAL_AI = "ethical_ai"
    SECURITY = "security"
    QUALITY = "quality"
    EXPLAINABILITY = "explainability"
    COMPLIANCE = "compliance"
    RESOURCE_USAGE = "resource_usage"
    OPERATIONAL = "operational"

class PolicyStatus(Enum):
    """Policy status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    DEPRECATED = "deprecated"

class ViolationSeverity(Enum):
    """Violation severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStatus(Enum):
    """Compliance status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REMEDIATION_REQUIRED = "remediation_required"

class AuditEventType(Enum):
    """Types of audit events"""
    MODEL_DEPLOYMENT = "model_deployment"
    DATA_ACCESS = "data_access"
    POLICY_VIOLATION = "policy_violation"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    COMPLIANCE_CHECK = "compliance_check"
    RISK_ASSESSMENT = "risk_assessment"
    REMEDIATION_ACTION = "remediation_action"

@dataclass
class GovernancePolicy:
    """Governance policy definition"""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    status: PolicyStatus
    rules: List[Dict[str, Any]]
    enforcement_mode: str  # "blocking", "warning", "logging"
    scope: Dict[str, Any]  # What this policy applies to
    created_by: str
    created_at: datetime
    updated_at: datetime
    version: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PolicyViolation:
    """Policy violation record"""
    violation_id: str
    policy_id: str
    resource_id: str
    resource_type: str
    severity: ViolationSeverity
    description: str
    context: Dict[str, Any]
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_action: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AuditEvent:
    """Audit event record"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    resource_id: Optional[str]
    resource_type: Optional[str]
    action: str
    outcome: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    report_id: str
    assessment_type: str
    scope: Dict[str, Any]
    compliance_status: ComplianceStatus
    compliance_score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]
    generated_at: datetime
    next_assessment_due: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RiskAssessment:
    """Risk assessment for AI systems"""
    assessment_id: str
    target_id: str
    target_type: str
    risk_categories: Dict[str, float]  # category -> risk score (0-1)
    overall_risk_score: float
    risk_factors: List[Dict[str, Any]]
    mitigation_strategies: List[str]
    assessed_at: datetime
    assessed_by: str
    next_assessment_due: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class AIGovernanceService:
    """
    Comprehensive AI Governance and Compliance Service
    """
    
    def __init__(self, 
                 storage_path: str = "governance",
                 audit_retention_days: int = 2555):  # 7 years
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Policy management
        self.policies: Dict[str, GovernancePolicy] = {}
        self.policy_violations: Dict[str, PolicyViolation] = {}
        self.active_policies_by_type: Dict[PolicyType, List[str]] = defaultdict(list)
        
        # Audit management
        self.audit_events: deque = deque(maxlen=1000000)  # Keep recent events in memory
        self.audit_retention_days = audit_retention_days
        
        # Compliance tracking
        self.compliance_reports: Dict[str, ComplianceReport] = {}
        self.risk_assessments: Dict[str, RiskAssessment] = {}
        
        # Service integrations
        self.model_registry_service: Optional[ModelRegistryService] = None
        self.data_versioning_service: Optional[DataVersioningService] = None
        self.feature_store_service: Optional[FeatureStoreService] = None
        self.vault_client = VaultClient()
        self.kg_client: Optional[KnowledgeGraphClient] = None
        
        # Policy engine configuration
        self.policy_engine_config = {
            "evaluation_interval": 300,  # 5 minutes
            "batch_size": 100,
            "max_concurrent_evaluations": 10,
            "enable_real_time_monitoring": True,
            "enable_automated_remediation": True
        }
        
        # Built-in policy templates
        self.policy_templates = self._load_policy_templates()
        
        # Regulatory frameworks
        self.regulatory_frameworks = {
            "GDPR": self._load_gdpr_requirements(),
            "CCPA": self._load_ccpa_requirements(),
            "AI_ACT": self._load_ai_act_requirements(),
            "SOX": self._load_sox_requirements(),
            "HIPAA": self._load_hipaa_requirements()
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Performance metrics
        self.governance_metrics = {
            "total_policies": 0,
            "active_policies": 0,
            "policy_violations": 0,
            "resolved_violations": 0,
            "compliance_score": 0.0,
            "audit_events": 0,
            "risk_assessments": 0,
            "automated_remediations": 0
        }
        
        logger.info("AI Governance Service initialized")
    
    async def initialize(self):
        """Initialize the AI governance service"""
        logger.info("Initializing AI Governance Service")
        
        # Initialize service integrations
        await self._initialize_service_integrations()
        
        # Load existing policies and audit data
        await self._load_governance_state()
        
        # Initialize default policies
        await self._initialize_default_policies()
        
        # Start background monitoring
        await self._start_background_monitoring()
        
        # Subscribe to system events
        await self._subscribe_to_system_events()
        
        logger.info("AI Governance Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the AI governance service"""
        logger.info("Shutting down AI Governance Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Save governance state
        await self._save_governance_state()
        
        logger.info("AI Governance Service shutdown complete")
    
    # ===== POLICY MANAGEMENT =====
    
    async def create_policy(
        self,
        name: str,
        description: str,
        policy_type: PolicyType,
        rules: List[Dict[str, Any]],
        enforcement_mode: str = "blocking",
        scope: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new governance policy"""
        
        policy_id = f"policy_{uuid.uuid4().hex[:12]}"
        
        # Validate policy rules
        await self._validate_policy_rules(rules, policy_type)
        
        # Create policy
        policy = GovernancePolicy(
            policy_id=policy_id,
            name=name,
            description=description,
            policy_type=policy_type,
            status=PolicyStatus.ACTIVE,
            rules=rules,
            enforcement_mode=enforcement_mode,
            scope=scope or {},
            created_by=created_by,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version="1.0"
        )
        
        # Store policy
        self.policies[policy_id] = policy
        self.active_policies_by_type[policy_type].append(policy_id)
        
        # Update metrics
        self.governance_metrics["total_policies"] += 1
        self.governance_metrics["active_policies"] += 1
        
        # Log audit event
        await self._log_audit_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="policy_created",
            resource_id=policy_id,
            resource_type="policy",
            details={"policy_name": name, "policy_type": policy_type.value},
            user_id=created_by
        )
        
        logger.info(f"Created governance policy: {policy_id}")
        return policy_id
    
    async def update_policy(
        self,
        policy_id: str,
        updates: Dict[str, Any],
        updated_by: str = "system"
    ) -> bool:
        """Update an existing policy"""
        
        if policy_id not in self.policies:
            return False
        
        policy = self.policies[policy_id]
        
        # Validate updates
        if "rules" in updates:
            await self._validate_policy_rules(updates["rules"], policy.policy_type)
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        
        policy.updated_at = datetime.utcnow()
        
        # Increment version
        version_parts = policy.version.split(".")
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        policy.version = ".".join(version_parts)
        
        # Log audit event
        await self._log_audit_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="policy_updated",
            resource_id=policy_id,
            resource_type="policy",
            details={"updates": list(updates.keys())},
            user_id=updated_by
        )
        
        logger.info(f"Updated governance policy: {policy_id}")
        return True
    
    async def activate_policy(self, policy_id: str, activated_by: str = "system") -> bool:
        """Activate a policy"""
        
        if policy_id not in self.policies:
            return False
        
        policy = self.policies[policy_id]
        
        if policy.status != PolicyStatus.ACTIVE:
            policy.status = PolicyStatus.ACTIVE
            policy.updated_at = datetime.utcnow()
            
            if policy_id not in self.active_policies_by_type[policy.policy_type]:
                self.active_policies_by_type[policy.policy_type].append(policy_id)
            
            self.governance_metrics["active_policies"] += 1
            
            await self._log_audit_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                action="policy_activated",
                resource_id=policy_id,
                resource_type="policy",
                details={"policy_name": policy.name},
                user_id=activated_by
            )
            
            logger.info(f"Activated governance policy: {policy_id}")
        
        return True
    
    async def deactivate_policy(self, policy_id: str, deactivated_by: str = "system") -> bool:
        """Deactivate a policy"""
        
        if policy_id not in self.policies:
            return False
        
        policy = self.policies[policy_id]
        
        if policy.status == PolicyStatus.ACTIVE:
            policy.status = PolicyStatus.INACTIVE
            policy.updated_at = datetime.utcnow()
            
            if policy_id in self.active_policies_by_type[policy.policy_type]:
                self.active_policies_by_type[policy.policy_type].remove(policy_id)
            
            self.governance_metrics["active_policies"] -= 1
            
            await self._log_audit_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                action="policy_deactivated",
                resource_id=policy_id,
                resource_type="policy",
                details={"policy_name": policy.name},
                user_id=deactivated_by
            )
            
            logger.info(f"Deactivated governance policy: {policy_id}")
        
        return True
    
    # ===== POLICY ENFORCEMENT =====
    
    async def evaluate_policies(
        self,
        resource_type: str,
        resource_id: str,
        context: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate all applicable policies for a resource"""
        
        evaluation_result = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "enforcement_actions": []
        }
        
        try:
            # Get applicable policies
            applicable_policies = await self._get_applicable_policies(resource_type, context)
            
            for policy_id in applicable_policies:
                policy = self.policies[policy_id]
                
                # Evaluate each rule in the policy
                for rule in policy.rules:
                    violation = await self._evaluate_rule(policy, rule, resource_type, resource_id, context)
                    
                    if violation:
                        # Record violation
                        violation_record = PolicyViolation(
                            violation_id=f"violation_{uuid.uuid4().hex[:12]}",
                            policy_id=policy_id,
                            resource_id=resource_id,
                            resource_type=resource_type,
                            severity=ViolationSeverity(violation.get("severity", "medium")),
                            description=violation["description"],
                            context=context,
                            detected_at=datetime.utcnow()
                        )
                        
                        self.policy_violations[violation_record.violation_id] = violation_record
                        self.governance_metrics["policy_violations"] += 1
                        
                        # Log audit event
                        await self._log_audit_event(
                            event_type=AuditEventType.POLICY_VIOLATION,
                            action="policy_violation_detected",
                            resource_id=resource_id,
                            resource_type=resource_type,
                            details={
                                "policy_id": policy_id,
                                "violation_id": violation_record.violation_id,
                                "severity": violation_record.severity.value,
                                "description": violation_record.description
                            },
                            user_id=user_id
                        )
                        
                        # Determine enforcement action
                        if policy.enforcement_mode == "blocking":
                            evaluation_result["compliant"] = False
                            evaluation_result["violations"].append(violation_record)
                            
                            # Apply enforcement action
                            enforcement_action = await self._apply_enforcement_action(policy, violation_record, context)
                            if enforcement_action:
                                evaluation_result["enforcement_actions"].append(enforcement_action)
                        
                        elif policy.enforcement_mode == "warning":
                            evaluation_result["warnings"].append(violation_record)
                        
                        # Always log violations regardless of enforcement mode
                        logger.warning(f"Policy violation detected: {violation_record.violation_id}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating policies: {e}", exc_info=True)
            return {
                "compliant": False,
                "violations": [],
                "warnings": [],
                "enforcement_actions": [],
                "error": str(e)
            }
    
    async def _evaluate_rule(
        self,
        policy: GovernancePolicy,
        rule: Dict[str, Any],
        resource_type: str,
        resource_id: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single policy rule"""
        
        try:
            rule_type = rule.get("type")
            
            if rule_type == "data_privacy":
                return await self._evaluate_data_privacy_rule(rule, resource_type, resource_id, context)
            elif rule_type == "model_fairness":
                return await self._evaluate_fairness_rule(rule, resource_type, resource_id, context)
            elif rule_type == "explainability":
                return await self._evaluate_explainability_rule(rule, resource_type, resource_id, context)
            elif rule_type == "security":
                return await self._evaluate_security_rule(rule, resource_type, resource_id, context)
            elif rule_type == "quality":
                return await self._evaluate_quality_rule(rule, resource_type, resource_id, context)
            elif rule_type == "resource_usage":
                return await self._evaluate_resource_usage_rule(rule, resource_type, resource_id, context)
            elif rule_type == "custom":
                return await self._evaluate_custom_rule(rule, resource_type, resource_id, context)
            else:
                logger.warning(f"Unknown rule type: {rule_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating rule: {e}", exc_info=True)
            return {
                "description": f"Error evaluating rule: {str(e)}",
                "severity": "medium"
            }
    
    async def _evaluate_data_privacy_rule(self, rule: Dict[str, Any], resource_type: str, resource_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate data privacy rules"""
        
        if resource_type not in ["dataset", "model", "feature_group"]:
            return None
        
        # Check for PII handling
        if rule.get("check_pii", False):
            pii_detected = context.get("contains_pii", False)
            if pii_detected and not context.get("pii_anonymized", False):
                return {
                    "description": "PII detected in data without proper anonymization",
                    "severity": "high"
                }
        
        # Check data retention policies
        if rule.get("max_retention_days"):
            data_age = context.get("data_age_days", 0)
            if data_age > rule["max_retention_days"]:
                return {
                    "description": f"Data retention period exceeded: {data_age} days > {rule['max_retention_days']} days",
                    "severity": "medium"
                }
        
        # Check consent requirements
        if rule.get("require_consent", False):
            consent_status = context.get("consent_status", "unknown")
            if consent_status != "granted":
                return {
                    "description": "Data processing without proper consent",
                    "severity": "high"
                }
        
        return None
    
    async def _evaluate_fairness_rule(self, rule: Dict[str, Any], resource_type: str, resource_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate model fairness rules"""
        
        if resource_type != "model":
            return None
        
        # Check bias metrics
        max_bias = rule.get("max_demographic_parity", 0.1)
        demographic_parity = context.get("demographic_parity", 0.0)
        
        if demographic_parity > max_bias:
            return {
                "description": f"Model exceeds demographic parity threshold: {demographic_parity:.3f} > {max_bias:.3f}",
                "severity": "high"
            }
        
        # Check equalized odds
        max_eq_odds = rule.get("max_equalized_odds", 0.1)
        equalized_odds = context.get("equalized_odds", 0.0)
        
        if equalized_odds > max_eq_odds:
            return {
                "description": f"Model exceeds equalized odds threshold: {equalized_odds:.3f} > {max_eq_odds:.3f}",
                "severity": "high"
            }
        
        return None
    
    async def _evaluate_explainability_rule(self, rule: Dict[str, Any], resource_type: str, resource_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate explainability rules"""
        
        if resource_type != "model":
            return None
        
        # Check if explanation is required
        if rule.get("require_explanation", False):
            has_explanation = context.get("has_explanation", False)
            if not has_explanation:
                return {
                    "description": "Model deployment requires explainability but no explanation is available",
                    "severity": "high"
                }
            
            # Check explanation quality
            min_quality = rule.get("min_explanation_quality", 0.7)
            explanation_quality = context.get("explanation_quality", 0.0)
            if explanation_quality < min_quality:
                return {
                    "description": f"Explanation quality below threshold: {explanation_quality:.3f} < {min_quality:.3f}",
                    "severity": "medium"
                }
        
        return None
    
    async def _evaluate_security_rule(self, rule: Dict[str, Any], resource_type: str, resource_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate security rules"""
        
        # Check encryption requirements
        if rule.get("require_encryption", False):
            is_encrypted = context.get("is_encrypted", False)
            if not is_encrypted:
                return {
                    "description": "Resource requires encryption but is not encrypted",
                    "severity": "high"
                }
        
        # Check access controls
        if rule.get("require_access_control", False):
            has_access_control = context.get("has_access_control", False)
            if not has_access_control:
                return {
                    "description": "Resource requires access control but none is configured",
                    "severity": "high"
                }
        
        # Check vulnerability scanning
        if rule.get("require_vulnerability_scan", False):
            last_scan = context.get("last_vulnerability_scan")
            if not last_scan:
                return {
                    "description": "Resource requires vulnerability scanning but none has been performed",
                    "severity": "medium"
                }
            
            max_scan_age = rule.get("max_scan_age_days", 30)
            scan_age = (datetime.utcnow() - datetime.fromisoformat(last_scan)).days
            if scan_age > max_scan_age:
                return {
                    "description": f"Vulnerability scan is outdated: {scan_age} days > {max_scan_age} days",
                    "severity": "medium"
                }
        
        return None
    
    async def _evaluate_quality_rule(self, rule: Dict[str, Any], resource_type: str, resource_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate quality rules"""
        
        # Check minimum accuracy for models
        if resource_type == "model" and rule.get("min_accuracy"):
            accuracy = context.get("accuracy", 0.0)
            min_accuracy = rule["min_accuracy"]
            if accuracy < min_accuracy:
                return {
                    "description": f"Model accuracy below threshold: {accuracy:.3f} < {min_accuracy:.3f}",
                    "severity": "medium"
                }
        
        # Check data quality scores
        if resource_type in ["dataset", "feature_group"] and rule.get("min_data_quality"):
            data_quality = context.get("data_quality_score", 0.0)
            min_quality = rule["min_data_quality"]
            if data_quality < min_quality:
                return {
                    "description": f"Data quality below threshold: {data_quality:.3f} < {min_quality:.3f}",
                    "severity": "medium"
                }
        
        return None
    
    async def _evaluate_resource_usage_rule(self, rule: Dict[str, Any], resource_type: str, resource_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate resource usage rules"""
        
        # Check CPU usage limits
        if rule.get("max_cpu_usage"):
            cpu_usage = context.get("cpu_usage", 0.0)
            max_cpu = rule["max_cpu_usage"]
            if cpu_usage > max_cpu:
                return {
                    "description": f"CPU usage exceeds limit: {cpu_usage:.1f}% > {max_cpu:.1f}%",
                    "severity": "medium"
                }
        
        # Check memory usage limits
        if rule.get("max_memory_usage"):
            memory_usage = context.get("memory_usage", 0.0)
            max_memory = rule["max_memory_usage"]
            if memory_usage > max_memory:
                return {
                    "description": f"Memory usage exceeds limit: {memory_usage:.1f}% > {max_memory:.1f}%",
                    "severity": "medium"
                }
        
        # Check storage usage limits
        if rule.get("max_storage_gb"):
            storage_usage = context.get("storage_usage_gb", 0.0)
            max_storage = rule["max_storage_gb"]
            if storage_usage > max_storage:
                return {
                    "description": f"Storage usage exceeds limit: {storage_usage:.1f}GB > {max_storage:.1f}GB",
                    "severity": "low"
                }
        
        return None
    
    async def _evaluate_custom_rule(self, rule: Dict[str, Any], resource_type: str, resource_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate custom rules using expressions"""
        
        try:
            expression = rule.get("expression")
            if not expression:
                return None
            
            # Simple expression evaluation (in production, use a safe evaluator)
            # For security, only allow specific functions and variables
            allowed_context = {
                "resource_type": resource_type,
                "resource_id": resource_id,
                **context
            }
            
            # Evaluate expression safely
            result = await self._safe_evaluate_expression(expression, allowed_context)
            
            if result:
                return {
                    "description": rule.get("description", "Custom rule violation"),
                    "severity": rule.get("severity", "medium")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating custom rule: {e}")
            return {
                "description": f"Error in custom rule evaluation: {str(e)}",
                "severity": "low"
            }
    
    async def _safe_evaluate_expression(self, expression: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a custom rule expression"""
        
        # Simple implementation - in production, use a proper expression evaluator
        # This is a simplified version for demonstration
        
        try:
            # Only allow simple comparison operations
            if re.match(r'^[\w\.\s><!=\d\.\'"]+$', expression):
                # Replace context variables
                for key, value in context.items():
                    expression = expression.replace(key, str(value))
                
                # Basic evaluation (very limited for security)
                return eval(expression, {"__builtins__": {}})
            
            return False
            
        except Exception:
            return False
    
    # ===== AUDIT MANAGEMENT =====
    
    async def log_audit_event(
        self,
        event_type: AuditEventType,
        action: str,
        outcome: str = "success",
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Log an audit event"""
        
        event_id = f"audit_{uuid.uuid4().hex[:12]}"
        
        audit_event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            resource_id=resource_id,
            resource_type=resource_type,
            action=action,
            outcome=outcome,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id
        )
        
        # Store in memory
        self.audit_events.append(audit_event)
        
        # Store in persistent storage
        await self._persist_audit_event(audit_event)
        
        # Update metrics
        self.governance_metrics["audit_events"] += 1
        
        # Publish to event stream
        await self._publish_audit_event(audit_event)
        
        return event_id
    
    async def _log_audit_event(self, event_type: AuditEventType, action: str, **kwargs):
        """Internal method to log audit events"""
        await self.log_audit_event(event_type, action, **kwargs)
    
    async def search_audit_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Search audit events with filters"""
        
        try:
            # Filter events
            filtered_events = []
            
            for event in self.audit_events:
                # Apply filters
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                if user_id and event.user_id != user_id:
                    continue
                if resource_id and event.resource_id != resource_id:
                    continue
                if action and event.action != action:
                    continue
                
                filtered_events.append(event)
                
                if len(filtered_events) >= limit:
                    break
            
            return filtered_events
            
        except Exception as e:
            logger.error(f"Error searching audit events: {e}", exc_info=True)
            return []
    
    # ===== COMPLIANCE REPORTING =====
    
    async def generate_compliance_report(
        self,
        assessment_type: str,
        scope: Dict[str, Any],
        framework: Optional[str] = None
    ) -> str:
        """Generate a compliance assessment report"""
        
        report_id = f"compliance_{uuid.uuid4().hex[:12]}"
        
        try:
            logger.info(f"Generating compliance report: {report_id}")
            
            # Gather compliance data
            findings = []
            compliance_score = 0.0
            recommendations = []
            
            # Evaluate policies
            policy_compliance = await self._assess_policy_compliance(scope)
            findings.extend(policy_compliance["findings"])
            recommendations.extend(policy_compliance["recommendations"])
            
            # Check regulatory compliance if framework specified
            if framework:
                regulatory_compliance = await self._assess_regulatory_compliance(framework, scope)
                findings.extend(regulatory_compliance["findings"])
                recommendations.extend(regulatory_compliance["recommendations"])
            
            # Calculate overall compliance score
            if findings:
                compliant_findings = sum(1 for f in findings if f.get("status") == "compliant")
                compliance_score = compliant_findings / len(findings)
            else:
                compliance_score = 1.0
            
            # Determine compliance status
            if compliance_score >= 0.9:
                status = ComplianceStatus.COMPLIANT
            elif compliance_score >= 0.7:
                status = ComplianceStatus.PENDING_REVIEW
            else:
                status = ComplianceStatus.NON_COMPLIANT
            
            # Generate risk assessment
            risk_assessment = await self._generate_compliance_risk_assessment(findings)
            
            # Create report
            report = ComplianceReport(
                report_id=report_id,
                assessment_type=assessment_type,
                scope=scope,
                compliance_status=status,
                compliance_score=compliance_score,
                findings=findings,
                recommendations=recommendations,
                risk_assessment=risk_assessment,
                generated_at=datetime.utcnow(),
                next_assessment_due=datetime.utcnow() + timedelta(days=90)
            )
            
            # Store report
            self.compliance_reports[report_id] = report
            
            # Log audit event
            await self._log_audit_event(
                event_type=AuditEventType.COMPLIANCE_CHECK,
                action="compliance_report_generated",
                resource_id=report_id,
                resource_type="compliance_report",
                details={
                    "assessment_type": assessment_type,
                    "compliance_score": compliance_score,
                    "status": status.value
                }
            )
            
            logger.info(f"Compliance report generated: {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}", exc_info=True)
            raise
    
    async def _assess_policy_compliance(self, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance with internal policies"""
        
        findings = []
        recommendations = []
        
        try:
            # Check active policies
            for policy_type, policy_ids in self.active_policies_by_type.items():
                for policy_id in policy_ids:
                    policy = self.policies[policy_id]
                    
                    # Check if policy applies to scope
                    if self._policy_applies_to_scope(policy, scope):
                        # Get recent violations for this policy
                        recent_violations = [
                            v for v in self.policy_violations.values()
                            if v.policy_id == policy_id and 
                            v.detected_at > datetime.utcnow() - timedelta(days=30)
                        ]
                        
                        finding = {
                            "policy_id": policy_id,
                            "policy_name": policy.name,
                            "policy_type": policy_type.value,
                            "violations_count": len(recent_violations),
                            "status": "compliant" if len(recent_violations) == 0 else "non_compliant"
                        }
                        
                        findings.append(finding)
                        
                        if recent_violations:
                            recommendations.append(f"Address {len(recent_violations)} violations in policy '{policy.name}'")
            
            return {
                "findings": findings,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error assessing policy compliance: {e}", exc_info=True)
            return {"findings": [], "recommendations": []}
    
    async def _assess_regulatory_compliance(self, framework: str, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance with regulatory frameworks"""
        
        findings = []
        recommendations = []
        
        try:
            requirements = self.regulatory_frameworks.get(framework, {})
            
            for requirement_id, requirement in requirements.items():
                # Check compliance with each requirement
                compliance_status = await self._check_regulatory_requirement(requirement, scope)
                
                finding = {
                    "requirement_id": requirement_id,
                    "requirement_name": requirement.get("name", requirement_id),
                    "framework": framework,
                    "status": compliance_status["status"],
                    "details": compliance_status["details"]
                }
                
                findings.append(finding)
                
                if compliance_status["status"] != "compliant":
                    recommendations.extend(compliance_status.get("recommendations", []))
            
            return {
                "findings": findings,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error assessing regulatory compliance: {e}", exc_info=True)
            return {"findings": [], "recommendations": []}
    
    # ===== RISK ASSESSMENT =====
    
    async def conduct_risk_assessment(
        self,
        target_id: str,
        target_type: str,
        assessed_by: str = "system"
    ) -> str:
        """Conduct a comprehensive risk assessment"""
        
        assessment_id = f"risk_{uuid.uuid4().hex[:12]}"
        
        try:
            logger.info(f"Conducting risk assessment: {assessment_id}")
            
            # Define risk categories
            risk_categories = {
                "privacy_risk": 0.0,
                "security_risk": 0.0,
                "fairness_risk": 0.0,
                "performance_risk": 0.0,
                "compliance_risk": 0.0,
                "operational_risk": 0.0
            }
            
            risk_factors = []
            mitigation_strategies = []
            
            # Assess each risk category
            if target_type == "model":
                privacy_assessment = await self._assess_privacy_risk(target_id)
                risk_categories["privacy_risk"] = privacy_assessment["score"]
                risk_factors.extend(privacy_assessment["factors"])
                mitigation_strategies.extend(privacy_assessment["mitigations"])
                
                fairness_assessment = await self._assess_fairness_risk(target_id)
                risk_categories["fairness_risk"] = fairness_assessment["score"]
                risk_factors.extend(fairness_assessment["factors"])
                mitigation_strategies.extend(fairness_assessment["mitigations"])
                
                performance_assessment = await self._assess_performance_risk(target_id)
                risk_categories["performance_risk"] = performance_assessment["score"]
                risk_factors.extend(performance_assessment["factors"])
                mitigation_strategies.extend(performance_assessment["mitigations"])
            
            elif target_type == "dataset":
                privacy_assessment = await self._assess_data_privacy_risk(target_id)
                risk_categories["privacy_risk"] = privacy_assessment["score"]
                risk_factors.extend(privacy_assessment["factors"])
                mitigation_strategies.extend(privacy_assessment["mitigations"])
            
            # Always assess security and compliance risks
            security_assessment = await self._assess_security_risk(target_id, target_type)
            risk_categories["security_risk"] = security_assessment["score"]
            risk_factors.extend(security_assessment["factors"])
            mitigation_strategies.extend(security_assessment["mitigations"])
            
            compliance_assessment = await self._assess_compliance_risk(target_id, target_type)
            risk_categories["compliance_risk"] = compliance_assessment["score"]
            risk_factors.extend(compliance_assessment["factors"])
            mitigation_strategies.extend(compliance_assessment["mitigations"])
            
            # Calculate overall risk score (weighted average)
            weights = {
                "privacy_risk": 0.25,
                "security_risk": 0.25,
                "fairness_risk": 0.20,
                "performance_risk": 0.15,
                "compliance_risk": 0.10,
                "operational_risk": 0.05
            }
            
            overall_risk_score = sum(
                risk_categories[category] * weights[category]
                for category in risk_categories
            )
            
            # Create risk assessment
            assessment = RiskAssessment(
                assessment_id=assessment_id,
                target_id=target_id,
                target_type=target_type,
                risk_categories=risk_categories,
                overall_risk_score=overall_risk_score,
                risk_factors=risk_factors,
                mitigation_strategies=list(set(mitigation_strategies)),  # Remove duplicates
                assessed_at=datetime.utcnow(),
                assessed_by=assessed_by,
                next_assessment_due=datetime.utcnow() + timedelta(days=90)
            )
            
            # Store assessment
            self.risk_assessments[assessment_id] = assessment
            self.governance_metrics["risk_assessments"] += 1
            
            # Log audit event
            await self._log_audit_event(
                event_type=AuditEventType.RISK_ASSESSMENT,
                action="risk_assessment_conducted",
                resource_id=target_id,
                resource_type=target_type,
                details={
                    "assessment_id": assessment_id,
                    "overall_risk_score": overall_risk_score,
                    "risk_categories": risk_categories
                },
                user_id=assessed_by
            )
            
            logger.info(f"Risk assessment completed: {assessment_id}")
            return assessment_id
            
        except Exception as e:
            logger.error(f"Error conducting risk assessment: {e}", exc_info=True)
            raise
    
    async def _assess_privacy_risk(self, model_id: str) -> Dict[str, Any]:
        """Assess privacy risk for a model"""
        
        risk_score = 0.0
        factors = []
        mitigations = []
        
        try:
            # Check if model uses sensitive data
            # This would integrate with the model registry and data lineage
            if self.model_registry_service:
                model_info = await self.model_registry_service.get_model_info(model_id)
                if model_info:
                    training_data = model_info.get("training_data", {})
                    if training_data.get("contains_pii", False):
                        risk_score += 0.3
                        factors.append("Model trained on data containing PII")
                        mitigations.append("Implement differential privacy")
                        mitigations.append("Use data anonymization techniques")
            
            # Check for explanation availability (can reveal sensitive information)
            explanation_available = True  # Would check with XAI service
            if explanation_available:
                risk_score += 0.2
                factors.append("Model explanations may reveal sensitive patterns")
                mitigations.append("Implement explanation privacy protection")
            
            return {
                "score": min(risk_score, 1.0),
                "factors": factors,
                "mitigations": mitigations
            }
            
        except Exception as e:
            logger.warning(f"Error assessing privacy risk: {e}")
            return {"score": 0.5, "factors": ["Unable to assess privacy risk"], "mitigations": []}
    
    async def _assess_fairness_risk(self, model_id: str) -> Dict[str, Any]:
        """Assess fairness risk for a model"""
        
        risk_score = 0.0
        factors = []
        mitigations = []
        
        try:
            # Check for bias in model predictions
            # This would integrate with the XAI service
            bias_metrics = {}  # Would get from XAI service
            
            demographic_parity = bias_metrics.get("demographic_parity", 0.0)
            if demographic_parity > 0.1:
                risk_score += 0.4
                factors.append(f"High demographic parity: {demographic_parity:.3f}")
                mitigations.append("Implement bias mitigation techniques")
                mitigations.append("Retrain model with balanced data")
            
            equalized_odds = bias_metrics.get("equalized_odds", 0.0)
            if equalized_odds > 0.1:
                risk_score += 0.3
                factors.append(f"High equalized odds difference: {equalized_odds:.3f}")
                mitigations.append("Apply post-processing fairness corrections")
            
            return {
                "score": min(risk_score, 1.0),
                "factors": factors,
                "mitigations": mitigations
            }
            
        except Exception as e:
            logger.warning(f"Error assessing fairness risk: {e}")
            return {"score": 0.3, "factors": ["Unable to assess fairness risk"], "mitigations": []}
    
    async def _assess_performance_risk(self, model_id: str) -> Dict[str, Any]:
        """Assess performance risk for a model"""
        
        risk_score = 0.0
        factors = []
        mitigations = []
        
        try:
            # Check model performance metrics
            # This would integrate with the model registry
            performance_metrics = {}  # Would get from model registry
            
            accuracy = performance_metrics.get("accuracy", 0.0)
            if accuracy < 0.8:
                risk_score += 0.3
                factors.append(f"Low model accuracy: {accuracy:.3f}")
                mitigations.append("Retrain model with more data")
                mitigations.append("Improve feature engineering")
            
            # Check for concept drift
            drift_score = performance_metrics.get("drift_score", 0.0)
            if drift_score > 0.3:
                risk_score += 0.4
                factors.append(f"High concept drift detected: {drift_score:.3f}")
                mitigations.append("Implement drift monitoring")
                mitigations.append("Schedule regular model retraining")
            
            return {
                "score": min(risk_score, 1.0),
                "factors": factors,
                "mitigations": mitigations
            }
            
        except Exception as e:
            logger.warning(f"Error assessing performance risk: {e}")
            return {"score": 0.3, "factors": ["Unable to assess performance risk"], "mitigations": []}
    
    async def _assess_security_risk(self, target_id: str, target_type: str) -> Dict[str, Any]:
        """Assess security risk for a resource"""
        
        risk_score = 0.0
        factors = []
        mitigations = []
        
        try:
            # Check encryption status
            is_encrypted = False  # Would check with security service
            if not is_encrypted:
                risk_score += 0.3
                factors.append("Resource is not encrypted")
                mitigations.append("Implement encryption at rest and in transit")
            
            # Check access controls
            has_access_control = False  # Would check with auth service
            if not has_access_control:
                risk_score += 0.4
                factors.append("No access controls configured")
                mitigations.append("Implement role-based access control")
            
            # Check for recent vulnerabilities
            has_vulnerabilities = False  # Would check with security scanner
            if has_vulnerabilities:
                risk_score += 0.5
                factors.append("Security vulnerabilities detected")
                mitigations.append("Apply security patches")
                mitigations.append("Conduct security audit")
            
            return {
                "score": min(risk_score, 1.0),
                "factors": factors,
                "mitigations": mitigations
            }
            
        except Exception as e:
            logger.warning(f"Error assessing security risk: {e}")
            return {"score": 0.4, "factors": ["Unable to assess security risk"], "mitigations": []}
    
    async def _assess_compliance_risk(self, target_id: str, target_type: str) -> Dict[str, Any]:
        """Assess compliance risk for a resource"""
        
        risk_score = 0.0
        factors = []
        mitigations = []
        
        try:
            # Check for recent policy violations
            recent_violations = [
                v for v in self.policy_violations.values()
                if v.resource_id == target_id and
                v.detected_at > datetime.utcnow() - timedelta(days=30)
            ]
            
            if recent_violations:
                violation_severity_scores = {
                    ViolationSeverity.LOW: 0.1,
                    ViolationSeverity.MEDIUM: 0.3,
                    ViolationSeverity.HIGH: 0.6,
                    ViolationSeverity.CRITICAL: 1.0
                }
                
                for violation in recent_violations:
                    risk_score += violation_severity_scores[violation.severity]
                
                factors.append(f"{len(recent_violations)} policy violations in last 30 days")
                mitigations.append("Address outstanding policy violations")
                mitigations.append("Implement compliance monitoring")
            
            return {
                "score": min(risk_score, 1.0),
                "factors": factors,
                "mitigations": mitigations
            }
            
        except Exception as e:
            logger.warning(f"Error assessing compliance risk: {e}")
            return {"score": 0.2, "factors": ["Unable to assess compliance risk"], "mitigations": []}
    
    # ===== AUTOMATED REMEDIATION =====
    
    async def _apply_enforcement_action(
        self,
        policy: GovernancePolicy,
        violation: PolicyViolation,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply automated enforcement action"""
        
        try:
            enforcement_action = None
            
            # Determine appropriate action based on violation
            if violation.severity == ViolationSeverity.CRITICAL:
                if violation.resource_type == "model":
                    # Immediately stop model serving
                    enforcement_action = await self._stop_model_serving(violation.resource_id)
                elif violation.resource_type == "dataset":
                    # Quarantine dataset
                    enforcement_action = await self._quarantine_dataset(violation.resource_id)
            
            elif violation.severity == ViolationSeverity.HIGH:
                if violation.resource_type == "model":
                    # Flag model for review
                    enforcement_action = await self._flag_model_for_review(violation.resource_id)
                elif violation.resource_type == "dataset":
                    # Restrict dataset access
                    enforcement_action = await self._restrict_dataset_access(violation.resource_id)
            
            if enforcement_action:
                # Log remediation action
                await self._log_audit_event(
                    event_type=AuditEventType.REMEDIATION_ACTION,
                    action="automated_remediation",
                    resource_id=violation.resource_id,
                    resource_type=violation.resource_type,
                    details={
                        "violation_id": violation.violation_id,
                        "action_type": enforcement_action["type"],
                        "policy_id": policy.policy_id
                    }
                )
                
                self.governance_metrics["automated_remediations"] += 1
            
            return enforcement_action
            
        except Exception as e:
            logger.error(f"Error applying enforcement action: {e}", exc_info=True)
            return None
    
    async def _stop_model_serving(self, model_id: str) -> Dict[str, Any]:
        """Stop model serving immediately"""
        # In practice, this would integrate with the model serving infrastructure
        logger.warning(f"Stopping model serving for model: {model_id}")
        return {
            "type": "stop_serving",
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _quarantine_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Quarantine a dataset"""
        # In practice, this would move the dataset to a quarantine location
        logger.warning(f"Quarantining dataset: {dataset_id}")
        return {
            "type": "quarantine_dataset",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _flag_model_for_review(self, model_id: str) -> Dict[str, Any]:
        """Flag model for manual review"""
        logger.info(f"Flagging model for review: {model_id}")
        return {
            "type": "flag_for_review",
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _restrict_dataset_access(self, dataset_id: str) -> Dict[str, Any]:
        """Restrict access to a dataset"""
        logger.info(f"Restricting access to dataset: {dataset_id}")
        return {
            "type": "restrict_access",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # ===== UTILITY METHODS =====
    
    async def _initialize_service_integrations(self):
        """Initialize integrations with other services"""
        
        try:
            self.model_registry_service = ModelRegistryService()
            self.data_versioning_service = DataVersioningService()
            self.feature_store_service = FeatureStoreService()
            self.kg_client = KnowledgeGraphClient()
            
            # Initialize services
            await self.model_registry_service.initialize()
            await self.data_versioning_service.initialize()
            await self.feature_store_service.initialize()
            
            logger.info("Service integrations initialized successfully")
            
        except Exception as e:
            logger.warning(f"Error initializing service integrations: {e}")
    
    async def _load_governance_state(self):
        """Load governance state from storage"""
        
        try:
            # Load policies
            policies_file = self.storage_path / "policies.json"
            if policies_file.exists():
                with open(policies_file, 'r') as f:
                    policies_data = json.load(f)
                    for policy_data in policies_data:
                        policy = GovernancePolicy(**policy_data)
                        self.policies[policy.policy_id] = policy
                        if policy.status == PolicyStatus.ACTIVE:
                            self.active_policies_by_type[policy.policy_type].append(policy.policy_id)
            
            # Load violations
            violations_file = self.storage_path / "violations.json"
            if violations_file.exists():
                with open(violations_file, 'r') as f:
                    violations_data = json.load(f)
                    for violation_data in violations_data:
                        violation_data['detected_at'] = datetime.fromisoformat(violation_data['detected_at'])
                        if violation_data.get('resolved_at'):
                            violation_data['resolved_at'] = datetime.fromisoformat(violation_data['resolved_at'])
                        violation = PolicyViolation(**violation_data)
                        self.policy_violations[violation.violation_id] = violation
            
            logger.info("Governance state loaded successfully")
            
        except Exception as e:
            logger.warning(f"Error loading governance state: {e}")
    
    async def _save_governance_state(self):
        """Save governance state to storage"""
        
        try:
            # Save policies
            policies_data = []
            for policy in self.policies.values():
                policy_dict = asdict(policy)
                policy_dict['created_at'] = policy.created_at.isoformat()
                policy_dict['updated_at'] = policy.updated_at.isoformat()
                policy_dict['policy_type'] = policy.policy_type.value
                policy_dict['status'] = policy.status.value
                policies_data.append(policy_dict)
            
            policies_file = self.storage_path / "policies.json"
            with open(policies_file, 'w') as f:
                json.dump(policies_data, f, indent=2)
            
            # Save violations
            violations_data = []
            for violation in self.policy_violations.values():
                violation_dict = asdict(violation)
                violation_dict['detected_at'] = violation.detected_at.isoformat()
                if violation.resolved_at:
                    violation_dict['resolved_at'] = violation.resolved_at.isoformat()
                violation_dict['severity'] = violation.severity.value
                violations_data.append(violation_dict)
            
            violations_file = self.storage_path / "violations.json"
            with open(violations_file, 'w') as f:
                json.dump(violations_data, f, indent=2)
            
            logger.info("Governance state saved successfully")
            
        except Exception as e:
            logger.warning(f"Error saving governance state: {e}")
    
    async def _initialize_default_policies(self):
        """Initialize default governance policies"""
        
        try:
            # Data Privacy Policy
            await self.create_policy(
                name="GDPR Data Privacy",
                description="Ensures GDPR compliance for data processing",
                policy_type=PolicyType.DATA_PRIVACY,
                rules=[
                    {
                        "type": "data_privacy",
                        "check_pii": True,
                        "require_consent": True,
                        "max_retention_days": 2555  # 7 years
                    }
                ],
                enforcement_mode="blocking",
                scope={"resource_types": ["dataset", "feature_group"]}
            )
            
            # Model Fairness Policy
            await self.create_policy(
                name="Model Fairness Standards",
                description="Ensures fair and unbiased model behavior",
                policy_type=PolicyType.MODEL_FAIRNESS,
                rules=[
                    {
                        "type": "model_fairness",
                        "max_demographic_parity": 0.1,
                        "max_equalized_odds": 0.1
                    }
                ],
                enforcement_mode="warning",
                scope={"resource_types": ["model"]}
            )
            
            # Explainability Policy
            await self.create_policy(
                name="High-Risk Model Explainability",
                description="Requires explanations for high-risk models",
                policy_type=PolicyType.EXPLAINABILITY,
                rules=[
                    {
                        "type": "explainability",
                        "require_explanation": True,
                        "min_explanation_quality": 0.7
                    }
                ],
                enforcement_mode="blocking",
                scope={"resource_types": ["model"], "risk_level": "high"}
            )
            
            # Security Policy
            await self.create_policy(
                name="Data Security Standards",
                description="Ensures proper security measures for sensitive data",
                policy_type=PolicyType.SECURITY,
                rules=[
                    {
                        "type": "security",
                        "require_encryption": True,
                        "require_access_control": True,
                        "require_vulnerability_scan": True,
                        "max_scan_age_days": 30
                    }
                ],
                enforcement_mode="blocking",
                scope={"resource_types": ["dataset", "model"], "sensitivity": "high"}
            )
            
            logger.info("Default policies initialized")
            
        except Exception as e:
            logger.error(f"Error initializing default policies: {e}")
    
    async def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        
        try:
            # Start policy evaluation monitoring
            task = asyncio.create_task(self._policy_evaluation_monitor())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            # Start compliance monitoring
            task = asyncio.create_task(self._compliance_monitor())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            # Start audit cleanup
            task = asyncio.create_task(self._audit_cleanup_monitor())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            logger.info("Background monitoring started")
            
        except Exception as e:
            logger.warning(f"Error starting background monitoring: {e}")
    
    async def _policy_evaluation_monitor(self):
        """Monitor for policy evaluation triggers"""
        
        while True:
            try:
                await asyncio.sleep(self.policy_engine_config["evaluation_interval"])
                
                # Periodic policy evaluation
                await self._run_periodic_policy_evaluation()
                
            except Exception as e:
                logger.error(f"Error in policy evaluation monitor: {e}")
    
    async def _compliance_monitor(self):
        """Monitor compliance status and generate reports"""
        
        while True:
            try:
                await asyncio.sleep(86400)  # Daily
                
                # Generate daily compliance summary
                await self._generate_daily_compliance_summary()
                
            except Exception as e:
                logger.error(f"Error in compliance monitor: {e}")
    
    async def _audit_cleanup_monitor(self):
        """Clean up old audit events"""
        
        while True:
            try:
                await asyncio.sleep(86400)  # Daily
                
                # Clean up old audit events
                cutoff_date = datetime.utcnow() - timedelta(days=self.audit_retention_days)
                
                # Remove old events from memory
                self.audit_events = deque(
                    [event for event in self.audit_events if event.timestamp > cutoff_date],
                    maxlen=1000000
                )
                
                logger.info("Completed audit cleanup")
                
            except Exception as e:
                logger.error(f"Error in audit cleanup: {e}")
    
    # ===== HELPER METHODS =====
    
    def _load_policy_templates(self) -> Dict[str, Any]:
        """Load policy templates"""
        return {
            "gdpr_privacy": {
                "name": "GDPR Privacy Policy",
                "rules": [{"type": "data_privacy", "check_pii": True, "require_consent": True}]
            },
            "model_fairness": {
                "name": "Model Fairness Policy",
                "rules": [{"type": "model_fairness", "max_demographic_parity": 0.1}]
            }
        }
    
    def _load_gdpr_requirements(self) -> Dict[str, Any]:
        """Load GDPR compliance requirements"""
        return {
            "data_protection": {
                "name": "Data Protection Measures",
                "requirements": ["encryption", "access_control", "consent_management"]
            },
            "right_to_erasure": {
                "name": "Right to Erasure",
                "requirements": ["deletion_capability", "audit_trail"]
            }
        }
    
    def _load_ccpa_requirements(self) -> Dict[str, Any]:
        """Load CCPA compliance requirements"""
        return {
            "data_transparency": {
                "name": "Data Processing Transparency",
                "requirements": ["data_inventory", "purpose_documentation"]
            }
        }
    
    def _load_ai_act_requirements(self) -> Dict[str, Any]:
        """Load EU AI Act compliance requirements"""
        return {
            "high_risk_systems": {
                "name": "High-Risk AI Systems",
                "requirements": ["risk_assessment", "explainability", "human_oversight"]
            }
        }
    
    def _load_sox_requirements(self) -> Dict[str, Any]:
        """Load SOX compliance requirements"""
        return {
            "financial_controls": {
                "name": "Financial Data Controls",
                "requirements": ["audit_trail", "access_control", "change_management"]
            }
        }
    
    def _load_hipaa_requirements(self) -> Dict[str, Any]:
        """Load HIPAA compliance requirements"""
        return {
            "phi_protection": {
                "name": "Protected Health Information",
                "requirements": ["encryption", "access_logging", "minimum_necessary"]
            }
        }
    
    # Placeholder methods for more detailed implementations
    async def _validate_policy_rules(self, rules: List[Dict[str, Any]], policy_type: PolicyType):
        """Validate policy rules"""
        pass
    
    async def _get_applicable_policies(self, resource_type: str, context: Dict[str, Any]) -> List[str]:
        """Get policies applicable to a resource"""
        applicable = []
        for policy_id, policy in self.policies.items():
            if policy.status == PolicyStatus.ACTIVE:
                if self._policy_applies_to_resource(policy, resource_type, context):
                    applicable.append(policy_id)
        return applicable
    
    def _policy_applies_to_resource(self, policy: GovernancePolicy, resource_type: str, context: Dict[str, Any]) -> bool:
        """Check if policy applies to a resource"""
        scope = policy.scope
        
        # Check resource type
        if "resource_types" in scope:
            if resource_type not in scope["resource_types"]:
                return False
        
        # Check other scope criteria
        if "risk_level" in scope:
            if context.get("risk_level") != scope["risk_level"]:
                return False
        
        if "sensitivity" in scope:
            if context.get("sensitivity") != scope["sensitivity"]:
                return False
        
        return True
    
    def _policy_applies_to_scope(self, policy: GovernancePolicy, scope: Dict[str, Any]) -> bool:
        """Check if policy applies to a compliance scope"""
        # Simplified implementation
        return True
    
    async def _persist_audit_event(self, event: AuditEvent):
        """Persist audit event to storage"""
        # In practice, this would write to a database or file system
        pass
    
    async def _publish_audit_event(self, event: AuditEvent):
        """Publish audit event to message stream"""
        try:
            await shared_pulsar_client.publish("q.governance.audit", asdict(event))
        except Exception as e:
            logger.warning(f"Failed to publish audit event: {e}")
    
    async def _subscribe_to_system_events(self):
        """Subscribe to system events for monitoring"""
        try:
            await shared_pulsar_client.subscribe(
                "q.model.deployed",
                self._handle_model_deployment_event,
                subscription_name="governance_model_deployment"
            )
            
            await shared_pulsar_client.subscribe(
                "q.data.accessed",
                self._handle_data_access_event,
                subscription_name="governance_data_access"
            )
            
        except Exception as e:
            logger.warning(f"Error subscribing to system events: {e}")
    
    async def _handle_model_deployment_event(self, event_data: Dict[str, Any]):
        """Handle model deployment events"""
        try:
            model_id = event_data.get("model_id")
            if model_id:
                # Evaluate policies for the deployed model
                await self.evaluate_policies("model", model_id, event_data)
                
        except Exception as e:
            logger.error(f"Error handling model deployment event: {e}")
    
    async def _handle_data_access_event(self, event_data: Dict[str, Any]):
        """Handle data access events"""
        try:
            dataset_id = event_data.get("dataset_id")
            user_id = event_data.get("user_id")
            
            if dataset_id:
                # Log data access audit event
                await self._log_audit_event(
                    event_type=AuditEventType.DATA_ACCESS,
                    action="data_accessed",
                    resource_id=dataset_id,
                    resource_type="dataset",
                    user_id=user_id,
                    details=event_data
                )
                
        except Exception as e:
            logger.error(f"Error handling data access event: {e}")
    
    async def _run_periodic_policy_evaluation(self):
        """Run periodic policy evaluation"""
        # Placeholder for periodic evaluation logic
        pass
    
    async def _generate_daily_compliance_summary(self):
        """Generate daily compliance summary"""
        # Placeholder for daily summary generation
        pass
    
    async def _check_regulatory_requirement(self, requirement: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with a regulatory requirement"""
        # Placeholder implementation
        return {
            "status": "compliant",
            "details": "Requirement satisfied",
            "recommendations": []
        }
    
    async def _generate_compliance_risk_assessment(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate risk assessment from compliance findings"""
        # Simplified risk assessment
        non_compliant_count = sum(1 for f in findings if f.get("status") != "compliant")
        risk_level = "low"
        if non_compliant_count > len(findings) * 0.5:
            risk_level = "high"
        elif non_compliant_count > len(findings) * 0.2:
            risk_level = "medium"
        
        return {
            "risk_level": risk_level,
            "non_compliant_items": non_compliant_count,
            "total_items": len(findings)
        }
    
    async def _assess_data_privacy_risk(self, dataset_id: str) -> Dict[str, Any]:
        """Assess privacy risk for a dataset"""
        # Placeholder implementation
        return {
            "score": 0.3,
            "factors": ["Contains PII data"],
            "mitigations": ["Implement data anonymization"]
        }

# Create global instance
ai_governance_service = AIGovernanceService() 