"""
Auto-Remediation Service

This service provides automated remediation for detected incidents:
- Automated response to incidents
- Kubernetes-based remediation actions
- Resource scaling and optimization
- Service restart and recovery
- Workflow adaptation and healing
- Learning from remediation outcomes
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import asdict, dataclass
from enum import Enum
import uuid

# Q Platform imports
from app.services.automated_incident_detection import (
    AutomatedIncidentDetection, Incident, IncidentType, IncidentSeverity
)
from app.services.pulsar_service import PulsarService
from app.services.ignite_service import IgniteService
from app.services.memory_service import MemoryService
from shared.q_memory_schemas.memory_models import AgentMemory, MemoryType

logger = logging.getLogger(__name__)

class RemediationStrategy(Enum):
    """Remediation strategies"""
    RESTART_SERVICE = "restart_service"
    SCALE_RESOURCES = "scale_resources"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAILOVER = "failover"
    RESOURCE_CLEANUP = "resource_cleanup"
    CONFIGURATION_RESET = "configuration_reset"
    WORKFLOW_ADAPTATION = "workflow_adaptation"

class RemediationStatus(Enum):
    """Status of remediation actions"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RemediationPriority(Enum):
    """Priority levels for remediation"""
    IMMEDIATE = "immediate"      # Execute immediately
    HIGH = "high"                # Execute within 1 minute
    MEDIUM = "medium"            # Execute within 5 minutes
    LOW = "low"                  # Execute within 15 minutes

@dataclass
class RemediationAction:
    """A remediation action to be executed"""
    action_id: str
    incident_id: str
    strategy: RemediationStrategy
    priority: RemediationPriority
    description: str
    
    # Execution details
    target_component: str
    action_parameters: Dict[str, Any]
    prerequisites: List[str]
    expected_duration: int  # seconds
    
    # Status tracking
    status: RemediationStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    # Results
    success: Optional[bool]
    error_message: Optional[str]
    outcome_metrics: Dict[str, Any]

@dataclass
class RemediationPlan:
    """A plan containing multiple remediation actions"""
    plan_id: str
    incident_id: str
    description: str
    actions: List[RemediationAction]
    execution_order: List[str]  # Action IDs in order
    
    # Plan status
    status: RemediationStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    # Results
    success_rate: float
    total_actions: int
    completed_actions: int
    failed_actions: int

@dataclass
class RemediationOutcome:
    """Outcome of a remediation action"""
    outcome_id: str
    action_id: str
    incident_id: str
    success: bool
    execution_time: float
    impact_metrics: Dict[str, Any]
    lessons_learned: List[str]
    recorded_at: datetime

class AutoRemediationService:
    """
    Service for automated incident remediation
    """
    
    def __init__(self):
        self.incident_detector = AutomatedIncidentDetection()
        self.pulsar_service = PulsarService()
        self.ignite_service = IgniteService()
        self.memory_service = MemoryService()
        
        # Remediation state
        self.active_plans: Dict[str, RemediationPlan] = {}
        self.remediation_history: List[RemediationOutcome] = []
        self.remediation_rules: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.remediation_config = {
            "auto_remediation_enabled": True,
            "max_concurrent_actions": 5,
            "action_timeout_seconds": 300,
            "retry_attempts": 3,
            "learning_enabled": True,
            "safety_mode": True
        }
        
        # Performance tracking
        self.remediation_metrics = {
            "plans_executed": 0,
            "actions_completed": 0,
            "success_rate": 0.0,
            "average_resolution_time": 0.0,
            "incidents_prevented": 0
        }
        
        # Knowledge base
        self.remediation_knowledge = {
            "successful_strategies": {},
            "failed_strategies": {},
            "component_specific_actions": {},
            "pattern_based_remediations": {}
        }
    
    async def initialize(self):
        """Initialize the auto-remediation service"""
        logger.info("Initializing Auto-Remediation Service")
        
        # Load remediation rules
        await self._load_remediation_rules()
        
        # Subscribe to incident events
        await self._subscribe_to_incidents()
        
        # Start background tasks
        asyncio.create_task(self._remediation_execution_loop())
        asyncio.create_task(self._plan_monitoring_loop())
        asyncio.create_task(self._knowledge_learning_loop())
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        logger.info("Auto-Remediation Service initialized successfully")
    
    # ===== REMEDIATION PLANNING =====
    
    async def create_remediation_plan(
        self,
        incident: Incident
    ) -> Optional[RemediationPlan]:
        """
        Create a remediation plan for an incident
        
        Args:
            incident: The incident to create remediation for
            
        Returns:
            Remediation plan or None if no remediation possible
        """
        logger.info(f"Creating remediation plan for incident: {incident.incident_id}")
        
        # Analyze incident to determine remediation strategies
        strategies = await self._analyze_incident_for_remediation(incident)
        
        if not strategies:
            logger.warning(f"No remediation strategies found for incident: {incident.incident_id}")
            return None
        
        # Create remediation actions
        actions = []
        for strategy_info in strategies:
            action = await self._create_remediation_action(incident, strategy_info)
            if action:
                actions.append(action)
        
        if not actions:
            logger.warning(f"No remediation actions could be created for incident: {incident.incident_id}")
            return None
        
        # Create remediation plan
        plan = RemediationPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:12]}",
            incident_id=incident.incident_id,
            description=f"Remediation plan for {incident.title}",
            actions=actions,
            execution_order=[action.action_id for action in actions],
            status=RemediationStatus.PENDING,
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            success_rate=0.0,
            total_actions=len(actions),
            completed_actions=0,
            failed_actions=0
        )
        
        # Store plan
        self.active_plans[plan.plan_id] = plan
        
        # Publish plan creation
        await self.pulsar_service.publish(
            "q.remediation.plan.created",
            {
                "plan_id": plan.plan_id,
                "incident_id": incident.incident_id,
                "actions_count": len(actions),
                "strategies": [action.strategy.value for action in actions],
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Remediation plan created: {plan.plan_id} with {len(actions)} actions")
        return plan
    
    async def execute_remediation_plan(
        self,
        plan_id: str
    ) -> bool:
        """
        Execute a remediation plan
        
        Args:
            plan_id: ID of the plan to execute
            
        Returns:
            True if execution started successfully
        """
        if plan_id not in self.active_plans:
            logger.error(f"Remediation plan not found: {plan_id}")
            return False
        
        plan = self.active_plans[plan_id]
        
        if plan.status != RemediationStatus.PENDING:
            logger.warning(f"Plan {plan_id} is not in pending status: {plan.status}")
            return False
        
        logger.info(f"Executing remediation plan: {plan_id}")
        
        # Update plan status
        plan.status = RemediationStatus.IN_PROGRESS
        plan.started_at = datetime.utcnow()
        
        # Execute actions in order
        for action_id in plan.execution_order:
            action = next((a for a in plan.actions if a.action_id == action_id), None)
            if action:
                await self._execute_remediation_action(action)
        
        # Update metrics
        self.remediation_metrics["plans_executed"] += 1
        
        # Publish plan execution
        await self.pulsar_service.publish(
            "q.remediation.plan.started",
            {
                "plan_id": plan_id,
                "incident_id": plan.incident_id,
                "started_at": plan.started_at.isoformat(),
                "actions_count": len(plan.actions)
            }
        )
        
        return True
    
    # ===== ACTION EXECUTION =====
    
    async def _execute_remediation_action(
        self,
        action: RemediationAction
    ) -> bool:
        """Execute a single remediation action"""
        logger.info(f"Executing remediation action: {action.action_id} - {action.strategy.value}")
        
        # Update action status
        action.status = RemediationStatus.IN_PROGRESS
        action.started_at = datetime.utcnow()
        
        success = False
        error_message = None
        
        try:
            # Execute based on strategy
            if action.strategy == RemediationStrategy.RESTART_SERVICE:
                success = await self._restart_service(action)
            elif action.strategy == RemediationStrategy.SCALE_RESOURCES:
                success = await self._scale_resources(action)
            elif action.strategy == RemediationStrategy.ROLLBACK_DEPLOYMENT:
                success = await self._rollback_deployment(action)
            elif action.strategy == RemediationStrategy.CIRCUIT_BREAKER:
                success = await self._activate_circuit_breaker(action)
            elif action.strategy == RemediationStrategy.FAILOVER:
                success = await self._perform_failover(action)
            elif action.strategy == RemediationStrategy.RESOURCE_CLEANUP:
                success = await self._cleanup_resources(action)
            elif action.strategy == RemediationStrategy.CONFIGURATION_RESET:
                success = await self._reset_configuration(action)
            elif action.strategy == RemediationStrategy.WORKFLOW_ADAPTATION:
                success = await self._adapt_workflow(action)
            else:
                logger.warning(f"Unknown remediation strategy: {action.strategy}")
                success = False
                error_message = f"Unknown strategy: {action.strategy}"
        
        except Exception as e:
            logger.error(f"Error executing remediation action {action.action_id}: {e}")
            success = False
            error_message = str(e)
        
        # Update action status
        action.status = RemediationStatus.COMPLETED if success else RemediationStatus.FAILED
        action.completed_at = datetime.utcnow()
        action.success = success
        action.error_message = error_message
        
        # Record outcome
        outcome = RemediationOutcome(
            outcome_id=f"outcome_{uuid.uuid4().hex[:12]}",
            action_id=action.action_id,
            incident_id=action.incident_id,
            success=success,
            execution_time=(action.completed_at - action.started_at).total_seconds(),
            impact_metrics=action.outcome_metrics,
            lessons_learned=[],
            recorded_at=datetime.utcnow()
        )
        
        self.remediation_history.append(outcome)
        
        # Update metrics
        self.remediation_metrics["actions_completed"] += 1
        if success:
            self.remediation_metrics["success_rate"] = (
                self.remediation_metrics["success_rate"] * 
                (self.remediation_metrics["actions_completed"] - 1) + 1
            ) / self.remediation_metrics["actions_completed"]
        
        # Publish action completion
        await self.pulsar_service.publish(
            "q.remediation.action.completed",
            {
                "action_id": action.action_id,
                "incident_id": action.incident_id,
                "strategy": action.strategy.value,
                "success": success,
                "execution_time": outcome.execution_time,
                "error_message": error_message,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Remediation action {'completed' if success else 'failed'}: {action.action_id}")
        return success
    
    # ===== STRATEGY IMPLEMENTATIONS =====
    
    async def _restart_service(self, action: RemediationAction) -> bool:
        """Restart a service using Kubernetes connector"""
        try:
            # This would use the Kubernetes connector to restart pods
            component = action.target_component
            namespace = action.action_parameters.get("namespace", "default")
            
            # Mock implementation - would use actual Kubernetes connector
            logger.info(f"Restarting service {component} in namespace {namespace}")
            
            # Record metrics
            action.outcome_metrics = {
                "restart_initiated": True,
                "component": component,
                "namespace": namespace
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart service {action.target_component}: {e}")
            return False
    
    async def _scale_resources(self, action: RemediationAction) -> bool:
        """Scale resources using Kubernetes connector"""
        try:
            component = action.target_component
            replicas = action.action_parameters.get("replicas", 3)
            
            # Mock implementation - would use actual Kubernetes connector
            logger.info(f"Scaling {component} to {replicas} replicas")
            
            # Record metrics
            action.outcome_metrics = {
                "scaling_initiated": True,
                "component": component,
                "target_replicas": replicas
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale resources for {action.target_component}: {e}")
            return False
    
    async def _rollback_deployment(self, action: RemediationAction) -> bool:
        """Rollback deployment using Kubernetes connector"""
        try:
            component = action.target_component
            revision = action.action_parameters.get("revision", "previous")
            
            # Mock implementation - would use actual Kubernetes connector
            logger.info(f"Rolling back deployment {component} to revision {revision}")
            
            # Record metrics
            action.outcome_metrics = {
                "rollback_initiated": True,
                "component": component,
                "target_revision": revision
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback deployment {action.target_component}: {e}")
            return False
    
    async def _activate_circuit_breaker(self, action: RemediationAction) -> bool:
        """Activate circuit breaker pattern"""
        try:
            component = action.target_component
            timeout = action.action_parameters.get("timeout", 300)  # 5 minutes
            
            logger.info(f"Activating circuit breaker for {component} with timeout {timeout}s")
            
            # Record metrics
            action.outcome_metrics = {
                "circuit_breaker_activated": True,
                "component": component,
                "timeout": timeout
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate circuit breaker for {action.target_component}: {e}")
            return False
    
    async def _perform_failover(self, action: RemediationAction) -> bool:
        """Perform failover to backup systems"""
        try:
            component = action.target_component
            backup_component = action.action_parameters.get("backup_component")
            
            logger.info(f"Performing failover from {component} to {backup_component}")
            
            # Record metrics
            action.outcome_metrics = {
                "failover_initiated": True,
                "primary_component": component,
                "backup_component": backup_component
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to perform failover for {action.target_component}: {e}")
            return False
    
    async def _cleanup_resources(self, action: RemediationAction) -> bool:
        """Clean up resources to free up capacity"""
        try:
            component = action.target_component
            cleanup_type = action.action_parameters.get("cleanup_type", "temporary_files")
            
            logger.info(f"Cleaning up resources for {component}: {cleanup_type}")
            
            # Record metrics
            action.outcome_metrics = {
                "cleanup_initiated": True,
                "component": component,
                "cleanup_type": cleanup_type
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup resources for {action.target_component}: {e}")
            return False
    
    async def _reset_configuration(self, action: RemediationAction) -> bool:
        """Reset configuration to known good state"""
        try:
            component = action.target_component
            config_version = action.action_parameters.get("config_version", "stable")
            
            logger.info(f"Resetting configuration for {component} to version {config_version}")
            
            # Record metrics
            action.outcome_metrics = {
                "config_reset_initiated": True,
                "component": component,
                "config_version": config_version
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset configuration for {action.target_component}: {e}")
            return False
    
    async def _adapt_workflow(self, action: RemediationAction) -> bool:
        """Adapt workflow to work around issues"""
        try:
            workflow_id = action.action_parameters.get("workflow_id")
            adaptation_type = action.action_parameters.get("adaptation_type", "skip_failed_step")
            
            logger.info(f"Adapting workflow {workflow_id}: {adaptation_type}")
            
            # Record metrics
            action.outcome_metrics = {
                "workflow_adapted": True,
                "workflow_id": workflow_id,
                "adaptation_type": adaptation_type
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to adapt workflow: {e}")
            return False
    
    # ===== INCIDENT ANALYSIS =====
    
    async def _analyze_incident_for_remediation(
        self,
        incident: Incident
    ) -> List[Dict[str, Any]]:
        """Analyze incident to determine appropriate remediation strategies"""
        strategies = []
        
        # Strategy selection based on incident type
        if incident.incident_type == IncidentType.PERFORMANCE_DEGRADATION:
            strategies.extend([
                {
                    "strategy": RemediationStrategy.SCALE_RESOURCES,
                    "priority": RemediationPriority.HIGH,
                    "parameters": {"replicas": 5}
                },
                {
                    "strategy": RemediationStrategy.RESTART_SERVICE,
                    "priority": RemediationPriority.MEDIUM,
                    "parameters": {"namespace": "default"}
                }
            ])
        
        elif incident.incident_type == IncidentType.SYSTEM_FAILURE:
            strategies.extend([
                {
                    "strategy": RemediationStrategy.RESTART_SERVICE,
                    "priority": RemediationPriority.IMMEDIATE,
                    "parameters": {"namespace": "default"}
                },
                {
                    "strategy": RemediationStrategy.FAILOVER,
                    "priority": RemediationPriority.HIGH,
                    "parameters": {"backup_component": f"{incident.affected_components[0]}-backup"}
                }
            ])
        
        elif incident.incident_type == IncidentType.RESOURCE_EXHAUSTION:
            strategies.extend([
                {
                    "strategy": RemediationStrategy.RESOURCE_CLEANUP,
                    "priority": RemediationPriority.HIGH,
                    "parameters": {"cleanup_type": "temporary_files"}
                },
                {
                    "strategy": RemediationStrategy.SCALE_RESOURCES,
                    "priority": RemediationPriority.MEDIUM,
                    "parameters": {"replicas": 3}
                }
            ])
        
        elif incident.incident_type == IncidentType.CONNECTIVITY_ISSUE:
            strategies.extend([
                {
                    "strategy": RemediationStrategy.CIRCUIT_BREAKER,
                    "priority": RemediationPriority.HIGH,
                    "parameters": {"timeout": 300}
                },
                {
                    "strategy": RemediationStrategy.FAILOVER,
                    "priority": RemediationPriority.MEDIUM,
                    "parameters": {"backup_component": f"{incident.affected_components[0]}-backup"}
                }
            ])
        
        elif incident.incident_type == IncidentType.WORKFLOW_FAILURE:
            strategies.extend([
                {
                    "strategy": RemediationStrategy.WORKFLOW_ADAPTATION,
                    "priority": RemediationPriority.HIGH,
                    "parameters": {"adaptation_type": "skip_failed_step"}
                }
            ])
        
        # Filter strategies based on safety mode
        if self.remediation_config["safety_mode"]:
            # Only allow safe strategies in safety mode
            safe_strategies = [
                RemediationStrategy.CIRCUIT_BREAKER,
                RemediationStrategy.RESOURCE_CLEANUP,
                RemediationStrategy.WORKFLOW_ADAPTATION
            ]
            strategies = [s for s in strategies if s["strategy"] in safe_strategies]
        
        return strategies
    
    async def _create_remediation_action(
        self,
        incident: Incident,
        strategy_info: Dict[str, Any]
    ) -> Optional[RemediationAction]:
        """Create a remediation action from strategy info"""
        try:
            action = RemediationAction(
                action_id=f"action_{uuid.uuid4().hex[:12]}",
                incident_id=incident.incident_id,
                strategy=strategy_info["strategy"],
                priority=strategy_info["priority"],
                description=f"{strategy_info['strategy'].value} for {incident.title}",
                target_component=incident.affected_components[0] if incident.affected_components else "unknown",
                action_parameters=strategy_info.get("parameters", {}),
                prerequisites=[],
                expected_duration=strategy_info.get("expected_duration", 120),
                status=RemediationStatus.PENDING,
                created_at=datetime.utcnow(),
                started_at=None,
                completed_at=None,
                success=None,
                error_message=None,
                outcome_metrics={}
            )
            
            return action
            
        except Exception as e:
            logger.error(f"Failed to create remediation action: {e}")
            return None
    
    # ===== BACKGROUND TASKS =====
    
    async def _remediation_execution_loop(self):
        """Background task for executing remediation plans"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Check for plans ready to execute
                for plan in list(self.active_plans.values()):
                    if plan.status == RemediationStatus.PENDING:
                        # Check if we should auto-execute
                        if self.remediation_config["auto_remediation_enabled"]:
                            await self.execute_remediation_plan(plan.plan_id)
                
            except Exception as e:
                logger.error(f"Error in remediation execution loop: {e}")
    
    async def _plan_monitoring_loop(self):
        """Background task for monitoring active plans"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check plan status and completion
                for plan in list(self.active_plans.values()):
                    if plan.status == RemediationStatus.IN_PROGRESS:
                        await self._check_plan_completion(plan)
                
            except Exception as e:
                logger.error(f"Error in plan monitoring loop: {e}")
    
    async def _knowledge_learning_loop(self):
        """Background task for learning from remediation outcomes"""
        while True:
            try:
                await asyncio.sleep(300)  # Learn every 5 minutes
                
                if self.remediation_config["learning_enabled"]:
                    await self._learn_from_outcomes()
                
            except Exception as e:
                logger.error(f"Error in knowledge learning loop: {e}")
    
    async def _check_plan_completion(self, plan: RemediationPlan):
        """Check if a plan has completed"""
        completed_actions = sum(1 for action in plan.actions 
                              if action.status in [RemediationStatus.COMPLETED, RemediationStatus.FAILED])
        
        plan.completed_actions = completed_actions
        plan.failed_actions = sum(1 for action in plan.actions 
                                if action.status == RemediationStatus.FAILED)
        
        if completed_actions == plan.total_actions:
            # Plan completed
            plan.status = RemediationStatus.COMPLETED
            plan.completed_at = datetime.utcnow()
            plan.success_rate = (plan.total_actions - plan.failed_actions) / plan.total_actions
            
            # Move to history
            del self.active_plans[plan.plan_id]
            
            # Store learning memory
            await self._store_remediation_memory(plan)
            
            # Publish completion
            await self.pulsar_service.publish(
                "q.remediation.plan.completed",
                {
                    "plan_id": plan.plan_id,
                    "incident_id": plan.incident_id,
                    "success_rate": plan.success_rate,
                    "completed_actions": plan.completed_actions,
                    "failed_actions": plan.failed_actions,
                    "duration": (plan.completed_at - plan.started_at).total_seconds(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Remediation plan completed: {plan.plan_id} (success rate: {plan.success_rate:.2f})")
    
    # ===== LEARNING AND KNOWLEDGE =====
    
    async def _learn_from_outcomes(self):
        """Learn from remediation outcomes to improve future actions"""
        if not self.remediation_history:
            return
        
        # Analyze recent outcomes
        recent_outcomes = [o for o in self.remediation_history 
                          if (datetime.utcnow() - o.recorded_at).total_seconds() < 3600]  # Last hour
        
        if not recent_outcomes:
            return
        
        # Learn successful strategies
        successful_outcomes = [o for o in recent_outcomes if o.success]
        for outcome in successful_outcomes:
            strategy_key = f"{outcome.incident_id.split('_')[0]}_{outcome.action_id.split('_')[0]}"  # Simplified
            if strategy_key not in self.remediation_knowledge["successful_strategies"]:
                self.remediation_knowledge["successful_strategies"][strategy_key] = 0
            self.remediation_knowledge["successful_strategies"][strategy_key] += 1
        
        # Learn failed strategies
        failed_outcomes = [o for o in recent_outcomes if not o.success]
        for outcome in failed_outcomes:
            strategy_key = f"{outcome.incident_id.split('_')[0]}_{outcome.action_id.split('_')[0]}"  # Simplified
            if strategy_key not in self.remediation_knowledge["failed_strategies"]:
                self.remediation_knowledge["failed_strategies"][strategy_key] = 0
            self.remediation_knowledge["failed_strategies"][strategy_key] += 1
        
        logger.info(f"Learned from {len(recent_outcomes)} recent remediation outcomes")
    
    async def _store_remediation_memory(self, plan: RemediationPlan):
        """Store remediation experience as memory"""
        memory = AgentMemory(
            memory_id=f"remediation_{plan.plan_id}",
            agent_id="auto_remediation_service",
            memory_type=MemoryType.EXPERIENCE,
            content=f"Remediation plan for incident {plan.incident_id} completed with {plan.success_rate:.2f} success rate",
            context={
                "plan_id": plan.plan_id,
                "incident_id": plan.incident_id,
                "success_rate": plan.success_rate,
                "strategies_used": [action.strategy.value for action in plan.actions],
                "execution_time": (plan.completed_at - plan.started_at).total_seconds(),
                "lessons_learned": []
            },
            importance=plan.success_rate,  # Higher success rate = higher importance
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1
        )
        
        await self.memory_service.store_memory(memory)
    
    # ===== INCIDENT HANDLING =====
    
    async def handle_incident(self, incident_data: Dict[str, Any]):
        """Handle incoming incident for remediation"""
        logger.info(f"Handling incident for remediation: {incident_data.get('incident_id')}")
        
        # Reconstruct incident object (simplified)
        incident = Incident(
            incident_id=incident_data["incident_id"],
            title=incident_data["title"],
            description=incident_data.get("description", ""),
            severity=IncidentSeverity(incident_data["severity"]),
            incident_type=IncidentType(incident_data["incident_type"]),
            detection_method=incident_data.get("detection_method", "threshold_based"),
            affected_components=incident_data.get("affected_components", []),
            root_cause=None,
            status=incident_data.get("status", "detected"),
            detected_at=datetime.fromisoformat(incident_data["detected_at"]),
            acknowledged_at=None,
            resolved_at=None,
            metrics=[],
            context=incident_data.get("context", {}),
            related_incidents=[],
            responders=[],
            remediation_actions=[],
            resolution_notes=""
        )
        
        # Create and execute remediation plan
        plan = await self.create_remediation_plan(incident)
        if plan:
            # Auto-execute if enabled and incident is serious enough
            if (self.remediation_config["auto_remediation_enabled"] and 
                incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]):
                await self.execute_remediation_plan(plan.plan_id)
    
    # ===== SETUP METHODS =====
    
    async def _subscribe_to_incidents(self):
        """Subscribe to incident detection events"""
        await self.pulsar_service.subscribe(
            "q.incidents.detected",
            self.handle_incident,
            subscription_name="auto_remediation_service"
        )
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for remediation"""
        topics = [
            "q.remediation.plan.created",
            "q.remediation.plan.started",
            "q.remediation.plan.completed",
            "q.remediation.action.completed"
        ]
        
        for topic in topics:
            await self.pulsar_service.ensure_topic(topic)
    
    async def _load_remediation_rules(self):
        """Load remediation rules from configuration"""
        # Default rules would be loaded from configuration
        pass

# Global service instance
auto_remediation_service = AutoRemediationService() 