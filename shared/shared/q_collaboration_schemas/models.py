"""
Q Platform Collaboration Data Models

This module defines the data structures for human-AI collaboration including:
- Expert identification and consultation workflows
- Real-time collaboration sessions
- Approval workflows with context-aware routing
- AI explanation and transparency features
- Training data generation from human corrections
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import uuid

# ===== ENUMS =====

class CollaborationType(Enum):
    """Types of human-AI collaboration"""
    REAL_TIME = "real_time"
    EXPERT_CONSULTATION = "expert_consultation"
    APPROVAL_WORKFLOW = "approval_workflow"
    TRAINING_CORRECTION = "training_correction"
    EXPLANATION_REQUEST = "explanation_request"

class ExpertiseArea(Enum):
    """Areas of expertise for expert identification"""
    TECHNICAL = "technical"
    DOMAIN_SPECIFIC = "domain_specific"
    PROCESS_OPTIMIZATION = "process_optimization"
    RISK_ASSESSMENT = "risk_assessment"
    QUALITY_ASSURANCE = "quality_assurance"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    PERFORMANCE = "performance"

class CollaborationStatus(Enum):
    """Status of collaboration sessions"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"

class ApprovalDecision(Enum):
    """Approval workflow decisions"""
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    ESCALATED = "escalated"
    DEFERRED = "deferred"

class ExplanationType(Enum):
    """Types of AI explanations"""
    DECISION_RATIONALE = "decision_rationale"
    PROCESS_STEPS = "process_steps"
    CONFIDENCE_FACTORS = "confidence_factors"
    RISK_ASSESSMENT = "risk_assessment"
    ALTERNATIVE_OPTIONS = "alternative_options"
    LEARNING_SOURCES = "learning_sources"

# ===== CORE MODELS =====

@dataclass
class ExpertProfile:
    """Profile of a human expert in the system"""
    user_id: str
    name: str
    expertise_areas: List[ExpertiseArea]
    specializations: List[str]
    performance_metrics: Dict[str, float]
    availability_schedule: Dict[str, List[str]]  # day -> time_slots
    current_load: int
    max_concurrent_sessions: int
    response_time_avg: float  # minutes
    success_rate: float  # 0.0 to 1.0
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CollaborationSession:
    """A human-AI collaboration session"""
    session_id: str
    collaboration_type: CollaborationType
    agent_id: str
    user_id: str
    workflow_id: Optional[str]
    task_id: Optional[str]
    
    # Session details
    title: str
    description: str
    priority: int  # 1-5, 5 being highest
    status: CollaborationStatus
    
    # Participants
    primary_human: str
    assigned_experts: List[str]
    participating_agents: List[str]
    
    # Context
    context_data: Dict[str, Any]
    shared_workspace: Dict[str, Any]
    decision_points: List[Dict[str, Any]]
    
    # Timing
    started_at: datetime
    estimated_duration: int  # minutes
    actual_duration: Optional[int]
    deadline: Optional[datetime]
    
    # Outcomes
    resolution: Optional[str]
    decisions_made: List[Dict[str, Any]]
    training_data_generated: List[Dict[str, Any]]
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExpertConsultationRequest:
    """Request for expert consultation"""
    request_id: str
    session_id: str
    requester_id: str  # agent or human
    
    # Request details
    expertise_needed: List[ExpertiseArea]
    specific_skills: List[str]
    urgency_level: int  # 1-5
    estimated_time: int  # minutes
    
    # Context
    problem_description: str
    current_state: Dict[str, Any]
    attempted_solutions: List[Dict[str, Any]]
    constraints: List[str]
    
    # Matching
    suggested_experts: List[str]
    assigned_expert: Optional[str]
    backup_experts: List[str]
    
    # Status
    status: CollaborationStatus
    response_required_by: datetime
    
    # Outcomes
    expert_response: Optional[Dict[str, Any]]
    resolution_quality: Optional[float]
    follow_up_needed: bool
    
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ApprovalWorkflow:
    """Context-aware approval workflow"""
    workflow_id: str
    session_id: str
    requester_id: str
    
    # Approval request
    request_type: str
    request_title: str
    request_description: str
    request_data: Dict[str, Any]
    
    # Context analysis
    risk_level: int  # 1-5
    impact_assessment: Dict[str, Any]
    compliance_requirements: List[str]
    stakeholder_analysis: Dict[str, Any]
    
    # Routing
    approval_chain: List[str]
    current_approver: str
    escalation_path: List[str]
    auto_approval_eligible: bool
    
    # Decisions
    decisions: List[Dict[str, Any]]
    current_decision: Optional[ApprovalDecision]
    conditions: List[str]
    
    # Timing
    submitted_at: datetime
    due_date: datetime
    decision_deadline: Optional[datetime]
    
    # Outcomes
    final_decision: Optional[ApprovalDecision]
    decision_rationale: Optional[str]
    conditions_met: bool
    
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AIExplanation:
    """AI decision explanation and transparency"""
    explanation_id: str
    session_id: str
    agent_id: str
    requester_id: str
    
    # Explanation request
    explanation_type: ExplanationType
    decision_context: Dict[str, Any]
    specific_questions: List[str]
    
    # Explanation content
    rationale: str
    process_steps: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    uncertainty_factors: List[str]
    
    # Supporting information
    data_sources: List[str]
    model_details: Dict[str, Any]
    alternative_options: List[Dict[str, Any]]
    risk_factors: List[Dict[str, Any]]
    
    # Validation
    human_feedback: Optional[Dict[str, Any]]
    accuracy_verified: Optional[bool]
    explanation_quality: Optional[float]
    
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingDataPoint:
    """Training data generated from human corrections"""
    data_id: str
    session_id: str
    agent_id: str
    corrector_id: str
    
    # Original context
    original_input: Dict[str, Any]
    original_output: Dict[str, Any]
    original_confidence: float
    
    # Human correction
    corrected_output: Dict[str, Any]
    correction_type: str
    correction_rationale: str
    
    # Learning context
    error_category: str
    improvement_area: str
    generalization_scope: str
    
    # Quality metrics
    correction_quality: float
    consensus_level: float  # if multiple humans involved
    validation_status: str
    
    # Usage tracking
    used_in_training: bool
    training_impact: Optional[float]
    
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CollaborationMetrics:
    """Metrics for collaboration effectiveness"""
    metric_id: str
    session_id: str
    time_period: str  # daily, weekly, monthly
    
    # Efficiency metrics
    average_response_time: float
    resolution_time: float
    collaboration_overhead: float
    
    # Quality metrics
    decision_accuracy: float
    expert_satisfaction: float
    user_satisfaction: float
    
    # Outcome metrics
    problems_resolved: int
    escalations_needed: int
    training_data_generated: int
    
    # Trend analysis
    improvement_trends: Dict[str, float]
    bottleneck_analysis: Dict[str, Any]
    
    calculated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

# ===== HELPER MODELS =====

@dataclass
class CollaborationContext:
    """Context for collaboration sessions"""
    workflow_state: Dict[str, Any]
    agent_capabilities: Dict[str, Any]
    user_preferences: Dict[str, Any]
    historical_interactions: List[Dict[str, Any]]
    environmental_factors: Dict[str, Any]
    
@dataclass
class RealTimeUpdate:
    """Real-time updates during collaboration"""
    update_id: str
    session_id: str
    timestamp: datetime
    update_type: str
    content: Dict[str, Any]
    sender_id: str
    visibility: List[str]  # who can see this update
    
@dataclass
class CollaborationTemplate:
    """Template for common collaboration patterns"""
    template_id: str
    name: str
    description: str
    collaboration_type: CollaborationType
    steps: List[Dict[str, Any]]
    roles: List[str]
    success_criteria: List[str]
    estimated_duration: int
    usage_count: int
    success_rate: float
    
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

# ===== UTILITY FUNCTIONS =====

def generate_collaboration_id() -> str:
    """Generate a unique collaboration ID"""
    return f"collab_{uuid.uuid4().hex[:12]}"

def generate_expert_consultation_id() -> str:
    """Generate a unique expert consultation ID"""
    return f"consult_{uuid.uuid4().hex[:12]}"

def generate_approval_workflow_id() -> str:
    """Generate a unique approval workflow ID"""
    return f"approval_{uuid.uuid4().hex[:12]}"

def generate_explanation_id() -> str:
    """Generate a unique explanation ID"""
    return f"explain_{uuid.uuid4().hex[:12]}"

def generate_training_data_id() -> str:
    """Generate a unique training data ID"""
    return f"train_{uuid.uuid4().hex[:12]}"

def calculate_collaboration_priority(
    urgency: int,
    impact: int,
    complexity: int,
    expertise_availability: float
) -> int:
    """
    Calculate collaboration priority score
    
    Args:
        urgency: 1-5 (5 most urgent)
        impact: 1-5 (5 highest impact)
        complexity: 1-5 (5 most complex)
        expertise_availability: 0.0-1.0 (1.0 fully available)
    
    Returns:
        Priority score 1-5
    """
    base_priority = (urgency + impact + complexity) / 3
    availability_factor = 1 - expertise_availability
    adjusted_priority = base_priority + (availability_factor * 2)
    
    return min(5, max(1, round(adjusted_priority)))

def match_experts_to_request(
    request: ExpertConsultationRequest,
    available_experts: List[ExpertProfile]
) -> List[str]:
    """
    Match experts to consultation request based on expertise and availability
    
    Args:
        request: Expert consultation request
        available_experts: List of available expert profiles
    
    Returns:
        List of matched expert IDs in priority order
    """
    matches = []
    
    for expert in available_experts:
        # Calculate expertise match
        expertise_match = len(set(request.expertise_needed) & set(expert.expertise_areas))
        skill_match = len(set(request.specific_skills) & set(expert.specializations))
        
        # Calculate availability score
        availability_score = 1.0 - (expert.current_load / expert.max_concurrent_sessions)
        
        # Calculate overall score
        score = (expertise_match * 0.4) + (skill_match * 0.3) + (availability_score * 0.3)
        
        if score > 0.3:  # Minimum threshold
            matches.append((expert.user_id, score))
    
    # Sort by score and return top matches
    matches.sort(key=lambda x: x[1], reverse=True)
    return [match[0] for match in matches[:5]]  # Top 5 matches 