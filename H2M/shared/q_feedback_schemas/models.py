# shared/q_feedback_schemas/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timezone
from enum import Enum
import uuid


class FeedbackType(str, Enum):
    """Types of feedback"""
    EXPLICIT_RATING = "explicit_rating"
    EXPLICIT_COMMENT = "explicit_comment"
    IMPLICIT_ACTION = "implicit_action"
    IMPLICIT_TIME = "implicit_time"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    

class FeedbackSentiment(str, Enum):
    """Sentiment analysis results"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"
    

class FeedbackContext(str, Enum):
    """Context where feedback was provided"""
    AGENT_RESPONSE = "agent_response"
    WORKFLOW_RESULT = "workflow_result"
    UI_INTERACTION = "ui_interaction"
    SEARCH_RESULT = "search_result"
    RECOMMENDATION = "recommendation"
    ERROR_RECOVERY = "error_recovery"
    

class BaseFeedback(BaseModel):
    """Base feedback model"""
    feedback_id: str = Field(default_factory=lambda: f"fb_{uuid.uuid4()}")
    user_id: str = Field(..., description="User who provided the feedback")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = Field(..., description="Session identifier for grouping feedback")
    context: FeedbackContext
    target_id: str = Field(..., description="ID of the target (agent, workflow, etc)")
    target_type: str = Field(..., description="Type of target (agent, workflow, ui_element)")
    

class ExplicitFeedback(BaseFeedback):
    """Explicit user feedback (ratings, comments, corrections)"""
    type: FeedbackType = Field(FeedbackType.EXPLICIT_RATING)
    rating: Optional[float] = Field(None, ge=1, le=5, description="1-5 star rating")
    comment: Optional[str] = Field(None, description="User's comment")
    tags: List[str] = Field(default_factory=list, description="User-provided tags")
    
    # For corrections
    original_content: Optional[str] = Field(None, description="What was shown to user")
    corrected_content: Optional[str] = Field(None, description="User's correction")
    
    # Sentiment analysis (filled by system)
    sentiment: Optional[FeedbackSentiment] = None
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    

class ImplicitFeedback(BaseFeedback):
    """Implicit feedback from user behavior"""
    type: FeedbackType = Field(FeedbackType.IMPLICIT_ACTION)
    action: str = Field(..., description="User action (click, scroll, copy, etc)")
    
    # Action-specific data
    action_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Time-based metrics
    time_to_action: Optional[float] = Field(None, description="Seconds from display to action")
    dwell_time: Optional[float] = Field(None, description="Time spent on element")
    interaction_count: int = Field(1, description="Number of interactions")
    

class FeedbackAggregation(BaseModel):
    """Aggregated feedback for a specific target"""
    target_id: str
    target_type: str
    aggregation_period: str  # "hourly", "daily", "weekly"
    period_start: str
    period_end: str
    
    # Metrics
    total_feedback_count: int = 0
    explicit_feedback_count: int = 0
    implicit_feedback_count: int = 0
    
    # Ratings
    average_rating: Optional[float] = None
    rating_distribution: Dict[int, int] = Field(default_factory=dict)
    
    # Sentiment
    sentiment_distribution: Dict[str, int] = Field(default_factory=dict)
    average_sentiment_score: Optional[float] = None
    
    # Behavioral metrics
    average_dwell_time: Optional[float] = None
    interaction_rate: Optional[float] = None
    abandonment_rate: Optional[float] = None
    
    # Common issues/themes
    common_tags: List[Dict[str, Any]] = Field(default_factory=list)
    common_corrections: List[Dict[str, Any]] = Field(default_factory=list)
    

class FeedbackPattern(BaseModel):
    """Identified patterns in feedback"""
    pattern_id: str = Field(default_factory=lambda: f"pat_{uuid.uuid4()}")
    pattern_type: str  # "recurring_issue", "preference_shift", "quality_trend"
    description: str
    
    affected_targets: List[str] = Field(default_factory=list)
    time_range: Dict[str, str]  # {"start": iso_date, "end": iso_date}
    
    # Pattern details
    frequency: int = Field(..., description="How often pattern occurs")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in pattern")
    impact_score: float = Field(..., ge=0, le=1, description="Estimated impact")
    
    # Supporting evidence
    example_feedback_ids: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    

class FeedbackAction(BaseModel):
    """Actions taken based on feedback"""
    action_id: str = Field(default_factory=lambda: f"act_{uuid.uuid4()}")
    pattern_id: Optional[str] = Field(None, description="Pattern that triggered action")
    feedback_ids: List[str] = Field(..., description="Feedback that led to action")
    
    action_type: str  # "agent_retrain", "workflow_update", "ui_change", "alert"
    description: str
    
    # Action details
    target_component: str
    changes_made: Dict[str, Any] = Field(default_factory=dict)
    
    # Tracking
    initiated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    status: str = Field("pending", description="pending, in_progress, completed, failed")
    
    # Impact measurement
    expected_impact: Optional[str] = None
    measured_impact: Optional[Dict[str, Any]] = None
    

class UserPreference(BaseModel):
    """Learned user preferences from feedback"""
    user_id: str
    preference_type: str  # "agent_personality", "response_style", "ui_layout"
    preference_key: str
    preference_value: Any
    
    # Confidence and source
    confidence: float = Field(0.5, ge=0, le=1)
    source_feedback_count: int = 0
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Context
    applicable_contexts: List[str] = Field(default_factory=list)
    

class FeedbackLoop(BaseModel):
    """Complete feedback loop tracking"""
    loop_id: str = Field(default_factory=lambda: f"loop_{uuid.uuid4()}")
    
    # Trigger
    trigger_type: str  # "threshold", "pattern", "manual"
    trigger_details: Dict[str, Any]
    
    # Feedback collection
    feedback_count: int
    feedback_time_range: Dict[str, str]
    
    # Analysis
    patterns_identified: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    
    # Actions
    actions_taken: List[str] = Field(default_factory=list)
    
    # Results
    loop_status: str = Field("open", description="open, closed, monitoring")
    outcome: Optional[str] = None
    metrics_before: Dict[str, Any] = Field(default_factory=dict)
    metrics_after: Dict[str, Any] = Field(default_factory=dict)
    improvement_percentage: Optional[float] = None
    
    # Timestamps
    opened_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    closed_at: Optional[str] = None 