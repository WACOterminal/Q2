"""
Context-Aware Decision Making Service

This service provides intelligent decision-making capabilities for the Q Platform:
- Context understanding and representation
- Multi-criteria decision making (MCDM)
- Temporal reasoning and trend analysis
- Situational awareness and environmental factors
- Adaptive behavior and learning from outcomes
- Machine learning-based decision optimization
- Rule-based decision engines
- Confidence scoring and uncertainty handling
- Decision explanation and interpretability
- Real-time decision support
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import statistics
import math

# ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    sklearn_available = True
except ImportError:
    sklearn_available = False
    logging.warning("Scikit-learn not available - ML decision capabilities will be limited")

try:
    import pandas as pd
    pandas_available = True
except ImportError:
    pandas_available = False
    logging.warning("Pandas not available - data analysis capabilities will be limited")

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Decision types"""
    CLASSIFICATION = "classification"
    OPTIMIZATION = "optimization"
    RANKING = "ranking"
    RESOURCE_ALLOCATION = "resource_allocation"
    SCHEDULING = "scheduling"
    ROUTING = "routing"
    CONFIGURATION = "configuration"
    APPROVAL = "approval"

class ContextType(Enum):
    """Context types"""
    SYSTEM = "system"
    USER = "user"
    TEMPORAL = "temporal"
    ENVIRONMENTAL = "environmental"
    BUSINESS = "business"
    TECHNICAL = "technical"
    SOCIAL = "social"
    ECONOMIC = "economic"

class DecisionStrategy(Enum):
    """Decision strategies"""
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"
    MULTI_CRITERIA = "multi_criteria"
    CONSENSUS = "consensus"
    OPTIMIZATION = "optimization"

class ConfidenceLevel(Enum):
    """Confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class DecisionOutcome(Enum):
    """Decision outcomes"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    PENDING = "pending"
    CANCELLED = "cancelled"

@dataclass
class ContextFactor:
    """Context factor representation"""
    factor_id: str
    name: str
    context_type: ContextType
    value: Any
    weight: float
    confidence: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DecisionCriteria:
    """Decision criteria representation"""
    criteria_id: str
    name: str
    description: str
    weight: float
    target_value: Any
    threshold: Optional[float] = None
    direction: str = "maximize"  # maximize, minimize, target
    mandatory: bool = False
    
    def __post_init__(self):
        if self.direction not in ["maximize", "minimize", "target"]:
            self.direction = "maximize"

@dataclass
class DecisionOption:
    """Decision option representation"""
    option_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    estimated_outcomes: Dict[str, Any]
    feasibility_score: float
    risk_score: float
    cost: float
    benefit: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DecisionRequest:
    """Decision request representation"""
    request_id: str
    decision_type: DecisionType
    context: Dict[str, Any]
    criteria: List[DecisionCriteria]
    options: List[DecisionOption]
    constraints: Dict[str, Any]
    strategy: DecisionStrategy
    user_id: Optional[str] = None
    priority: int = 0
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DecisionResult:
    """Decision result representation"""
    result_id: str
    request_id: str
    selected_option: str
    confidence_score: float
    confidence_level: ConfidenceLevel
    reasoning: str
    criteria_scores: Dict[str, float]
    alternative_options: List[Tuple[str, float]]
    context_factors: List[ContextFactor]
    decision_time: datetime
    expiry_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DecisionFeedback:
    """Decision feedback representation"""
    feedback_id: str
    result_id: str
    outcome: DecisionOutcome
    actual_metrics: Dict[str, Any]
    user_satisfaction: Optional[float] = None
    lessons_learned: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.lessons_learned is None:
            self.lessons_learned = []
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class DecisionRule:
    """Decision rule representation"""
    rule_id: str
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    priority: int
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class DecisionModel:
    """Decision model representation"""
    model_id: str
    name: str
    model_type: str
    decision_type: DecisionType
    features: List[str]
    accuracy: float
    precision: float
    recall: float
    last_trained: datetime
    training_data_size: int
    version: str = "1.0"
    
@dataclass
class ContextProfile:
    """Context profile for decision making"""
    profile_id: str
    name: str
    context_factors: List[ContextFactor]
    temporal_patterns: Dict[str, Any]
    behavioral_patterns: Dict[str, Any]
    preferences: Dict[str, Any]
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

class ContextAwareDecisionService:
    """
    Comprehensive Context-Aware Decision Making Service
    """
    
    def __init__(self, storage_path: str = "context_decisions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Decision management
        self.decision_requests: Dict[str, DecisionRequest] = {}
        self.decision_results: Dict[str, DecisionResult] = {}
        self.decision_feedback: Dict[str, DecisionFeedback] = {}
        
        # Context management
        self.context_profiles: Dict[str, ContextProfile] = {}
        self.context_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Rule engine
        self.decision_rules: Dict[str, DecisionRule] = {}
        self.rule_execution_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # ML models
        self.decision_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, DecisionModel] = {}
        self.feature_scalers: Dict[str, StandardScaler] = {}
        
        # Configuration
        self.config = {
            "default_confidence_threshold": 0.7,
            "max_decision_time": 300,  # seconds
            "context_window": 3600,    # seconds
            "min_training_samples": 100,
            "model_retrain_interval": 86400,  # seconds
            "rule_evaluation_timeout": 10,
            "enable_adaptive_learning": True,
            "enable_explanation_generation": True,
            "uncertainty_threshold": 0.5
        }
        
        # Performance metrics
        self.metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "average_decision_time": 0.0,
            "average_confidence": 0.0,
            "rule_hits": 0,
            "ml_predictions": 0,
            "feedback_collected": 0
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Service integrations
        self.vault_client = VaultClient()
        
        logger.info("Context-Aware Decision Service initialized")
    
    async def initialize(self):
        """Initialize the context-aware decision service"""
        logger.info("Initializing Context-Aware Decision Service")
        
        # Load existing data
        await self._load_decision_data()
        
        # Initialize ML models
        await self._initialize_ml_models()
        
        # Load decision rules
        await self._load_decision_rules()
        
        # Start background tasks
        await self._start_background_tasks()
        
        logger.info("Context-Aware Decision Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the context-aware decision service"""
        logger.info("Shutting down Context-Aware Decision Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Save data
        await self._save_decision_data()
        await self._save_decision_rules()
        
        logger.info("Context-Aware Decision Service shutdown complete")
    
    # ===== DECISION MAKING =====
    
    async def make_decision(self, request: DecisionRequest) -> DecisionResult:
        """Make a context-aware decision"""
        
        start_time = datetime.utcnow()
        
        try:
            # Generate request ID if not provided
            if not request.request_id:
                request.request_id = f"req_{uuid.uuid4().hex[:12]}"
            
            # Store request
            self.decision_requests[request.request_id] = request
            
            # Analyze context
            context_factors = await self._analyze_context(request)
            
            # Apply decision strategy
            if request.strategy == DecisionStrategy.RULE_BASED:
                result = await self._rule_based_decision(request, context_factors)
            elif request.strategy == DecisionStrategy.ML_BASED:
                result = await self._ml_based_decision(request, context_factors)
            elif request.strategy == DecisionStrategy.HYBRID:
                result = await self._hybrid_decision(request, context_factors)
            elif request.strategy == DecisionStrategy.MULTI_CRITERIA:
                result = await self._multi_criteria_decision(request, context_factors)
            elif request.strategy == DecisionStrategy.CONSENSUS:
                result = await self._consensus_decision(request, context_factors)
            elif request.strategy == DecisionStrategy.OPTIMIZATION:
                result = await self._optimization_decision(request, context_factors)
            else:
                result = await self._hybrid_decision(request, context_factors)
            
            # Calculate decision time
            decision_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metrics
            self.metrics["total_decisions"] += 1
            self.metrics["average_decision_time"] = (
                (self.metrics["average_decision_time"] * (self.metrics["total_decisions"] - 1) + decision_time) / 
                self.metrics["total_decisions"]
            )
            self.metrics["average_confidence"] = (
                (self.metrics["average_confidence"] * (self.metrics["total_decisions"] - 1) + result.confidence_score) / 
                self.metrics["total_decisions"]
            )
            
            # Store result
            self.decision_results[result.result_id] = result
            
            # Update context history
            await self._update_context_history(request, result, context_factors)
            
            logger.info(f"Decision made: {result.result_id} for request {request.request_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            raise
    
    async def _analyze_context(self, request: DecisionRequest) -> List[ContextFactor]:
        """Analyze context for decision making"""
        
        context_factors = []
        
        try:
            # System context
            system_context = await self._get_system_context()
            context_factors.extend(system_context)
            
            # User context
            if request.user_id:
                user_context = await self._get_user_context(request.user_id)
                context_factors.extend(user_context)
            
            # Temporal context
            temporal_context = await self._get_temporal_context()
            context_factors.extend(temporal_context)
            
            # Business context
            business_context = await self._get_business_context(request.context)
            context_factors.extend(business_context)
            
            # Historical context
            historical_context = await self._get_historical_context(request.decision_type)
            context_factors.extend(historical_context)
            
            return context_factors
            
        except Exception as e:
            logger.error(f"Error analyzing context: {e}")
            return []
    
    async def _rule_based_decision(self, request: DecisionRequest, context_factors: List[ContextFactor]) -> DecisionResult:
        """Make decision using rule-based approach"""
        
        try:
            # Evaluate rules
            applicable_rules = []
            for rule in self.decision_rules.values():
                if rule.enabled and await self._evaluate_rule(rule, request, context_factors):
                    applicable_rules.append(rule)
            
            if not applicable_rules:
                # No rules match, use default logic
                return await self._default_decision_logic(request, context_factors)
            
            # Sort by priority
            applicable_rules.sort(key=lambda x: x.priority, reverse=True)
            
            # Execute highest priority rule
            selected_rule = applicable_rules[0]
            result = await self._execute_rule_action(selected_rule, request, context_factors)
            
            # Update metrics
            self.metrics["rule_hits"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in rule-based decision: {e}")
            return await self._default_decision_logic(request, context_factors)
    
    async def _ml_based_decision(self, request: DecisionRequest, context_factors: List[ContextFactor]) -> DecisionResult:
        """Make decision using machine learning approach"""
        
        try:
            # Get appropriate model
            model_key = f"{request.decision_type.value}_model"
            if model_key not in self.decision_models:
                # Train model if not exists
                await self._train_decision_model(request.decision_type)
            
            if model_key not in self.decision_models:
                # Fallback to rule-based if no model
                return await self._rule_based_decision(request, context_factors)
            
            model = self.decision_models[model_key]
            scaler = self.feature_scalers.get(model_key)
            
            # Prepare features
            features = await self._prepare_features(request, context_factors)
            
            if scaler:
                features = scaler.transform([features])
            else:
                features = [features]
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                prediction = np.argmax(probabilities)
                confidence = np.max(probabilities)
            else:
                prediction = model.predict(features)[0]
                confidence = 0.8  # Default confidence
            
            # Map prediction to option
            if prediction < len(request.options):
                selected_option = request.options[prediction]
                
                result = DecisionResult(
                    result_id=f"res_{uuid.uuid4().hex[:12]}",
                    request_id=request.request_id,
                    selected_option=selected_option.option_id,
                    confidence_score=float(confidence),
                    confidence_level=self._get_confidence_level(confidence),
                    reasoning=f"ML model prediction with {confidence:.2f} confidence",
                    criteria_scores={},
                    alternative_options=[],
                    context_factors=context_factors,
                    decision_time=datetime.utcnow()
                )
                
                # Update metrics
                self.metrics["ml_predictions"] += 1
                
                return result
            
            # Fallback if prediction doesn't map to option
            return await self._default_decision_logic(request, context_factors)
            
        except Exception as e:
            logger.error(f"Error in ML-based decision: {e}")
            return await self._default_decision_logic(request, context_factors)
    
    async def _hybrid_decision(self, request: DecisionRequest, context_factors: List[ContextFactor]) -> DecisionResult:
        """Make decision using hybrid approach (rules + ML)"""
        
        try:
            # First try rule-based
            rule_result = await self._rule_based_decision(request, context_factors)
            
            # If confidence is high enough, use rule result
            if rule_result.confidence_score >= self.config["default_confidence_threshold"]:
                rule_result.reasoning = f"Rule-based decision: {rule_result.reasoning}"
                return rule_result
            
            # Otherwise, try ML-based
            ml_result = await self._ml_based_decision(request, context_factors)
            
            # Combine results
            if ml_result.confidence_score > rule_result.confidence_score:
                ml_result.reasoning = f"Hybrid decision (ML preferred): {ml_result.reasoning}"
                return ml_result
            else:
                rule_result.reasoning = f"Hybrid decision (Rule preferred): {rule_result.reasoning}"
                return rule_result
            
        except Exception as e:
            logger.error(f"Error in hybrid decision: {e}")
            return await self._default_decision_logic(request, context_factors)
    
    async def _multi_criteria_decision(self, request: DecisionRequest, context_factors: List[ContextFactor]) -> DecisionResult:
        """Make decision using multi-criteria decision making (MCDM)"""
        
        try:
            # Calculate scores for each option against each criterion
            option_scores = {}
            criteria_scores = {}
            
            for option in request.options:
                total_score = 0.0
                option_criteria_scores = {}
                
                for criterion in request.criteria:
                    score = await self._evaluate_criterion(option, criterion, context_factors)
                    weighted_score = score * criterion.weight
                    total_score += weighted_score
                    option_criteria_scores[criterion.criteria_id] = score
                
                option_scores[option.option_id] = total_score
                criteria_scores[option.option_id] = option_criteria_scores
            
            # Select best option
            best_option_id = max(option_scores, key=option_scores.get)
            best_score = option_scores[best_option_id]
            
            # Calculate confidence based on score separation
            scores = list(option_scores.values())
            if len(scores) > 1:
                sorted_scores = sorted(scores, reverse=True)
                confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
            else:
                confidence = 1.0
            
            # Prepare alternative options
            alternative_options = [(oid, score) for oid, score in option_scores.items() if oid != best_option_id]
            alternative_options.sort(key=lambda x: x[1], reverse=True)
            
            result = DecisionResult(
                result_id=f"res_{uuid.uuid4().hex[:12]}",
                request_id=request.request_id,
                selected_option=best_option_id,
                confidence_score=confidence,
                confidence_level=self._get_confidence_level(confidence),
                reasoning=f"Multi-criteria decision with score {best_score:.3f}",
                criteria_scores=criteria_scores[best_option_id],
                alternative_options=alternative_options,
                context_factors=context_factors,
                decision_time=datetime.utcnow()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-criteria decision: {e}")
            return await self._default_decision_logic(request, context_factors)
    
    async def _consensus_decision(self, request: DecisionRequest, context_factors: List[ContextFactor]) -> DecisionResult:
        """Make decision using consensus approach"""
        
        try:
            # Get multiple decision approaches
            approaches = [
                await self._rule_based_decision(request, context_factors),
                await self._ml_based_decision(request, context_factors),
                await self._multi_criteria_decision(request, context_factors)
            ]
            
            # Count votes for each option
            option_votes = defaultdict(int)
            option_confidences = defaultdict(list)
            
            for result in approaches:
                option_votes[result.selected_option] += 1
                option_confidences[result.selected_option].append(result.confidence_score)
            
            # Find consensus
            if option_votes:
                winning_option = max(option_votes, key=option_votes.get)
                vote_count = option_votes[winning_option]
                
                # Calculate consensus confidence
                confidences = option_confidences[winning_option]
                consensus_confidence = statistics.mean(confidences) * (vote_count / len(approaches))
                
                result = DecisionResult(
                    result_id=f"res_{uuid.uuid4().hex[:12]}",
                    request_id=request.request_id,
                    selected_option=winning_option,
                    confidence_score=consensus_confidence,
                    confidence_level=self._get_confidence_level(consensus_confidence),
                    reasoning=f"Consensus decision with {vote_count}/{len(approaches)} votes",
                    criteria_scores={},
                    alternative_options=[],
                    context_factors=context_factors,
                    decision_time=datetime.utcnow()
                )
                
                return result
            
            # Fallback
            return await self._default_decision_logic(request, context_factors)
            
        except Exception as e:
            logger.error(f"Error in consensus decision: {e}")
            return await self._default_decision_logic(request, context_factors)
    
    async def _optimization_decision(self, request: DecisionRequest, context_factors: List[ContextFactor]) -> DecisionResult:
        """Make decision using optimization approach"""
        
        try:
            # Simple optimization using weighted sum
            best_option = None
            best_score = float('-inf')
            
            for option in request.options:
                # Calculate utility score
                utility = option.benefit - option.cost
                
                # Apply risk adjustment
                risk_adjusted_utility = utility * (1 - option.risk_score)
                
                # Apply feasibility weight
                final_score = risk_adjusted_utility * option.feasibility_score
                
                if final_score > best_score:
                    best_score = final_score
                    best_option = option
            
            if best_option:
                # Calculate confidence based on score difference
                scores = []
                for option in request.options:
                    utility = option.benefit - option.cost
                    risk_adjusted_utility = utility * (1 - option.risk_score)
                    final_score = risk_adjusted_utility * option.feasibility_score
                    scores.append(final_score)
                
                if len(scores) > 1:
                    sorted_scores = sorted(scores, reverse=True)
                    confidence = (sorted_scores[0] - sorted_scores[1]) / (sorted_scores[0] + 1e-6)
                else:
                    confidence = 1.0
                
                result = DecisionResult(
                    result_id=f"res_{uuid.uuid4().hex[:12]}",
                    request_id=request.request_id,
                    selected_option=best_option.option_id,
                    confidence_score=confidence,
                    confidence_level=self._get_confidence_level(confidence),
                    reasoning=f"Optimization decision with utility score {best_score:.3f}",
                    criteria_scores={},
                    alternative_options=[],
                    context_factors=context_factors,
                    decision_time=datetime.utcnow()
                )
                
                return result
            
            # Fallback
            return await self._default_decision_logic(request, context_factors)
            
        except Exception as e:
            logger.error(f"Error in optimization decision: {e}")
            return await self._default_decision_logic(request, context_factors)
    
    # ===== CONTEXT ANALYSIS =====
    
    async def _get_system_context(self) -> List[ContextFactor]:
        """Get system context factors"""
        
        try:
            factors = []
            
            # CPU usage
            import psutil
            cpu_usage = psutil.cpu_percent()
            factors.append(ContextFactor(
                factor_id="system_cpu_usage",
                name="CPU Usage",
                context_type=ContextType.SYSTEM,
                value=cpu_usage,
                weight=0.8,
                confidence=0.9,
                timestamp=datetime.utcnow(),
                source="system_monitor"
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            factors.append(ContextFactor(
                factor_id="system_memory_usage",
                name="Memory Usage",
                context_type=ContextType.SYSTEM,
                value=memory.percent,
                weight=0.8,
                confidence=0.9,
                timestamp=datetime.utcnow(),
                source="system_monitor"
            ))
            
            # Current load
            factors.append(ContextFactor(
                factor_id="system_load",
                name="System Load",
                context_type=ContextType.SYSTEM,
                value=len(self.decision_requests),
                weight=0.6,
                confidence=0.8,
                timestamp=datetime.utcnow(),
                source="decision_service"
            ))
            
            return factors
            
        except Exception as e:
            logger.error(f"Error getting system context: {e}")
            return []
    
    async def _get_user_context(self, user_id: str) -> List[ContextFactor]:
        """Get user context factors"""
        
        try:
            factors = []
            
            # User activity level
            factors.append(ContextFactor(
                factor_id="user_activity_level",
                name="User Activity Level",
                context_type=ContextType.USER,
                value="high",  # Simplified
                weight=0.7,
                confidence=0.7,
                timestamp=datetime.utcnow(),
                source="user_tracking"
            ))
            
            # User preferences
            factors.append(ContextFactor(
                factor_id="user_preferences",
                name="User Preferences",
                context_type=ContextType.USER,
                value={"risk_tolerance": "medium", "response_time": "fast"},
                weight=0.8,
                confidence=0.8,
                timestamp=datetime.utcnow(),
                source="user_profile"
            ))
            
            return factors
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return []
    
    async def _get_temporal_context(self) -> List[ContextFactor]:
        """Get temporal context factors"""
        
        try:
            factors = []
            now = datetime.utcnow()
            
            # Time of day
            factors.append(ContextFactor(
                factor_id="time_of_day",
                name="Time of Day",
                context_type=ContextType.TEMPORAL,
                value=now.hour,
                weight=0.6,
                confidence=1.0,
                timestamp=now,
                source="system_clock"
            ))
            
            # Day of week
            factors.append(ContextFactor(
                factor_id="day_of_week",
                name="Day of Week",
                context_type=ContextType.TEMPORAL,
                value=now.weekday(),
                weight=0.5,
                confidence=1.0,
                timestamp=now,
                source="system_clock"
            ))
            
            # Business hours
            is_business_hours = 9 <= now.hour <= 17 and now.weekday() < 5
            factors.append(ContextFactor(
                factor_id="business_hours",
                name="Business Hours",
                context_type=ContextType.TEMPORAL,
                value=is_business_hours,
                weight=0.7,
                confidence=1.0,
                timestamp=now,
                source="business_calendar"
            ))
            
            return factors
            
        except Exception as e:
            logger.error(f"Error getting temporal context: {e}")
            return []
    
    async def _get_business_context(self, context_data: Dict[str, Any]) -> List[ContextFactor]:
        """Get business context factors"""
        
        try:
            factors = []
            
            # Business priority
            priority = context_data.get("business_priority", "medium")
            factors.append(ContextFactor(
                factor_id="business_priority",
                name="Business Priority",
                context_type=ContextType.BUSINESS,
                value=priority,
                weight=0.9,
                confidence=0.8,
                timestamp=datetime.utcnow(),
                source="business_rules"
            ))
            
            # Budget constraints
            budget = context_data.get("budget_available", 10000)
            factors.append(ContextFactor(
                factor_id="budget_available",
                name="Budget Available",
                context_type=ContextType.BUSINESS,
                value=budget,
                weight=0.8,
                confidence=0.7,
                timestamp=datetime.utcnow(),
                source="financial_system"
            ))
            
            return factors
            
        except Exception as e:
            logger.error(f"Error getting business context: {e}")
            return []
    
    async def _get_historical_context(self, decision_type: DecisionType) -> List[ContextFactor]:
        """Get historical context factors"""
        
        try:
            factors = []
            
            # Historical success rate
            success_rate = await self._calculate_historical_success_rate(decision_type)
            factors.append(ContextFactor(
                factor_id="historical_success_rate",
                name="Historical Success Rate",
                context_type=ContextType.BUSINESS,
                value=success_rate,
                weight=0.6,
                confidence=0.8,
                timestamp=datetime.utcnow(),
                source="decision_history"
            ))
            
            # Recent trends
            factors.append(ContextFactor(
                factor_id="recent_trends",
                name="Recent Decision Trends",
                context_type=ContextType.BUSINESS,
                value="stable",  # Simplified
                weight=0.5,
                confidence=0.6,
                timestamp=datetime.utcnow(),
                source="trend_analysis"
            ))
            
            return factors
            
        except Exception as e:
            logger.error(f"Error getting historical context: {e}")
            return []
    
    # ===== UTILITY METHODS =====
    
    def _get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    async def _default_decision_logic(self, request: DecisionRequest, context_factors: List[ContextFactor]) -> DecisionResult:
        """Default decision logic when other methods fail"""
        
        try:
            # Simple heuristic: choose option with highest benefit-to-cost ratio
            best_option = None
            best_ratio = float('-inf')
            
            for option in request.options:
                ratio = option.benefit / max(option.cost, 1.0)  # Avoid division by zero
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_option = option
            
            if best_option:
                result = DecisionResult(
                    result_id=f"res_{uuid.uuid4().hex[:12]}",
                    request_id=request.request_id,
                    selected_option=best_option.option_id,
                    confidence_score=0.5,  # Default confidence
                    confidence_level=ConfidenceLevel.MEDIUM,
                    reasoning="Default decision logic - highest benefit-to-cost ratio",
                    criteria_scores={},
                    alternative_options=[],
                    context_factors=context_factors,
                    decision_time=datetime.utcnow()
                )
                
                return result
            
            # Last resort: choose first option
            if request.options:
                result = DecisionResult(
                    result_id=f"res_{uuid.uuid4().hex[:12]}",
                    request_id=request.request_id,
                    selected_option=request.options[0].option_id,
                    confidence_score=0.3,  # Low confidence
                    confidence_level=ConfidenceLevel.LOW,
                    reasoning="Default decision logic - first available option",
                    criteria_scores={},
                    alternative_options=[],
                    context_factors=context_factors,
                    decision_time=datetime.utcnow()
                )
                
                return result
            
            raise Exception("No options available for decision")
            
        except Exception as e:
            logger.error(f"Error in default decision logic: {e}")
            raise
    
    async def _calculate_historical_success_rate(self, decision_type: DecisionType) -> float:
        """Calculate historical success rate for a decision type"""
        
        try:
            # Filter feedback by decision type
            relevant_feedback = []
            for feedback in self.decision_feedback.values():
                if feedback.result_id in self.decision_results:
                    result = self.decision_results[feedback.result_id]
                    if result.request_id in self.decision_requests:
                        request = self.decision_requests[result.request_id]
                        if request.decision_type == decision_type:
                            relevant_feedback.append(feedback)
            
            if not relevant_feedback:
                return 0.5  # Default success rate
            
            # Calculate success rate
            successful = sum(1 for f in relevant_feedback if f.outcome == DecisionOutcome.SUCCESS)
            return successful / len(relevant_feedback)
            
        except Exception as e:
            logger.error(f"Error calculating historical success rate: {e}")
            return 0.5
    
    async def _evaluate_rule(self, rule: DecisionRule, request: DecisionRequest, context_factors: List[ContextFactor]) -> bool:
        """Evaluate if a rule applies to the current decision"""
        
        try:
            # Simple rule evaluation
            for condition in rule.conditions:
                condition_type = condition.get("type")
                
                if condition_type == "context":
                    # Check context condition
                    factor_name = condition.get("factor_name")
                    operator = condition.get("operator", "equals")
                    value = condition.get("value")
                    
                    matching_factors = [f for f in context_factors if f.name == factor_name]
                    if not matching_factors:
                        return False
                    
                    factor_value = matching_factors[0].value
                    
                    if operator == "equals" and factor_value != value:
                        return False
                    elif operator == "greater_than" and factor_value <= value:
                        return False
                    elif operator == "less_than" and factor_value >= value:
                        return False
                    elif operator == "contains" and value not in str(factor_value):
                        return False
                
                elif condition_type == "decision_type":
                    # Check decision type condition
                    if request.decision_type.value != condition.get("value"):
                        return False
                
                elif condition_type == "user":
                    # Check user condition
                    if request.user_id != condition.get("value"):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating rule: {e}")
            return False
    
    async def _execute_rule_action(self, rule: DecisionRule, request: DecisionRequest, context_factors: List[ContextFactor]) -> DecisionResult:
        """Execute rule action"""
        
        try:
            # Simple rule action execution
            for action in rule.actions:
                action_type = action.get("type")
                
                if action_type == "select_option":
                    option_id = action.get("option_id")
                    confidence = action.get("confidence", 0.8)
                    
                    # Find the option
                    selected_option = None
                    for option in request.options:
                        if option.option_id == option_id:
                            selected_option = option
                            break
                    
                    if selected_option:
                        result = DecisionResult(
                            result_id=f"res_{uuid.uuid4().hex[:12]}",
                            request_id=request.request_id,
                            selected_option=option_id,
                            confidence_score=confidence,
                            confidence_level=self._get_confidence_level(confidence),
                            reasoning=f"Rule-based decision: {rule.name}",
                            criteria_scores={},
                            alternative_options=[],
                            context_factors=context_factors,
                            decision_time=datetime.utcnow()
                        )
                        
                        return result
                
                elif action_type == "select_best_option":
                    # Select option with highest benefit
                    best_option = max(request.options, key=lambda x: x.benefit)
                    confidence = action.get("confidence", 0.8)
                    
                    result = DecisionResult(
                        result_id=f"res_{uuid.uuid4().hex[:12]}",
                        request_id=request.request_id,
                        selected_option=best_option.option_id,
                        confidence_score=confidence,
                        confidence_level=self._get_confidence_level(confidence),
                        reasoning=f"Rule-based decision: {rule.name} - selected best option",
                        criteria_scores={},
                        alternative_options=[],
                        context_factors=context_factors,
                        decision_time=datetime.utcnow()
                    )
                    
                    return result
            
            # Default action if no specific action
            return await self._default_decision_logic(request, context_factors)
            
        except Exception as e:
            logger.error(f"Error executing rule action: {e}")
            return await self._default_decision_logic(request, context_factors)
    
    async def _evaluate_criterion(self, option: DecisionOption, criterion: DecisionCriteria, context_factors: List[ContextFactor]) -> float:
        """Evaluate an option against a criterion"""
        
        try:
            # Simple criterion evaluation
            if criterion.criteria_id == "cost":
                if criterion.direction == "minimize":
                    return 1.0 - (option.cost / max(opt.cost for opt in [option]))
                else:
                    return option.cost / max(opt.cost for opt in [option])
            
            elif criterion.criteria_id == "benefit":
                if criterion.direction == "maximize":
                    return option.benefit / max(opt.benefit for opt in [option])
                else:
                    return 1.0 - (option.benefit / max(opt.benefit for opt in [option]))
            
            elif criterion.criteria_id == "risk":
                if criterion.direction == "minimize":
                    return 1.0 - option.risk_score
                else:
                    return option.risk_score
            
            elif criterion.criteria_id == "feasibility":
                return option.feasibility_score
            
            # Default evaluation
            return 0.5
            
        except Exception as e:
            logger.error(f"Error evaluating criterion: {e}")
            return 0.5
    
    async def _prepare_features(self, request: DecisionRequest, context_factors: List[ContextFactor]) -> List[float]:
        """Prepare features for ML model"""
        
        try:
            features = []
            
            # Basic features
            features.append(len(request.options))
            features.append(len(request.criteria))
            features.append(request.priority)
            
            # Context features
            system_factors = [f for f in context_factors if f.context_type == ContextType.SYSTEM]
            user_factors = [f for f in context_factors if f.context_type == ContextType.USER]
            temporal_factors = [f for f in context_factors if f.context_type == ContextType.TEMPORAL]
            
            features.append(len(system_factors))
            features.append(len(user_factors))
            features.append(len(temporal_factors))
            
            # Aggregate option features
            if request.options:
                avg_cost = statistics.mean(opt.cost for opt in request.options)
                avg_benefit = statistics.mean(opt.benefit for opt in request.options)
                avg_risk = statistics.mean(opt.risk_score for opt in request.options)
                avg_feasibility = statistics.mean(opt.feasibility_score for opt in request.options)
                
                features.extend([avg_cost, avg_benefit, avg_risk, avg_feasibility])
            else:
                features.extend([0, 0, 0, 0])
            
            # Pad or truncate to fixed size
            target_size = 20
            if len(features) < target_size:
                features.extend([0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return [0] * 20
    
    # ===== MACHINE LEARNING =====
    
    async def _initialize_ml_models(self):
        """Initialize ML models"""
        
        try:
            if not sklearn_available:
                return
            
            # Initialize models for different decision types
            for decision_type in DecisionType:
                model_key = f"{decision_type.value}_model"
                
                # Check if we have training data
                if await self._has_training_data(decision_type):
                    await self._train_decision_model(decision_type)
            
            logger.info("ML models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    async def _train_decision_model(self, decision_type: DecisionType):
        """Train ML model for a specific decision type"""
        
        try:
            if not sklearn_available:
                return
            
            # Collect training data
            training_data = await self._collect_training_data(decision_type)
            
            if len(training_data) < self.config["min_training_samples"]:
                logger.warning(f"Not enough training data for {decision_type.value}")
                return
            
            # Prepare features and labels
            X = []
            y = []
            
            for data_point in training_data:
                features = data_point["features"]
                outcome = data_point["outcome"]
                
                X.append(features)
                y.append(outcome)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Store model
            model_key = f"{decision_type.value}_model"
            self.decision_models[model_key] = model
            self.feature_scalers[model_key] = scaler
            
            # Store metadata
            self.model_metadata[model_key] = DecisionModel(
                model_id=model_key,
                name=f"{decision_type.value} Decision Model",
                model_type="RandomForest",
                decision_type=decision_type,
                features=["option_count", "criteria_count", "priority", "context_factors"],
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                last_trained=datetime.utcnow(),
                training_data_size=len(training_data)
            )
            
            logger.info(f"Trained model for {decision_type.value} with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error training decision model: {e}")
    
    async def _has_training_data(self, decision_type: DecisionType) -> bool:
        """Check if we have sufficient training data"""
        
        try:
            # Count relevant feedback
            count = 0
            for feedback in self.decision_feedback.values():
                if feedback.result_id in self.decision_results:
                    result = self.decision_results[feedback.result_id]
                    if result.request_id in self.decision_requests:
                        request = self.decision_requests[result.request_id]
                        if request.decision_type == decision_type:
                            count += 1
            
            return count >= self.config["min_training_samples"]
            
        except Exception as e:
            logger.error(f"Error checking training data: {e}")
            return False
    
    async def _collect_training_data(self, decision_type: DecisionType) -> List[Dict[str, Any]]:
        """Collect training data for a decision type"""
        
        try:
            training_data = []
            
            for feedback in self.decision_feedback.values():
                if feedback.result_id in self.decision_results:
                    result = self.decision_results[feedback.result_id]
                    if result.request_id in self.decision_requests:
                        request = self.decision_requests[result.request_id]
                        if request.decision_type == decision_type:
                            # Prepare features
                            features = await self._prepare_features(request, result.context_factors)
                            
                            # Determine outcome
                            outcome = 1 if feedback.outcome == DecisionOutcome.SUCCESS else 0
                            
                            training_data.append({
                                "features": features,
                                "outcome": outcome,
                                "request_id": request.request_id,
                                "result_id": result.result_id,
                                "feedback_id": feedback.feedback_id
                            })
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return []
    
    # ===== FEEDBACK AND LEARNING =====
    
    async def record_feedback(self, feedback: DecisionFeedback) -> bool:
        """Record feedback for a decision"""
        
        try:
            # Store feedback
            self.decision_feedback[feedback.feedback_id] = feedback
            
            # Update metrics
            self.metrics["feedback_collected"] += 1
            
            if feedback.outcome == DecisionOutcome.SUCCESS:
                self.metrics["successful_decisions"] += 1
            
            # Trigger adaptive learning if enabled
            if self.config["enable_adaptive_learning"]:
                await self._trigger_adaptive_learning(feedback)
            
            logger.info(f"Recorded feedback: {feedback.feedback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
    
    async def _trigger_adaptive_learning(self, feedback: DecisionFeedback):
        """Trigger adaptive learning based on feedback"""
        
        try:
            # Get the decision details
            if feedback.result_id not in self.decision_results:
                return
            
            result = self.decision_results[feedback.result_id]
            if result.request_id not in self.decision_requests:
                return
            
            request = self.decision_requests[result.request_id]
            
            # Update context profiles
            await self._update_context_profiles(request, result, feedback)
            
            # Retrain models if needed
            if await self._should_retrain_model(request.decision_type):
                await self._train_decision_model(request.decision_type)
            
        except Exception as e:
            logger.error(f"Error in adaptive learning: {e}")
    
    async def _update_context_profiles(self, request: DecisionRequest, result: DecisionResult, feedback: DecisionFeedback):
        """Update context profiles based on feedback"""
        
        try:
            # Update context factor weights based on outcome
            for factor in result.context_factors:
                # Increase weight if decision was successful
                if feedback.outcome == DecisionOutcome.SUCCESS:
                    factor.weight = min(1.0, factor.weight * 1.1)
                else:
                    factor.weight = max(0.1, factor.weight * 0.9)
            
        except Exception as e:
            logger.error(f"Error updating context profiles: {e}")
    
    async def _should_retrain_model(self, decision_type: DecisionType) -> bool:
        """Check if model should be retrained"""
        
        try:
            model_key = f"{decision_type.value}_model"
            
            if model_key not in self.model_metadata:
                return True
            
            model_meta = self.model_metadata[model_key]
            
            # Check if enough time has passed
            time_since_training = (datetime.utcnow() - model_meta.last_trained).total_seconds()
            if time_since_training >= self.config["model_retrain_interval"]:
                return True
            
            # Check if we have significantly more data
            current_data_size = len(await self._collect_training_data(decision_type))
            if current_data_size > model_meta.training_data_size * 1.5:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if model should be retrained: {e}")
            return False
    
    # ===== CONTEXT MANAGEMENT =====
    
    async def _update_context_history(self, request: DecisionRequest, result: DecisionResult, context_factors: List[ContextFactor]):
        """Update context history"""
        
        try:
            # Store context snapshot
            context_snapshot = {
                "request_id": request.request_id,
                "result_id": result.result_id,
                "timestamp": datetime.utcnow().isoformat(),
                "decision_type": request.decision_type.value,
                "context_factors": [asdict(factor) for factor in context_factors]
            }
            
            # Add to history
            history_key = f"{request.decision_type.value}_context"
            self.context_history[history_key].append(context_snapshot)
            
        except Exception as e:
            logger.error(f"Error updating context history: {e}")
    
    # ===== BACKGROUND TASKS =====
    
    async def _start_background_tasks(self):
        """Start background tasks"""
        
        try:
            # Model retraining task
            task = asyncio.create_task(self._model_retraining_loop())
            self.background_tasks.add(task)
            
            # Context analysis task
            task = asyncio.create_task(self._context_analysis_loop())
            self.background_tasks.add(task)
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def _model_retraining_loop(self):
        """Background model retraining loop"""
        
        while True:
            try:
                await asyncio.sleep(self.config["model_retrain_interval"])
                
                # Check each decision type
                for decision_type in DecisionType:
                    if await self._should_retrain_model(decision_type):
                        await self._train_decision_model(decision_type)
                
            except Exception as e:
                logger.error(f"Error in model retraining loop: {e}")
    
    async def _context_analysis_loop(self):
        """Background context analysis loop"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Analyze context patterns
                await self._analyze_context_patterns()
                
            except Exception as e:
                logger.error(f"Error in context analysis loop: {e}")
    
    async def _analyze_context_patterns(self):
        """Analyze context patterns for insights"""
        
        try:
            # Analyze temporal patterns
            for context_type in ContextType:
                await self._analyze_temporal_patterns(context_type)
            
        except Exception as e:
            logger.error(f"Error analyzing context patterns: {e}")
    
    async def _analyze_temporal_patterns(self, context_type: ContextType):
        """Analyze temporal patterns for a context type"""
        
        try:
            # Simple temporal analysis
            # In a full implementation, this would do sophisticated pattern analysis
            pass
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
    
    # ===== PERSISTENCE =====
    
    async def _load_decision_data(self):
        """Load decision data from storage"""
        
        try:
            # Load decision requests
            requests_file = self.storage_path / "decision_requests.json"
            if requests_file.exists():
                with open(requests_file, 'r') as f:
                    requests_data = json.load(f)
                    for req_data in requests_data:
                        request = DecisionRequest(**req_data)
                        self.decision_requests[request.request_id] = request
            
            # Load decision results
            results_file = self.storage_path / "decision_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                    for res_data in results_data:
                        result = DecisionResult(**res_data)
                        self.decision_results[result.result_id] = result
            
            # Load feedback
            feedback_file = self.storage_path / "decision_feedback.json"
            if feedback_file.exists():
                with open(feedback_file, 'r') as f:
                    feedback_data = json.load(f)
                    for fb_data in feedback_data:
                        feedback = DecisionFeedback(**fb_data)
                        self.decision_feedback[feedback.feedback_id] = feedback
            
            logger.info("Decision data loaded from storage")
            
        except Exception as e:
            logger.warning(f"Error loading decision data: {e}")
    
    async def _save_decision_data(self):
        """Save decision data to storage"""
        
        try:
            # Save decision requests
            requests_data = []
            for request in self.decision_requests.values():
                requests_data.append(asdict(request))
            
            requests_file = self.storage_path / "decision_requests.json"
            with open(requests_file, 'w') as f:
                json.dump(requests_data, f, indent=2, default=str)
            
            # Save decision results
            results_data = []
            for result in self.decision_results.values():
                results_data.append(asdict(result))
            
            results_file = self.storage_path / "decision_results.json"
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            # Save feedback
            feedback_data = []
            for feedback in self.decision_feedback.values():
                feedback_data.append(asdict(feedback))
            
            feedback_file = self.storage_path / "decision_feedback.json"
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2, default=str)
            
            logger.info("Decision data saved to storage")
            
        except Exception as e:
            logger.warning(f"Error saving decision data: {e}")
    
    async def _load_decision_rules(self):
        """Load decision rules from storage"""
        
        try:
            rules_file = self.storage_path / "decision_rules.json"
            if rules_file.exists():
                with open(rules_file, 'r') as f:
                    rules_data = json.load(f)
                    for rule_data in rules_data:
                        rule = DecisionRule(**rule_data)
                        self.decision_rules[rule.rule_id] = rule
            
            logger.info("Decision rules loaded from storage")
            
        except Exception as e:
            logger.warning(f"Error loading decision rules: {e}")
    
    async def _save_decision_rules(self):
        """Save decision rules to storage"""
        
        try:
            rules_data = []
            for rule in self.decision_rules.values():
                rules_data.append(asdict(rule))
            
            rules_file = self.storage_path / "decision_rules.json"
            with open(rules_file, 'w') as f:
                json.dump(rules_data, f, indent=2, default=str)
            
            logger.info("Decision rules saved to storage")
            
        except Exception as e:
            logger.warning(f"Error saving decision rules: {e}")
    
    # ===== API METHODS =====
    
    async def get_decision_metrics(self) -> Dict[str, Any]:
        """Get decision metrics"""
        
        return {
            "total_decisions": self.metrics["total_decisions"],
            "successful_decisions": self.metrics["successful_decisions"],
            "success_rate": self.metrics["successful_decisions"] / max(self.metrics["total_decisions"], 1),
            "average_decision_time": self.metrics["average_decision_time"],
            "average_confidence": self.metrics["average_confidence"],
            "rule_hits": self.metrics["rule_hits"],
            "ml_predictions": self.metrics["ml_predictions"],
            "feedback_collected": self.metrics["feedback_collected"],
            "total_rules": len(self.decision_rules),
            "active_models": len(self.decision_models)
        }
    
    async def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get decision history"""
        
        history = []
        for result in list(self.decision_results.values())[-limit:]:
            request = self.decision_requests.get(result.request_id)
            feedback = None
            for fb in self.decision_feedback.values():
                if fb.result_id == result.result_id:
                    feedback = fb
                    break
            
            history.append({
                "result_id": result.result_id,
                "request_id": result.request_id,
                "decision_type": request.decision_type.value if request else "unknown",
                "selected_option": result.selected_option,
                "confidence_score": result.confidence_score,
                "confidence_level": result.confidence_level.value,
                "decision_time": result.decision_time.isoformat(),
                "outcome": feedback.outcome.value if feedback else "unknown",
                "reasoning": result.reasoning
            })
        
        return history
    
    async def create_decision_rule(self, rule: DecisionRule) -> bool:
        """Create a decision rule"""
        
        try:
            self.decision_rules[rule.rule_id] = rule
            return True
            
        except Exception as e:
            logger.error(f"Error creating decision rule: {e}")
            return False
    
    async def update_decision_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update a decision rule"""
        
        try:
            if rule_id not in self.decision_rules:
                return False
            
            rule = self.decision_rules[rule_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(rule, field):
                    setattr(rule, field, value)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating decision rule: {e}")
            return False
    
    async def delete_decision_rule(self, rule_id: str) -> bool:
        """Delete a decision rule"""
        
        try:
            if rule_id in self.decision_rules:
                del self.decision_rules[rule_id]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting decision rule: {e}")
            return False

# Global service instance
context_aware_decision_service = ContextAwareDecisionService() 