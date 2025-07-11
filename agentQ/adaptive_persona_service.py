"""
Adaptive Persona Service for Q Platform

This service provides ML-driven adaptive persona switching capabilities:
- Context-aware persona selection
- Performance-based persona optimization
- Dynamic role adaptation
- Persona learning and evolution
- Multi-dimensional persona modeling
- Real-time persona switching
- Persona performance analytics
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)

class PersonaType(Enum):
    """Persona types for agents"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    DIPLOMATIC = "diplomatic"
    ASSERTIVE = "assertive"
    EMPATHETIC = "empathetic"
    TECHNICAL = "technical"
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    COLLABORATIVE = "collaborative"
    INDEPENDENT = "independent"

class PersonaIntensity(Enum):
    """Persona intensity levels"""
    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"
    DOMINANT = "dominant"

class ContextType(Enum):
    """Context types for persona selection"""
    TASK_DOMAIN = "task_domain"
    USER_INTERACTION = "user_interaction"
    TEAM_COLLABORATION = "team_collaboration"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    EMERGENCY = "emergency"

class AdaptationTrigger(Enum):
    """Triggers for persona adaptation"""
    PERFORMANCE_DECLINE = "performance_decline"
    CONTEXT_CHANGE = "context_change"
    USER_FEEDBACK = "user_feedback"
    TASK_REQUIREMENT = "task_requirement"
    TEAM_DYNAMICS = "team_dynamics"
    LEARNING_OPPORTUNITY = "learning_opportunity"
    SCHEDULED_REVIEW = "scheduled_review"

@dataclass
class PersonaProfile:
    """Detailed persona profile definition"""
    persona_id: str
    persona_name: str
    persona_type: PersonaType
    intensity: PersonaIntensity
    description: str
    characteristics: Dict[str, float]
    behavioral_patterns: Dict[str, Any]
    communication_style: Dict[str, Any]
    decision_making_style: Dict[str, Any]
    learning_preferences: Dict[str, Any]
    interaction_preferences: Dict[str, Any]
    strengths: List[str]
    weaknesses: List[str]
    optimal_contexts: List[ContextType]
    created_at: datetime
    usage_count: int = 0
    success_rate: float = 0.0
    average_performance: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class PersonaInstance:
    """Active persona instance for an agent"""
    instance_id: str
    agent_id: str
    persona_id: str
    activation_time: datetime
    context: Dict[str, Any]
    adaptation_reason: AdaptationTrigger
    expected_duration: int
    performance_metrics: Dict[str, float]
    interaction_history: List[Dict[str, Any]]
    status: str = "active"
    deactivation_time: Optional[datetime] = None

@dataclass
class ContextState:
    """Current context state for persona selection"""
    context_id: str
    agent_id: str
    context_type: ContextType
    features: Dict[str, Any]
    importance_weights: Dict[str, float]
    temporal_features: Dict[str, Any]
    environmental_features: Dict[str, Any]
    task_features: Dict[str, Any]
    user_features: Dict[str, Any]
    team_features: Dict[str, Any]
    timestamp: datetime
    confidence_score: float = 0.0

@dataclass
class PersonaPerformanceMetrics:
    """Performance metrics for persona evaluation"""
    persona_id: str
    agent_id: str
    context_type: ContextType
    task_completion_rate: float
    user_satisfaction: float
    response_quality: float
    collaboration_effectiveness: float
    learning_progress: float
    adaptability_score: float
    consistency_score: float
    innovation_score: float
    measurement_period: Tuple[datetime, datetime]
    sample_size: int
    computed_at: datetime

# Neural Network Models for Persona Selection and Adaptation

class PersonaSelector(nn.Module):
    """Neural network for persona selection"""
    
    def __init__(self, context_dim: int, num_personas: int, hidden_dim: int = 128):
        super(PersonaSelector, self).__init__()
        self.context_dim = context_dim
        self.num_personas = num_personas
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Persona scoring
        self.persona_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_personas),
            nn.Softmax(dim=1)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, context_features):
        encoded = self.context_encoder(context_features)
        persona_scores = self.persona_scorer(encoded)
        confidence = self.confidence_estimator(encoded)
        
        return persona_scores, confidence

class PersonaAdaptationNet(nn.Module):
    """Neural network for persona adaptation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PersonaAdaptationNet, self).__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value function
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=1)
        )
        
    def forward(self, state):
        encoded = self.state_encoder(state)
        value = self.value_head(encoded)
        policy = self.policy_head(encoded)
        
        return value, policy

class PersonaEvolutionNet(nn.Module):
    """Neural network for persona evolution"""
    
    def __init__(self, persona_dim: int, performance_dim: int, hidden_dim: int = 96):
        super(PersonaEvolutionNet, self).__init__()
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(persona_dim + performance_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Evolution predictor
        self.evolution_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, persona_dim),
            nn.Tanh()
        )
        
    def forward(self, persona_features, performance_features):
        combined = torch.cat([persona_features, performance_features], dim=1)
        fused = self.fusion_layer(combined)
        evolution = self.evolution_predictor(fused)
        
        return evolution

class AdaptivePersonaService:
    """
    Adaptive Persona Service for ML-driven persona switching
    """
    
    def __init__(self, 
                 context_dim: int = 128,
                 persona_dim: int = 64,
                 max_personas: int = 20,
                 kg_client: Optional[KnowledgeGraphClient] = None):
        
        self.context_dim = context_dim
        self.persona_dim = persona_dim
        self.max_personas = max_personas
        self.kg_client = kg_client or KnowledgeGraphClient()
        
        # Persona management
        self.persona_profiles: Dict[str, PersonaProfile] = {}
        self.persona_instances: Dict[str, PersonaInstance] = {}
        self.context_states: Dict[str, ContextState] = {}
        self.performance_metrics: Dict[str, List[PersonaPerformanceMetrics]] = defaultdict(list)
        
        # Agent persona mapping
        self.agent_current_personas: Dict[str, str] = {}  # agent_id -> persona_instance_id
        self.agent_persona_history: Dict[str, List[str]] = defaultdict(list)
        
        # ML models
        self.persona_selector = PersonaSelector(context_dim, max_personas)
        self.adaptation_net = PersonaAdaptationNet(context_dim + persona_dim, max_personas)
        self.evolution_net = PersonaEvolutionNet(persona_dim, 10)  # 10 performance metrics
        
        # Optimizers
        self.selector_optimizer = optim.Adam(self.persona_selector.parameters(), lr=0.001)
        self.adaptation_optimizer = optim.Adam(self.adaptation_net.parameters(), lr=0.0005)
        self.evolution_optimizer = optim.Adam(self.evolution_net.parameters(), lr=0.0001)
        
        # Experience replay
        self.persona_experience_buffer: deque = deque(maxlen=10000)
        self.adaptation_experience_buffer: deque = deque(maxlen=5000)
        
        # Configuration
        self.config = {
            "adaptation_threshold": 0.7,
            "performance_window": 3600,  # seconds
            "min_activation_duration": 300,  # seconds
            "max_adaptation_frequency": 5,  # per hour
            "context_update_frequency": 60,  # seconds
            "learning_rate_decay": 0.995,
            "performance_weight": 0.4,
            "context_weight": 0.3,
            "user_feedback_weight": 0.3,
            "batch_size": 32,
            "train_frequency": 100
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Performance metrics
        self.service_metrics = {
            "total_personas": 0,
            "active_instances": 0,
            "adaptations_per_hour": 0.0,
            "average_adaptation_accuracy": 0.0,
            "persona_utilization": 0.0,
            "context_recognition_accuracy": 0.0,
            "user_satisfaction": 0.0,
            "system_performance": 0.0
        }
        
        # Initialize built-in personas
        self._initialize_builtin_personas()
        
    async def initialize(self):
        """Initialize the adaptive persona service"""
        logger.info("Initializing Adaptive Persona Service")
        
        # Initialize KnowledgeGraph client
        await self.kg_client.initialize()
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        # Start background tasks
        self.background_tasks.add(asyncio.create_task(self._context_monitoring()))
        self.background_tasks.add(asyncio.create_task(self._adaptation_monitoring()))
        self.background_tasks.add(asyncio.create_task(self._performance_tracking()))
        self.background_tasks.add(asyncio.create_task(self._persona_evolution()))
        self.background_tasks.add(asyncio.create_task(self._learning_loop()))
        
        logger.info("Adaptive Persona Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the adaptive persona service"""
        logger.info("Shutting down Adaptive Persona Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Adaptive Persona Service shut down successfully")
    
    def _initialize_builtin_personas(self):
        """Initialize built-in persona profiles"""
        
        builtin_personas = [
            {
                "persona_type": PersonaType.ANALYTICAL,
                "characteristics": {
                    "logical_reasoning": 0.9,
                    "attention_to_detail": 0.8,
                    "systematic_approach": 0.9,
                    "creativity": 0.3,
                    "emotional_intelligence": 0.4,
                    "social_skills": 0.4,
                    "risk_tolerance": 0.3,
                    "adaptability": 0.5
                },
                "optimal_contexts": [ContextType.PROBLEM_SOLVING, ContextType.DECISION_MAKING],
                "strengths": ["Data analysis", "Pattern recognition", "Systematic thinking"],
                "weaknesses": ["Limited creativity", "May overthink simple problems"]
            },
            {
                "persona_type": PersonaType.CREATIVE,
                "characteristics": {
                    "logical_reasoning": 0.5,
                    "attention_to_detail": 0.4,
                    "systematic_approach": 0.3,
                    "creativity": 0.9,
                    "emotional_intelligence": 0.7,
                    "social_skills": 0.7,
                    "risk_tolerance": 0.8,
                    "adaptability": 0.8
                },
                "optimal_contexts": [ContextType.PROBLEM_SOLVING, ContextType.LEARNING],
                "strengths": ["Innovation", "Brainstorming", "Alternative solutions"],
                "weaknesses": ["May lack structure", "Can be inconsistent"]
            },
            {
                "persona_type": PersonaType.DIPLOMATIC,
                "characteristics": {
                    "logical_reasoning": 0.6,
                    "attention_to_detail": 0.6,
                    "systematic_approach": 0.5,
                    "creativity": 0.6,
                    "emotional_intelligence": 0.9,
                    "social_skills": 0.9,
                    "risk_tolerance": 0.4,
                    "adaptability": 0.7
                },
                "optimal_contexts": [ContextType.COMMUNICATION, ContextType.TEAM_COLLABORATION],
                "strengths": ["Conflict resolution", "Consensus building", "Relationship management"],
                "weaknesses": ["May avoid difficult decisions", "Can be slow to act"]
            },
            {
                "persona_type": PersonaType.ASSERTIVE,
                "characteristics": {
                    "logical_reasoning": 0.7,
                    "attention_to_detail": 0.6,
                    "systematic_approach": 0.7,
                    "creativity": 0.5,
                    "emotional_intelligence": 0.5,
                    "social_skills": 0.6,
                    "risk_tolerance": 0.8,
                    "adaptability": 0.6
                },
                "optimal_contexts": [ContextType.DECISION_MAKING, ContextType.EMERGENCY],
                "strengths": ["Quick decision making", "Clear communication", "Goal-oriented"],
                "weaknesses": ["May be perceived as aggressive", "Less collaborative"]
            },
            {
                "persona_type": PersonaType.EMPATHETIC,
                "characteristics": {
                    "logical_reasoning": 0.5,
                    "attention_to_detail": 0.7,
                    "systematic_approach": 0.4,
                    "creativity": 0.6,
                    "emotional_intelligence": 0.9,
                    "social_skills": 0.9,
                    "risk_tolerance": 0.3,
                    "adaptability": 0.7
                },
                "optimal_contexts": [ContextType.USER_INTERACTION, ContextType.COMMUNICATION],
                "strengths": ["User understanding", "Emotional support", "Trust building"],
                "weaknesses": ["May be overly cautious", "Can be indecisive"]
            },
            {
                "persona_type": PersonaType.TECHNICAL,
                "characteristics": {
                    "logical_reasoning": 0.9,
                    "attention_to_detail": 0.9,
                    "systematic_approach": 0.8,
                    "creativity": 0.4,
                    "emotional_intelligence": 0.3,
                    "social_skills": 0.4,
                    "risk_tolerance": 0.2,
                    "adaptability": 0.5
                },
                "optimal_contexts": [ContextType.TASK_DOMAIN, ContextType.PROBLEM_SOLVING],
                "strengths": ["Technical expertise", "Precision", "Reliability"],
                "weaknesses": ["Limited social skills", "May be rigid"]
            }
        ]
        
        for persona_data in builtin_personas:
            persona_id = f"builtin_{persona_data['persona_type'].value}"
            
            persona_profile = PersonaProfile(
                persona_id=persona_id,
                persona_name=f"Built-in {persona_data['persona_type'].value.title()}",
                persona_type=persona_data['persona_type'],
                intensity=PersonaIntensity.MODERATE,
                description=f"Built-in {persona_data['persona_type'].value} persona",
                characteristics=persona_data['characteristics'],
                behavioral_patterns={
                    "communication_frequency": 0.5,
                    "response_time": 0.7,
                    "collaboration_tendency": 0.6,
                    "learning_approach": 0.5
                },
                communication_style={
                    "formality": 0.5,
                    "directness": 0.5,
                    "supportiveness": 0.5,
                    "technical_depth": 0.5
                },
                decision_making_style={
                    "speed": 0.5,
                    "consultation": 0.5,
                    "risk_assessment": 0.5,
                    "consensus_seeking": 0.5
                },
                learning_preferences={
                    "exploration": 0.5,
                    "structured_learning": 0.5,
                    "collaborative_learning": 0.5,
                    "feedback_integration": 0.5
                },
                interaction_preferences={
                    "group_size": 0.5,
                    "interaction_depth": 0.5,
                    "proactive_engagement": 0.5,
                    "support_provision": 0.5
                },
                strengths=persona_data['strengths'],
                weaknesses=persona_data['weaknesses'],
                optimal_contexts=persona_data['optimal_contexts'],
                created_at=datetime.utcnow(),
                metadata={"builtin": True}
            )
            
            self.persona_profiles[persona_id] = persona_profile
    
    # ===== PERSONA MANAGEMENT =====
    
    async def create_persona(
        self,
        persona_name: str,
        persona_type: PersonaType,
        intensity: PersonaIntensity,
        description: str,
        characteristics: Dict[str, float],
        behavioral_patterns: Dict[str, Any],
        communication_style: Dict[str, Any],
        decision_making_style: Dict[str, Any],
        learning_preferences: Dict[str, Any],
        interaction_preferences: Dict[str, Any],
        strengths: List[str],
        weaknesses: List[str],
        optimal_contexts: List[ContextType],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new persona profile"""
        
        persona_id = f"persona_{uuid.uuid4().hex[:12]}"
        
        persona_profile = PersonaProfile(
            persona_id=persona_id,
            persona_name=persona_name,
            persona_type=persona_type,
            intensity=intensity,
            description=description,
            characteristics=characteristics,
            behavioral_patterns=behavioral_patterns,
            communication_style=communication_style,
            decision_making_style=decision_making_style,
            learning_preferences=learning_preferences,
            interaction_preferences=interaction_preferences,
            strengths=strengths,
            weaknesses=weaknesses,
            optimal_contexts=optimal_contexts,
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.persona_profiles[persona_id] = persona_profile
        
        # Store in KnowledgeGraph
        await self._store_persona_in_kg(persona_profile)
        
        # Publish persona creation event
        await shared_pulsar_client.publish(
            "q.agents.persona.created",
            {
                "persona_id": persona_id,
                "persona_name": persona_name,
                "persona_type": persona_type.value,
                "intensity": intensity.value,
                "characteristics": characteristics,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Created persona: {persona_name} ({persona_id})")
        return persona_id
    
    async def get_persona(self, persona_id: str) -> Optional[PersonaProfile]:
        """Get persona profile by ID"""
        return self.persona_profiles.get(persona_id)
    
    async def list_personas(
        self,
        persona_type: Optional[PersonaType] = None,
        intensity: Optional[PersonaIntensity] = None,
        context_type: Optional[ContextType] = None
    ) -> List[PersonaProfile]:
        """List persona profiles with optional filtering"""
        
        personas = list(self.persona_profiles.values())
        
        if persona_type:
            personas = [p for p in personas if p.persona_type == persona_type]
        
        if intensity:
            personas = [p for p in personas if p.intensity == intensity]
        
        if context_type:
            personas = [p for p in personas if context_type in p.optimal_contexts]
        
        return personas
    
    # ===== CONTEXT ANALYSIS =====
    
    async def analyze_context(
        self,
        agent_id: str,
        context_type: ContextType,
        features: Dict[str, Any],
        importance_weights: Optional[Dict[str, float]] = None
    ) -> str:
        """Analyze context and create context state"""
        
        context_id = f"context_{uuid.uuid4().hex[:12]}"
        
        # Extract different types of features
        temporal_features = self._extract_temporal_features()
        environmental_features = self._extract_environmental_features(features)
        task_features = self._extract_task_features(features)
        user_features = self._extract_user_features(features)
        team_features = self._extract_team_features(features)
        
        # Calculate confidence score
        confidence_score = self._calculate_context_confidence(features)
        
        context_state = ContextState(
            context_id=context_id,
            agent_id=agent_id,
            context_type=context_type,
            features=features,
            importance_weights=importance_weights or {},
            temporal_features=temporal_features,
            environmental_features=environmental_features,
            task_features=task_features,
            user_features=user_features,
            team_features=team_features,
            timestamp=datetime.utcnow(),
            confidence_score=confidence_score
        )
        
        self.context_states[context_id] = context_state
        
        # Store in KnowledgeGraph
        await self._store_context_in_kg(context_state)
        
        logger.info(f"Analyzed context: {context_type.value} for agent {agent_id}")
        return context_id
    
    def _extract_temporal_features(self) -> Dict[str, Any]:
        """Extract temporal features from current time"""
        
        now = datetime.utcnow()
        
        return {
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "month": now.month,
            "is_weekend": now.weekday() >= 5,
            "is_business_hours": 9 <= now.hour <= 17,
            "quarter": (now.month - 1) // 3 + 1
        }
    
    def _extract_environmental_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract environmental features"""
        
        return {
            "system_load": features.get("system_load", 0.5),
            "network_latency": features.get("network_latency", 0.0),
            "concurrent_users": features.get("concurrent_users", 1),
            "resource_availability": features.get("resource_availability", 1.0),
            "system_health": features.get("system_health", 1.0)
        }
    
    def _extract_task_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task-related features"""
        
        return {
            "task_complexity": features.get("task_complexity", 0.5),
            "task_urgency": features.get("task_urgency", 0.5),
            "task_domain": features.get("task_domain", "general"),
            "estimated_duration": features.get("estimated_duration", 3600),
            "required_skills": features.get("required_skills", []),
            "task_priority": features.get("task_priority", "medium")
        }
    
    def _extract_user_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user-related features"""
        
        return {
            "user_experience_level": features.get("user_experience_level", 0.5),
            "user_communication_style": features.get("user_communication_style", "neutral"),
            "user_satisfaction_history": features.get("user_satisfaction_history", 0.7),
            "user_preferred_interaction": features.get("user_preferred_interaction", "standard"),
            "user_domain_expertise": features.get("user_domain_expertise", 0.5)
        }
    
    def _extract_team_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract team-related features"""
        
        return {
            "team_size": features.get("team_size", 1),
            "team_experience": features.get("team_experience", 0.5),
            "team_collaboration_level": features.get("team_collaboration_level", 0.5),
            "team_communication_frequency": features.get("team_communication_frequency", 0.5),
            "team_decision_style": features.get("team_decision_style", "consensus")
        }
    
    def _calculate_context_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for context analysis"""
        
        # Simple confidence calculation based on feature completeness
        total_features = len(features)
        non_null_features = sum(1 for v in features.values() if v is not None)
        
        if total_features == 0:
            return 0.0
        
        return non_null_features / total_features
    
    # ===== PERSONA SELECTION =====
    
    async def select_persona(
        self,
        agent_id: str,
        context_id: str,
        selection_criteria: Optional[Dict[str, Any]] = None
    ) -> str:
        """Select optimal persona for given context"""
        
        context_state = self.context_states.get(context_id)
        if not context_state:
            raise ValueError(f"Context not found: {context_id}")
        
        # Convert context to feature vector
        context_vector = self._context_to_vector(context_state)
        
        # Use ML model for persona selection
        persona_scores, confidence = await self._ml_persona_selection(context_vector)
        
        # Get candidate personas
        candidate_personas = await self._get_candidate_personas(context_state, selection_criteria)
        
        # Select best persona
        best_persona_id = await self._select_best_persona(
            candidate_personas, persona_scores, confidence, context_state
        )
        
        # Create persona instance
        instance_id = await self._activate_persona(
            agent_id, best_persona_id, context_state, AdaptationTrigger.CONTEXT_CHANGE
        )
        
        logger.info(f"Selected persona {best_persona_id} for agent {agent_id}")
        return instance_id
    
    def _context_to_vector(self, context_state: ContextState) -> np.ndarray:
        """Convert context state to feature vector"""
        
        # Create feature vector from context state
        features = []
        
        # Add temporal features
        temporal = context_state.temporal_features
        features.extend([
            temporal.get("hour_of_day", 0) / 24,
            temporal.get("day_of_week", 0) / 7,
            temporal.get("month", 0) / 12,
            float(temporal.get("is_weekend", False)),
            float(temporal.get("is_business_hours", False)),
            temporal.get("quarter", 0) / 4
        ])
        
        # Add environmental features
        env = context_state.environmental_features
        features.extend([
            env.get("system_load", 0.5),
            env.get("network_latency", 0.0),
            min(env.get("concurrent_users", 1) / 100, 1.0),
            env.get("resource_availability", 1.0),
            env.get("system_health", 1.0)
        ])
        
        # Add task features
        task = context_state.task_features
        features.extend([
            task.get("task_complexity", 0.5),
            task.get("task_urgency", 0.5),
            min(task.get("estimated_duration", 3600) / 10800, 1.0),  # normalize to 3 hours
            len(task.get("required_skills", [])) / 10
        ])
        
        # Add user features
        user = context_state.user_features
        features.extend([
            user.get("user_experience_level", 0.5),
            user.get("user_satisfaction_history", 0.7),
            user.get("user_domain_expertise", 0.5)
        ])
        
        # Add team features
        team = context_state.team_features
        features.extend([
            min(team.get("team_size", 1) / 10, 1.0),
            team.get("team_experience", 0.5),
            team.get("team_collaboration_level", 0.5),
            team.get("team_communication_frequency", 0.5)
        ])
        
        # Pad or truncate to context_dim
        while len(features) < self.context_dim:
            features.append(0.0)
        
        return np.array(features[:self.context_dim])
    
    async def _ml_persona_selection(self, context_vector: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use ML model to select persona"""
        
        context_tensor = torch.FloatTensor(context_vector).unsqueeze(0)
        
        with torch.no_grad():
            persona_scores, confidence = self.persona_selector(context_tensor)
        
        return persona_scores, confidence
    
    async def _get_candidate_personas(
        self,
        context_state: ContextState,
        selection_criteria: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get candidate personas for selection"""
        
        candidates = []
        
        # Filter by optimal contexts
        for persona_id, persona in self.persona_profiles.items():
            if context_state.context_type in persona.optimal_contexts:
                candidates.append(persona_id)
        
        # Apply additional criteria
        if selection_criteria:
            filtered_candidates = []
            for persona_id in candidates:
                persona = self.persona_profiles[persona_id]
                
                # Check persona type
                if "persona_type" in selection_criteria:
                    if persona.persona_type != selection_criteria["persona_type"]:
                        continue
                
                # Check intensity
                if "intensity" in selection_criteria:
                    if persona.intensity != selection_criteria["intensity"]:
                        continue
                
                # Check minimum success rate
                if "min_success_rate" in selection_criteria:
                    if persona.success_rate < selection_criteria["min_success_rate"]:
                        continue
                
                filtered_candidates.append(persona_id)
            
            candidates = filtered_candidates
        
        # If no candidates, use all personas
        if not candidates:
            candidates = list(self.persona_profiles.keys())
        
        return candidates
    
    async def _select_best_persona(
        self,
        candidate_personas: List[str],
        persona_scores: torch.Tensor,
        confidence: torch.Tensor,
        context_state: ContextState
    ) -> str:
        """Select best persona from candidates"""
        
        best_persona_id = None
        best_score = -1
        
        for i, persona_id in enumerate(candidate_personas):
            if i >= len(persona_scores[0]):
                break
            
            persona = self.persona_profiles[persona_id]
            
            # ML model score
            ml_score = float(persona_scores[0][i])
            
            # Historical performance score
            performance_score = persona.success_rate
            
            # Context compatibility score
            context_score = self._calculate_context_compatibility(persona, context_state)
            
            # Combined score
            combined_score = (
                ml_score * 0.4 +
                performance_score * 0.3 +
                context_score * 0.3
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_persona_id = persona_id
        
        return best_persona_id or candidate_personas[0]
    
    def _calculate_context_compatibility(
        self,
        persona: PersonaProfile,
        context_state: ContextState
    ) -> float:
        """Calculate compatibility between persona and context"""
        
        # Base compatibility from optimal contexts
        base_compatibility = 1.0 if context_state.context_type in persona.optimal_contexts else 0.3
        
        # Adjust based on context features
        task_features = context_state.task_features
        
        # Task complexity compatibility
        complexity_match = 1.0 - abs(
            task_features.get("task_complexity", 0.5) - 
            persona.characteristics.get("logical_reasoning", 0.5)
        )
        
        # Urgency compatibility
        urgency_match = 1.0 - abs(
            task_features.get("task_urgency", 0.5) - 
            persona.characteristics.get("adaptability", 0.5)
        )
        
        # User interaction compatibility
        user_features = context_state.user_features
        social_match = 1.0 - abs(
            user_features.get("user_communication_style", 0.5) - 
            persona.characteristics.get("social_skills", 0.5)
        )
        
        # Weighted combination
        compatibility = (
            base_compatibility * 0.4 +
            complexity_match * 0.2 +
            urgency_match * 0.2 +
            social_match * 0.2
        )
        
        return compatibility
    
    # ===== PERSONA ACTIVATION =====
    
    async def _activate_persona(
        self,
        agent_id: str,
        persona_id: str,
        context_state: ContextState,
        adaptation_reason: AdaptationTrigger
    ) -> str:
        """Activate persona for an agent"""
        
        instance_id = f"instance_{uuid.uuid4().hex[:12]}"
        
        # Deactivate current persona if exists
        if agent_id in self.agent_current_personas:
            await self._deactivate_persona(agent_id)
        
        # Create persona instance
        persona_instance = PersonaInstance(
            instance_id=instance_id,
            agent_id=agent_id,
            persona_id=persona_id,
            activation_time=datetime.utcnow(),
            context=context_state.features,
            adaptation_reason=adaptation_reason,
            expected_duration=3600,  # 1 hour default
            performance_metrics={
                "interactions": 0,
                "success_rate": 0.0,
                "response_time": 0.0,
                "user_satisfaction": 0.0
            },
            interaction_history=[]
        )
        
        self.persona_instances[instance_id] = persona_instance
        self.agent_current_personas[agent_id] = instance_id
        self.agent_persona_history[agent_id].append(instance_id)
        
        # Update persona usage
        persona = self.persona_profiles[persona_id]
        persona.usage_count += 1
        
        # Publish persona activation event
        await shared_pulsar_client.publish(
            "q.agents.persona.activated",
            {
                "instance_id": instance_id,
                "agent_id": agent_id,
                "persona_id": persona_id,
                "persona_type": persona.persona_type.value,
                "context_type": context_state.context_type.value,
                "adaptation_reason": adaptation_reason.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Activated persona {persona_id} for agent {agent_id}")
        return instance_id
    
    async def _deactivate_persona(self, agent_id: str) -> bool:
        """Deactivate current persona for an agent"""
        
        if agent_id not in self.agent_current_personas:
            return False
        
        instance_id = self.agent_current_personas[agent_id]
        persona_instance = self.persona_instances.get(instance_id)
        
        if not persona_instance:
            return False
        
        # Update instance
        persona_instance.status = "inactive"
        persona_instance.deactivation_time = datetime.utcnow()
        
        # Calculate final performance metrics
        await self._calculate_instance_performance(instance_id)
        
        # Remove from active mapping
        del self.agent_current_personas[agent_id]
        
        # Publish deactivation event
        await shared_pulsar_client.publish(
            "q.agents.persona.deactivated",
            {
                "instance_id": instance_id,
                "agent_id": agent_id,
                "persona_id": persona_instance.persona_id,
                "duration": (persona_instance.deactivation_time - persona_instance.activation_time).total_seconds(),
                "performance_metrics": persona_instance.performance_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Deactivated persona instance {instance_id} for agent {agent_id}")
        return True
    
    async def _calculate_instance_performance(self, instance_id: str):
        """Calculate performance metrics for a persona instance"""
        
        persona_instance = self.persona_instances.get(instance_id)
        if not persona_instance:
            return
        
        # Calculate metrics from interaction history
        if persona_instance.interaction_history:
            interactions = persona_instance.interaction_history
            
            # Success rate
            successes = sum(1 for interaction in interactions if interaction.get("success", False))
            persona_instance.performance_metrics["success_rate"] = successes / len(interactions)
            
            # Average response time
            response_times = [interaction.get("response_time", 0) for interaction in interactions]
            persona_instance.performance_metrics["response_time"] = np.mean(response_times)
            
            # User satisfaction
            satisfaction_scores = [interaction.get("user_satisfaction", 0) for interaction in interactions]
            persona_instance.performance_metrics["user_satisfaction"] = np.mean(satisfaction_scores)
            
            # Update persona overall metrics
            persona = self.persona_profiles[persona_instance.persona_id]
            persona.average_performance = (
                persona.average_performance * (persona.usage_count - 1) +
                persona_instance.performance_metrics["success_rate"]
            ) / persona.usage_count
    
    # ===== ADAPTATION LOGIC =====
    
    async def should_adapt_persona(
        self,
        agent_id: str,
        trigger: AdaptationTrigger,
        context_data: Dict[str, Any]
    ) -> bool:
        """Determine if persona should be adapted"""
        
        # Check if agent has active persona
        if agent_id not in self.agent_current_personas:
            return True
        
        instance_id = self.agent_current_personas[agent_id]
        persona_instance = self.persona_instances.get(instance_id)
        
        if not persona_instance:
            return True
        
        # Check minimum activation duration
        activation_duration = (datetime.utcnow() - persona_instance.activation_time).total_seconds()
        if activation_duration < self.config["min_activation_duration"]:
            return False
        
        # Check adaptation frequency
        recent_adaptations = [
            inst for inst in self.agent_persona_history[agent_id][-5:]  # Last 5 adaptations
            if self.persona_instances.get(inst)
        ]
        
        recent_count = len(recent_adaptations)
        if recent_count >= self.config["max_adaptation_frequency"]:
            return False
        
        # Trigger-specific logic
        if trigger == AdaptationTrigger.PERFORMANCE_DECLINE:
            current_performance = persona_instance.performance_metrics.get("success_rate", 0.0)
            return current_performance < self.config["adaptation_threshold"]
        
        elif trigger == AdaptationTrigger.CONTEXT_CHANGE:
            # Check if context has changed significantly
            return self._has_context_changed_significantly(persona_instance, context_data)
        
        elif trigger == AdaptationTrigger.USER_FEEDBACK:
            # Check user feedback score
            user_satisfaction = context_data.get("user_satisfaction", 0.0)
            return user_satisfaction < self.config["adaptation_threshold"]
        
        elif trigger == AdaptationTrigger.TASK_REQUIREMENT:
            # Check if current persona matches task requirements
            return not self._persona_matches_task_requirements(persona_instance, context_data)
        
        return False
    
    def _has_context_changed_significantly(
        self,
        persona_instance: PersonaInstance,
        new_context: Dict[str, Any]
    ) -> bool:
        """Check if context has changed significantly"""
        
        old_context = persona_instance.context
        
        # Calculate context similarity
        similarity = self._calculate_context_similarity(old_context, new_context)
        
        # Adapt if similarity is below threshold
        return similarity < 0.7
    
    def _calculate_context_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two contexts"""
        
        # Simple similarity based on common keys
        common_keys = set(context1.keys()) & set(context2.keys())
        
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        for key in common_keys:
            val1 = context1[key]
            val2 = context2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1.0)
                similarity = 1.0 - abs(val1 - val2) / max_val
            else:
                # String/boolean similarity
                similarity = 1.0 if val1 == val2 else 0.0
            
            similarity_scores.append(similarity)
        
        return np.mean(similarity_scores)
    
    def _persona_matches_task_requirements(
        self,
        persona_instance: PersonaInstance,
        context_data: Dict[str, Any]
    ) -> bool:
        """Check if persona matches task requirements"""
        
        persona = self.persona_profiles[persona_instance.persona_id]
        task_requirements = context_data.get("task_requirements", {})
        
        # Check required characteristics
        for characteristic, required_level in task_requirements.items():
            persona_level = persona.characteristics.get(characteristic, 0.0)
            if persona_level < required_level:
                return False
        
        return True
    
    # ===== PERFORMANCE TRACKING =====
    
    async def record_interaction(
        self,
        agent_id: str,
        interaction_data: Dict[str, Any]
    ):
        """Record interaction for performance tracking"""
        
        if agent_id not in self.agent_current_personas:
            return
        
        instance_id = self.agent_current_personas[agent_id]
        persona_instance = self.persona_instances.get(instance_id)
        
        if not persona_instance:
            return
        
        # Add interaction to history
        interaction_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "interaction_type": interaction_data.get("type", "unknown"),
            "success": interaction_data.get("success", False),
            "response_time": interaction_data.get("response_time", 0.0),
            "user_satisfaction": interaction_data.get("user_satisfaction", 0.0),
            "context": interaction_data.get("context", {}),
            "metadata": interaction_data.get("metadata", {})
        }
        
        persona_instance.interaction_history.append(interaction_record)
        
        # Update performance metrics
        persona_instance.performance_metrics["interactions"] += 1
        
        # Calculate running averages
        interactions = persona_instance.interaction_history
        if interactions:
            successes = sum(1 for i in interactions if i.get("success", False))
            persona_instance.performance_metrics["success_rate"] = successes / len(interactions)
            
            response_times = [i.get("response_time", 0) for i in interactions]
            persona_instance.performance_metrics["response_time"] = np.mean(response_times)
            
            satisfaction_scores = [i.get("user_satisfaction", 0) for i in interactions]
            persona_instance.performance_metrics["user_satisfaction"] = np.mean(satisfaction_scores)
    
    async def compute_persona_performance(
        self,
        persona_id: str,
        context_type: ContextType,
        time_window: int = 3600
    ) -> PersonaPerformanceMetrics:
        """Compute performance metrics for a persona"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=time_window)
        
        # Get relevant instances
        relevant_instances = [
            instance for instance in self.persona_instances.values()
            if (instance.persona_id == persona_id and
                instance.activation_time >= start_time and
                instance.activation_time <= end_time)
        ]
        
        if not relevant_instances:
            # Return default metrics
            return PersonaPerformanceMetrics(
                persona_id=persona_id,
                agent_id="all",
                context_type=context_type,
                task_completion_rate=0.0,
                user_satisfaction=0.0,
                response_quality=0.0,
                collaboration_effectiveness=0.0,
                learning_progress=0.0,
                adaptability_score=0.0,
                consistency_score=0.0,
                innovation_score=0.0,
                measurement_period=(start_time, end_time),
                sample_size=0,
                computed_at=datetime.utcnow()
            )
        
        # Calculate metrics
        all_interactions = []
        for instance in relevant_instances:
            all_interactions.extend(instance.interaction_history)
        
        sample_size = len(all_interactions)
        
        if sample_size == 0:
            task_completion_rate = 0.0
            user_satisfaction = 0.0
            response_quality = 0.0
        else:
            task_completion_rate = sum(1 for i in all_interactions if i.get("success", False)) / sample_size
            user_satisfaction = np.mean([i.get("user_satisfaction", 0) for i in all_interactions])
            response_quality = np.mean([i.get("response_quality", 0) for i in all_interactions])
        
        # Calculate other metrics (simplified)
        collaboration_effectiveness = user_satisfaction * 0.8  # Simplified
        learning_progress = min(1.0, len(relevant_instances) / 10)  # Based on usage
        adaptability_score = 1.0 - (len(relevant_instances) / max(len(relevant_instances), 1))  # Stability
        consistency_score = 1.0 - np.std([i.get("response_time", 0) for i in all_interactions]) / 1000
        innovation_score = np.mean([i.get("creativity_score", 0.5) for i in all_interactions])
        
        return PersonaPerformanceMetrics(
            persona_id=persona_id,
            agent_id="all",
            context_type=context_type,
            task_completion_rate=task_completion_rate,
            user_satisfaction=user_satisfaction,
            response_quality=response_quality,
            collaboration_effectiveness=collaboration_effectiveness,
            learning_progress=learning_progress,
            adaptability_score=adaptability_score,
            consistency_score=consistency_score,
            innovation_score=innovation_score,
            measurement_period=(start_time, end_time),
            sample_size=sample_size,
            computed_at=datetime.utcnow()
        )
    
    # ===== BACKGROUND TASKS =====
    
    async def _context_monitoring(self):
        """Monitor context changes"""
        
        while True:
            try:
                # Monitor active persona instances for context changes
                for agent_id, instance_id in self.agent_current_personas.items():
                    persona_instance = self.persona_instances.get(instance_id)
                    if not persona_instance:
                        continue
                    
                    # Check if adaptation is needed
                    if await self.should_adapt_persona(
                        agent_id, AdaptationTrigger.CONTEXT_CHANGE, {}
                    ):
                        logger.info(f"Context change detected for agent {agent_id}")
                        # Trigger adaptation logic would go here
                
                await asyncio.sleep(self.config["context_update_frequency"])
                
            except Exception as e:
                logger.error(f"Error in context monitoring: {e}")
                await asyncio.sleep(self.config["context_update_frequency"])
    
    async def _adaptation_monitoring(self):
        """Monitor persona adaptation needs"""
        
        while True:
            try:
                # Check performance-based adaptation needs
                for agent_id, instance_id in self.agent_current_personas.items():
                    persona_instance = self.persona_instances.get(instance_id)
                    if not persona_instance:
                        continue
                    
                    # Check performance
                    performance = persona_instance.performance_metrics.get("success_rate", 1.0)
                    if performance < self.config["adaptation_threshold"]:
                        logger.info(f"Performance decline detected for agent {agent_id}: {performance}")
                        # Trigger adaptation logic would go here
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in adaptation monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _performance_tracking(self):
        """Track persona performance"""
        
        while True:
            try:
                # Update persona performance metrics
                for persona_id in self.persona_profiles:
                    for context_type in ContextType:
                        metrics = await self.compute_persona_performance(
                            persona_id, context_type, self.config["performance_window"]
                        )
                        
                        # Store metrics
                        self.performance_metrics[persona_id].append(metrics)
                        
                        # Keep only recent metrics
                        cutoff_time = datetime.utcnow() - timedelta(days=7)
                        self.performance_metrics[persona_id] = [
                            m for m in self.performance_metrics[persona_id]
                            if m.computed_at >= cutoff_time
                        ]
                
                await asyncio.sleep(1800)  # Update every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(1800)
    
    async def _persona_evolution(self):
        """Evolve personas based on performance"""
        
        while True:
            try:
                # Evolve personas based on performance data
                for persona_id, persona in self.persona_profiles.items():
                    if persona.metadata.get("builtin", False):
                        continue  # Don't evolve built-in personas
                    
                    # Get recent performance metrics
                    recent_metrics = [
                        m for m in self.performance_metrics[persona_id]
                        if (datetime.utcnow() - m.computed_at).total_seconds() < 86400  # Last 24 hours
                    ]
                    
                    if len(recent_metrics) >= 5:  # Need sufficient data
                        await self._evolve_persona(persona_id, recent_metrics)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in persona evolution: {e}")
                await asyncio.sleep(3600)
    
    async def _evolve_persona(self, persona_id: str, metrics: List[PersonaPerformanceMetrics]):
        """Evolve a persona based on performance metrics"""
        
        persona = self.persona_profiles[persona_id]
        
        # Calculate average performance
        avg_performance = np.mean([m.task_completion_rate for m in metrics])
        
        # If performance is good, don't change much
        if avg_performance > 0.8:
            return
        
        # Identify areas for improvement
        improvements = {}
        
        # Low user satisfaction -> increase empathy/social skills
        avg_satisfaction = np.mean([m.user_satisfaction for m in metrics])
        if avg_satisfaction < 0.6:
            improvements["emotional_intelligence"] = 0.1
            improvements["social_skills"] = 0.1
        
        # Low collaboration -> increase collaborative traits
        avg_collaboration = np.mean([m.collaboration_effectiveness for m in metrics])
        if avg_collaboration < 0.6:
            improvements["social_skills"] = 0.05
            improvements["adaptability"] = 0.05
        
        # Low consistency -> increase systematic approach
        avg_consistency = np.mean([m.consistency_score for m in metrics])
        if avg_consistency < 0.6:
            improvements["systematic_approach"] = 0.1
            improvements["attention_to_detail"] = 0.05
        
        # Apply improvements
        for characteristic, improvement in improvements.items():
            if characteristic in persona.characteristics:
                old_value = persona.characteristics[characteristic]
                new_value = min(1.0, old_value + improvement)
                persona.characteristics[characteristic] = new_value
                
                logger.info(f"Evolved persona {persona_id}: {characteristic} {old_value:.2f} -> {new_value:.2f}")
        
        # Update performance metrics
        persona.success_rate = avg_performance
    
    async def _learning_loop(self):
        """Main learning loop for ML models"""
        
        while True:
            try:
                # Train persona selector
                await self._train_persona_selector()
                
                # Train adaptation network
                await self._train_adaptation_network()
                
                # Train evolution network
                await self._train_evolution_network()
                
                await asyncio.sleep(self.config["train_frequency"])
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(self.config["train_frequency"])
    
    async def _train_persona_selector(self):
        """Train the persona selector network"""
        
        if len(self.persona_experience_buffer) < self.config["batch_size"]:
            return
        
        # Sample batch
        batch = random.sample(self.persona_experience_buffer, self.config["batch_size"])
        
        # Prepare training data
        contexts = []
        targets = []
        
        for experience in batch:
            context_vector = experience["context_vector"]
            selected_persona = experience["selected_persona"]
            performance = experience["performance"]
            
            contexts.append(context_vector)
            
            # Create target based on performance
            target = torch.zeros(self.max_personas)
            persona_index = experience["persona_index"]
            target[persona_index] = performance
            targets.append(target)
        
        contexts = torch.FloatTensor(contexts)
        targets = torch.FloatTensor(targets)
        
        # Forward pass
        persona_scores, confidence = self.persona_selector(contexts)
        
        # Calculate loss
        loss = F.mse_loss(persona_scores, targets)
        
        # Backward pass
        self.selector_optimizer.zero_grad()
        loss.backward()
        self.selector_optimizer.step()
    
    async def _train_adaptation_network(self):
        """Train the adaptation network"""
        
        if len(self.adaptation_experience_buffer) < self.config["batch_size"]:
            return
        
        # Sample batch
        batch = random.sample(self.adaptation_experience_buffer, self.config["batch_size"])
        
        # Prepare training data (simplified)
        states = torch.FloatTensor([exp["state"] for exp in batch])
        actions = torch.LongTensor([exp["action"] for exp in batch])
        rewards = torch.FloatTensor([exp["reward"] for exp in batch])
        
        # Forward pass
        values, policies = self.adaptation_net(states)
        
        # Calculate loss (simplified policy gradient)
        policy_loss = F.cross_entropy(policies, actions)
        value_loss = F.mse_loss(values.squeeze(), rewards)
        
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.adaptation_optimizer.zero_grad()
        total_loss.backward()
        self.adaptation_optimizer.step()
    
    async def _train_evolution_network(self):
        """Train the evolution network"""
        
        # This would train the evolution network based on persona evolution data
        # Implementation would be similar to other training methods
        pass
    
    # ===== KNOWLEDGEGRAPH INTEGRATION =====
    
    async def _store_persona_in_kg(self, persona_profile: PersonaProfile):
        """Store persona profile in KnowledgeGraph"""
        
        try:
            vertex_data = {
                "persona_id": persona_profile.persona_id,
                "persona_name": persona_profile.persona_name,
                "persona_type": persona_profile.persona_type.value,
                "intensity": persona_profile.intensity.value,
                "description": persona_profile.description,
                "characteristics": persona_profile.characteristics,
                "behavioral_patterns": persona_profile.behavioral_patterns,
                "communication_style": persona_profile.communication_style,
                "decision_making_style": persona_profile.decision_making_style,
                "learning_preferences": persona_profile.learning_preferences,
                "interaction_preferences": persona_profile.interaction_preferences,
                "strengths": persona_profile.strengths,
                "weaknesses": persona_profile.weaknesses,
                "optimal_contexts": [ctx.value for ctx in persona_profile.optimal_contexts],
                "created_at": persona_profile.created_at.isoformat(),
                "usage_count": persona_profile.usage_count,
                "success_rate": persona_profile.success_rate,
                "average_performance": persona_profile.average_performance,
                "metadata": persona_profile.metadata
            }
            
            await self.kg_client.add_vertex(
                "Persona", 
                persona_profile.persona_id, 
                vertex_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store persona in KnowledgeGraph: {e}")
    
    async def _store_context_in_kg(self, context_state: ContextState):
        """Store context state in KnowledgeGraph"""
        
        try:
            vertex_data = {
                "context_id": context_state.context_id,
                "agent_id": context_state.agent_id,
                "context_type": context_state.context_type.value,
                "features": context_state.features,
                "importance_weights": context_state.importance_weights,
                "temporal_features": context_state.temporal_features,
                "environmental_features": context_state.environmental_features,
                "task_features": context_state.task_features,
                "user_features": context_state.user_features,
                "team_features": context_state.team_features,
                "timestamp": context_state.timestamp.isoformat(),
                "confidence_score": context_state.confidence_score
            }
            
            await self.kg_client.add_vertex(
                "Context", 
                context_state.context_id, 
                vertex_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store context in KnowledgeGraph: {e}")
    
    # ===== UTILITY METHODS =====
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for persona service"""
        
        topics = [
            "q.agents.persona.created",
            "q.agents.persona.activated",
            "q.agents.persona.deactivated",
            "q.agents.persona.evolved",
            "q.agents.persona.performance.updated"
        ]
        
        logger.info("Adaptive persona service Pulsar topics configured")
    
    # ===== PUBLIC API =====
    
    async def get_active_persona(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get active persona for an agent"""
        
        if agent_id not in self.agent_current_personas:
            return None
        
        instance_id = self.agent_current_personas[agent_id]
        persona_instance = self.persona_instances.get(instance_id)
        
        if not persona_instance:
            return None
        
        persona = self.persona_profiles[persona_instance.persona_id]
        
        return {
            "instance_id": instance_id,
            "persona_id": persona.persona_id,
            "persona_name": persona.persona_name,
            "persona_type": persona.persona_type.value,
            "intensity": persona.intensity.value,
            "characteristics": persona.characteristics,
            "activation_time": persona_instance.activation_time.isoformat(),
            "performance_metrics": persona_instance.performance_metrics,
            "status": persona_instance.status
        }
    
    async def get_persona_history(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get persona history for an agent"""
        
        if agent_id not in self.agent_persona_history:
            return []
        
        history = []
        instance_ids = self.agent_persona_history[agent_id][-limit:]
        
        for instance_id in instance_ids:
            persona_instance = self.persona_instances.get(instance_id)
            if persona_instance:
                persona = self.persona_profiles[persona_instance.persona_id]
                
                history.append({
                    "instance_id": instance_id,
                    "persona_id": persona.persona_id,
                    "persona_name": persona.persona_name,
                    "persona_type": persona.persona_type.value,
                    "activation_time": persona_instance.activation_time.isoformat(),
                    "deactivation_time": persona_instance.deactivation_time.isoformat() if persona_instance.deactivation_time else None,
                    "adaptation_reason": persona_instance.adaptation_reason.value,
                    "performance_metrics": persona_instance.performance_metrics,
                    "status": persona_instance.status
                })
        
        return history
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics"""
        
        # Update metrics
        self.service_metrics["total_personas"] = len(self.persona_profiles)
        self.service_metrics["active_instances"] = len(self.agent_current_personas)
        
        # Calculate persona utilization
        if self.persona_profiles:
            total_usage = sum(p.usage_count for p in self.persona_profiles.values())
            self.service_metrics["persona_utilization"] = total_usage / (len(self.persona_profiles) * 100)
        
        return {
            "service_metrics": self.service_metrics,
            "config": self.config,
            "experience_buffer_size": len(self.persona_experience_buffer),
            "adaptation_buffer_size": len(self.adaptation_experience_buffer)
        }

# Global instance
adaptive_persona_service = AdaptivePersonaService() 