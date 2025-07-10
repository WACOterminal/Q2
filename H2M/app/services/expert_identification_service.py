"""
Expert Identification Service

This service provides intelligent expert identification and matching capabilities:
- Automatic expert profiling based on historical performance
- Machine learning-based expertise assessment
- Real-time expert matching for consultation requests
- Expert availability management and scheduling
- Performance tracking and continuous improvement
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import asdict

# Q Platform imports
from shared.q_collaboration_schemas.models import (
    ExpertProfile, ExpertConsultationRequest, ExpertiseArea,
    CollaborationSession, CollaborationType, CollaborationStatus,
    match_experts_to_request, calculate_collaboration_priority
)
from shared.q_memory_schemas.memory_models import AgentMemory, MemoryType
from shared.q_analytics_schemas.models import PerformanceMetrics
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.memory_service import MemoryService
from app.services.pulsar_service import PulsarService
from app.services.ignite_service import IgniteService

logger = logging.getLogger(__name__)

class ExpertIdentificationService:
    """
    Service for identifying and matching experts to consultation requests
    """
    
    def __init__(self):
        self.knowledge_graph = KnowledgeGraphService()
        self.memory_service = MemoryService()
        self.pulsar_service = PulsarService()
        self.ignite_service = IgniteService()
        
        # Cache configurations
        self.expert_cache_ttl = 3600  # 1 hour
        self.performance_cache_ttl = 1800  # 30 minutes
        
        # ML model configurations
        self.expertise_model = None
        self.matching_model = None
        
    async def initialize(self):
        """Initialize the expert identification service"""
        logger.info("Initializing Expert Identification Service")
        
        # Load ML models for expertise assessment
        await self._load_expertise_models()
        
        # Initialize real-time expert monitoring
        await self._setup_expert_monitoring()
        
        logger.info("Expert Identification Service initialized successfully")
    
    # ===== EXPERT PROFILING =====
    
    async def create_expert_profile(
        self, 
        user_id: str, 
        name: str, 
        initial_expertise: List[ExpertiseArea],
        specializations: List[str]
    ) -> ExpertProfile:
        """
        Create a new expert profile
        
        Args:
            user_id: Unique user identifier
            name: Expert name
            initial_expertise: Initial expertise areas
            specializations: Specific specializations
            
        Returns:
            Created expert profile
        """
        logger.info(f"Creating expert profile for user: {user_id}")
        
        # Initialize performance metrics
        initial_metrics = {
            "consultation_count": 0,
            "success_rate": 0.0,
            "average_response_time": 0.0,
            "satisfaction_score": 0.0,
            "complexity_handled": 0.0,
            "knowledge_depth": 0.0
        }
        
        # Create expert profile
        expert_profile = ExpertProfile(
            user_id=user_id,
            name=name,
            expertise_areas=initial_expertise,
            specializations=specializations,
            performance_metrics=initial_metrics,
            availability_schedule={},
            current_load=0,
            max_concurrent_sessions=3,  # Default
            response_time_avg=0.0,
            success_rate=0.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store in knowledge graph
        await self._store_expert_profile(expert_profile)
        
        # Cache the profile
        await self.ignite_service.put(
            f"expert_profile:{user_id}",
            asdict(expert_profile),
            ttl=self.expert_cache_ttl
        )
        
        logger.info(f"Expert profile created for user: {user_id}")
        return expert_profile
    
    async def update_expert_profile(
        self, 
        user_id: str, 
        updates: Dict[str, Any]
    ) -> ExpertProfile:
        """
        Update an existing expert profile
        
        Args:
            user_id: Expert user ID
            updates: Profile updates
            
        Returns:
            Updated expert profile
        """
        logger.info(f"Updating expert profile for user: {user_id}")
        
        # Get current profile
        current_profile = await self.get_expert_profile(user_id)
        if not current_profile:
            raise ValueError(f"Expert profile not found for user: {user_id}")
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(current_profile, key):
                setattr(current_profile, key, value)
        
        current_profile.updated_at = datetime.utcnow()
        
        # Store updated profile
        await self._store_expert_profile(current_profile)
        
        # Update cache
        await self.ignite_service.put(
            f"expert_profile:{user_id}",
            asdict(current_profile),
            ttl=self.expert_cache_ttl
        )
        
        logger.info(f"Expert profile updated for user: {user_id}")
        return current_profile
    
    async def get_expert_profile(self, user_id: str) -> Optional[ExpertProfile]:
        """
        Get expert profile by user ID
        
        Args:
            user_id: Expert user ID
            
        Returns:
            Expert profile or None if not found
        """
        # Check cache first
        cached_profile = await self.ignite_service.get(f"expert_profile:{user_id}")
        if cached_profile:
            return ExpertProfile(**cached_profile)
        
        # Load from knowledge graph
        profile_data = await self.knowledge_graph.get_expert_profile(user_id)
        if profile_data:
            profile = ExpertProfile(**profile_data)
            
            # Cache for future use
            await self.ignite_service.put(
                f"expert_profile:{user_id}",
                asdict(profile),
                ttl=self.expert_cache_ttl
            )
            
            return profile
        
        return None
    
    # ===== EXPERTISE ASSESSMENT =====
    
    async def assess_expertise(
        self, 
        user_id: str,
        domain: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Assess user expertise in a specific domain
        
        Args:
            user_id: User to assess
            domain: Domain of expertise
            context: Assessment context
            
        Returns:
            Expertise assessment scores
        """
        logger.info(f"Assessing expertise for user: {user_id} in domain: {domain}")
        
        # Get historical performance data
        historical_data = await self._get_expertise_history(user_id, domain)
        
        # Get recent collaboration outcomes
        recent_outcomes = await self._get_recent_collaboration_outcomes(user_id, domain)
        
        # Analyze knowledge graph connections
        knowledge_connections = await self._analyze_knowledge_connections(user_id, domain)
        
        # Calculate expertise scores
        expertise_scores = {
            "knowledge_depth": await self._calculate_knowledge_depth(historical_data),
            "practical_experience": await self._calculate_practical_experience(recent_outcomes),
            "network_influence": await self._calculate_network_influence(knowledge_connections),
            "learning_velocity": await self._calculate_learning_velocity(historical_data),
            "problem_solving": await self._calculate_problem_solving_ability(recent_outcomes)
        }
        
        # Overall expertise score
        expertise_scores["overall"] = np.mean(list(expertise_scores.values()))
        
        logger.info(f"Expertise assessment completed for user: {user_id}")
        return expertise_scores
    
    async def update_expertise_from_collaboration(
        self, 
        user_id: str,
        collaboration_session: CollaborationSession
    ):
        """
        Update expertise based on collaboration outcomes
        
        Args:
            user_id: Expert user ID
            collaboration_session: Completed collaboration session
        """
        logger.info(f"Updating expertise for user: {user_id} from session: {collaboration_session.session_id}")
        
        # Extract learning signals
        learning_signals = await self._extract_learning_signals(collaboration_session)
        
        # Update expertise areas
        await self._update_expertise_areas(user_id, learning_signals)
        
        # Update performance metrics
        await self._update_performance_metrics(user_id, collaboration_session)
        
        # Store learning memory
        await self._store_expertise_memory(user_id, learning_signals)
        
        logger.info(f"Expertise updated for user: {user_id}")
    
    # ===== EXPERT MATCHING =====
    
    async def find_experts_for_consultation(
        self, 
        request: ExpertConsultationRequest
    ) -> List[Tuple[str, float]]:
        """
        Find the best experts for a consultation request
        
        Args:
            request: Expert consultation request
            
        Returns:
            List of (expert_id, match_score) tuples
        """
        logger.info(f"Finding experts for consultation request: {request.request_id}")
        
        # Get all available experts
        available_experts = await self._get_available_experts()
        
        # Filter by expertise areas
        relevant_experts = [
            expert for expert in available_experts
            if any(area in expert.expertise_areas for area in request.expertise_needed)
        ]
        
        # Calculate match scores
        expert_matches = []
        for expert in relevant_experts:
            match_score = await self._calculate_match_score(request, expert)
            expert_matches.append((expert.user_id, match_score))
        
        # Sort by match score
        expert_matches.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(expert_matches)} experts for consultation")
        return expert_matches[:10]  # Top 10 matches
    
    async def _calculate_match_score(
        self, 
        request: ExpertConsultationRequest,
        expert: ExpertProfile
    ) -> float:
        """
        Calculate match score between request and expert
        
        Args:
            request: Consultation request
            expert: Expert profile
            
        Returns:
            Match score (0.0 to 1.0)
        """
        # Expertise match
        expertise_match = len(set(request.expertise_needed) & set(expert.expertise_areas))
        expertise_score = expertise_match / len(request.expertise_needed)
        
        # Skill match
        skill_match = len(set(request.specific_skills) & set(expert.specializations))
        skill_score = skill_match / max(len(request.specific_skills), 1)
        
        # Availability score
        availability_score = 1.0 - (expert.current_load / expert.max_concurrent_sessions)
        
        # Performance score
        performance_score = expert.success_rate
        
        # Response time score (lower is better)
        response_time_score = 1.0 - min(expert.response_time_avg / 60.0, 1.0)  # Normalize to hours
        
        # Urgency match
        urgency_score = 1.0 if request.urgency_level <= 3 else availability_score
        
        # Weighted combination
        match_score = (
            expertise_score * 0.3 +
            skill_score * 0.2 +
            availability_score * 0.2 +
            performance_score * 0.15 +
            response_time_score * 0.1 +
            urgency_score * 0.05
        )
        
        return min(1.0, max(0.0, match_score))
    
    # ===== AVAILABILITY MANAGEMENT =====
    
    async def update_expert_availability(
        self, 
        user_id: str, 
        schedule: Dict[str, List[str]]
    ):
        """
        Update expert availability schedule
        
        Args:
            user_id: Expert user ID
            schedule: Availability schedule (day -> time_slots)
        """
        logger.info(f"Updating availability for expert: {user_id}")
        
        expert_profile = await self.get_expert_profile(user_id)
        if expert_profile:
            expert_profile.availability_schedule = schedule
            expert_profile.updated_at = datetime.utcnow()
            
            await self._store_expert_profile(expert_profile)
            
            # Update cache
            await self.ignite_service.put(
                f"expert_profile:{user_id}",
                asdict(expert_profile),
                ttl=self.expert_cache_ttl
            )
    
    async def check_expert_availability(
        self, 
        user_id: str, 
        requested_time: datetime
    ) -> bool:
        """
        Check if expert is available at requested time
        
        Args:
            user_id: Expert user ID
            requested_time: Requested consultation time
            
        Returns:
            True if available, False otherwise
        """
        expert_profile = await self.get_expert_profile(user_id)
        if not expert_profile:
            return False
        
        # Check current load
        if expert_profile.current_load >= expert_profile.max_concurrent_sessions:
            return False
        
        # Check schedule
        day_of_week = requested_time.strftime("%A").lower()
        time_slot = requested_time.strftime("%H:%M")
        
        if day_of_week in expert_profile.availability_schedule:
            available_slots = expert_profile.availability_schedule[day_of_week]
            # Simple time slot check (could be more sophisticated)
            return any(time_slot >= slot.split("-")[0] and time_slot <= slot.split("-")[1] 
                      for slot in available_slots if "-" in slot)
        
        return False
    
    # ===== PERFORMANCE TRACKING =====
    
    async def track_consultation_outcome(
        self, 
        expert_id: str,
        consultation_id: str,
        outcome: Dict[str, Any]
    ):
        """
        Track consultation outcome for expert performance
        
        Args:
            expert_id: Expert user ID
            consultation_id: Consultation ID
            outcome: Consultation outcome data
        """
        logger.info(f"Tracking consultation outcome for expert: {expert_id}")
        
        # Update performance metrics
        await self._update_consultation_metrics(expert_id, outcome)
        
        # Store outcome in knowledge graph
        await self.knowledge_graph.store_consultation_outcome(
            expert_id, consultation_id, outcome
        )
        
        # Publish metrics update
        await self.pulsar_service.publish(
            "q.experts.performance.updated",
            {
                "expert_id": expert_id,
                "consultation_id": consultation_id,
                "outcome": outcome,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    # ===== PRIVATE METHODS =====
    
    async def _load_expertise_models(self):
        """Load ML models for expertise assessment"""
        # This would load pre-trained models
        # For now, using placeholder
        self.expertise_model = "placeholder_expertise_model"
        self.matching_model = "placeholder_matching_model"
        logger.info("Expertise models loaded")
    
    async def _setup_expert_monitoring(self):
        """Setup real-time expert monitoring"""
        # Subscribe to collaboration events
        await self.pulsar_service.subscribe(
            "q.collaboration.session.completed",
            self._handle_collaboration_completed
        )
        
        logger.info("Expert monitoring setup completed")
    
    async def _handle_collaboration_completed(self, message: Dict[str, Any]):
        """Handle collaboration completion events"""
        try:
            session_data = message.get("session", {})
            expert_id = session_data.get("assigned_expert")
            
            if expert_id:
                # Update expertise based on collaboration
                await self.update_expertise_from_collaboration(
                    expert_id, 
                    CollaborationSession(**session_data)
                )
                
        except Exception as e:
            logger.error(f"Error handling collaboration completion: {e}")
    
    async def _store_expert_profile(self, profile: ExpertProfile):
        """Store expert profile in knowledge graph"""
        await self.knowledge_graph.store_expert_profile(asdict(profile))
    
    async def _get_available_experts(self) -> List[ExpertProfile]:
        """Get all available experts"""
        # This would query the knowledge graph for all experts
        # For now, returning placeholder
        return []
    
    async def _get_expertise_history(self, user_id: str, domain: str) -> List[Dict[str, Any]]:
        """Get historical expertise data"""
        return await self.knowledge_graph.get_expertise_history(user_id, domain)
    
    async def _get_recent_collaboration_outcomes(self, user_id: str, domain: str) -> List[Dict[str, Any]]:
        """Get recent collaboration outcomes"""
        return await self.knowledge_graph.get_recent_collaboration_outcomes(user_id, domain)
    
    async def _analyze_knowledge_connections(self, user_id: str, domain: str) -> Dict[str, Any]:
        """Analyze knowledge graph connections"""
        return await self.knowledge_graph.analyze_knowledge_connections(user_id, domain)
    
    async def _calculate_knowledge_depth(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate knowledge depth score"""
        if not historical_data:
            return 0.0
        
        # Calculate based on complexity of problems solved
        complexity_scores = [item.get("complexity", 0) for item in historical_data]
        return np.mean(complexity_scores) if complexity_scores else 0.0
    
    async def _calculate_practical_experience(self, recent_outcomes: List[Dict[str, Any]]) -> float:
        """Calculate practical experience score"""
        if not recent_outcomes:
            return 0.0
        
        # Calculate based on success rate and variety
        success_rates = [item.get("success_rate", 0) for item in recent_outcomes]
        return np.mean(success_rates) if success_rates else 0.0
    
    async def _calculate_network_influence(self, knowledge_connections: Dict[str, Any]) -> float:
        """Calculate network influence score"""
        # Calculate based on graph centrality and connections
        centrality = knowledge_connections.get("centrality", 0)
        connections = knowledge_connections.get("connections", 0)
        
        return min(1.0, (centrality * 0.7) + (connections / 100 * 0.3))
    
    async def _calculate_learning_velocity(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate learning velocity score"""
        if len(historical_data) < 2:
            return 0.0
        
        # Calculate improvement over time
        # This is a simplified version
        recent_performance = np.mean([item.get("performance", 0) for item in historical_data[-5:]])
        early_performance = np.mean([item.get("performance", 0) for item in historical_data[:5]])
        
        if early_performance == 0:
            return 0.0
        
        improvement = (recent_performance - early_performance) / early_performance
        return min(1.0, max(0.0, improvement))
    
    async def _calculate_problem_solving_ability(self, recent_outcomes: List[Dict[str, Any]]) -> float:
        """Calculate problem solving ability score"""
        if not recent_outcomes:
            return 0.0
        
        # Calculate based on problem complexity and resolution time
        problem_scores = []
        for outcome in recent_outcomes:
            complexity = outcome.get("complexity", 0)
            resolution_time = outcome.get("resolution_time", 0)
            success = outcome.get("success", False)
            
            if success and resolution_time > 0:
                efficiency = complexity / resolution_time
                problem_scores.append(efficiency)
        
        return np.mean(problem_scores) if problem_scores else 0.0
    
    async def _extract_learning_signals(self, session: CollaborationSession) -> Dict[str, Any]:
        """Extract learning signals from collaboration session"""
        signals = {
            "domain": session.metadata.get("domain", "general"),
            "complexity": session.metadata.get("complexity", 1),
            "success": session.status == CollaborationStatus.COMPLETED,
            "duration": session.actual_duration or 0,
            "decisions_made": len(session.decisions_made),
            "training_data": len(session.training_data_generated)
        }
        
        return signals
    
    async def _update_expertise_areas(self, user_id: str, learning_signals: Dict[str, Any]):
        """Update expertise areas based on learning signals"""
        # This would update the expert's expertise areas
        # based on successful collaborations in new domains
        pass
    
    async def _update_performance_metrics(self, user_id: str, session: CollaborationSession):
        """Update performance metrics"""
        expert_profile = await self.get_expert_profile(user_id)
        if expert_profile:
            # Update metrics based on session outcome
            if session.status == CollaborationStatus.COMPLETED:
                expert_profile.performance_metrics["consultation_count"] += 1
                # Update success rate, response time, etc.
                await self._store_expert_profile(expert_profile)
    
    async def _store_expertise_memory(self, user_id: str, learning_signals: Dict[str, Any]):
        """Store expertise learning as memory"""
        memory = AgentMemory(
            memory_id=f"expertise_{user_id}_{datetime.utcnow().timestamp()}",
            agent_id=user_id,
            memory_type=MemoryType.EXPERTISE,
            content=f"Learning from collaboration: {learning_signals}",
            context=learning_signals,
            importance=0.7,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1
        )
        
        await self.memory_service.store_memory(memory)
    
    async def _update_consultation_metrics(self, expert_id: str, outcome: Dict[str, Any]):
        """Update consultation metrics"""
        expert_profile = await self.get_expert_profile(expert_id)
        if expert_profile:
            # Update metrics based on outcome
            success = outcome.get("success", False)
            duration = outcome.get("duration", 0)
            satisfaction = outcome.get("satisfaction", 0)
            
            # Update running averages
            current_count = expert_profile.performance_metrics.get("consultation_count", 0)
            current_success_rate = expert_profile.performance_metrics.get("success_rate", 0)
            
            new_count = current_count + 1
            new_success_rate = ((current_success_rate * current_count) + (1 if success else 0)) / new_count
            
            expert_profile.performance_metrics["consultation_count"] = new_count
            expert_profile.performance_metrics["success_rate"] = new_success_rate
            
            if duration > 0:
                current_response_time = expert_profile.performance_metrics.get("average_response_time", 0)
                new_response_time = ((current_response_time * current_count) + duration) / new_count
                expert_profile.performance_metrics["average_response_time"] = new_response_time
            
            if satisfaction > 0:
                current_satisfaction = expert_profile.performance_metrics.get("satisfaction_score", 0)
                new_satisfaction = ((current_satisfaction * current_count) + satisfaction) / new_count
                expert_profile.performance_metrics["satisfaction_score"] = new_satisfaction
            
            await self._store_expert_profile(expert_profile)

# Global service instance
expert_identification_service = ExpertIdentificationService() 