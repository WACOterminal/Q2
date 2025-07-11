import logging
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from agentQ.app.core.toolbox import Tool
from shared.q_messaging_schemas.schemas import TaskAnnouncement, AgentBid
from shared.ignite_client import shared_ignite_client
from shared.pulsar_client import shared_pulsar_client

logger = logging.getLogger(__name__)

# --- Enums for Task Analysis ---
class TaskComplexity(Enum):
    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    EXPERT = 5

class TaskUrgency(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# --- Data Classes ---
@dataclass
class AgentCapabilityProfile:
    """Profile of agent's capabilities and performance history"""
    agent_id: str
    specializations: List[str]
    skill_levels: Dict[str, float]  # skill -> proficiency (0-1)
    historical_performance: Dict[str, float]  # metric -> value
    resource_constraints: Dict[str, float]  # resource -> availability
    success_rate: float
    average_completion_time: float
    reliability_score: float

@dataclass
class TaskAnalysis:
    """Detailed analysis of a task"""
    complexity: TaskComplexity
    urgency: TaskUrgency
    required_skills: List[str]
    estimated_duration: float  # hours
    resource_requirements: Dict[str, float]
    dependencies: List[str]
    risk_factors: List[str]
    potential_value: float

@dataclass
class MarketContext:
    """Current market conditions for bidding"""
    average_bid_value: float
    competition_level: float  # 0-1
    demand_supply_ratio: float
    recent_winning_bids: List[float]
    agent_availability: int

class SophisticatedBiddingEngine:
    """Advanced bidding algorithm that considers multiple factors"""
    
    def __init__(self):
        self.capability_cache = {}
        self.market_data_cache = {}
        self.performance_history = {}
        self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize ML models for bid optimization"""
        # In production, these would be trained models
        # For now, we'll use heuristic-based scoring
        self.bid_optimizer = self._create_bid_optimizer()
        self.task_analyzer = self._create_task_analyzer()
    
    def _create_bid_optimizer(self):
        """Create a bid optimization model"""
        # This would be a trained model in production
        # Using a simple scoring function for now
        def optimize_bid(features: Dict[str, float]) -> float:
            # Features: skill_match, resource_availability, competition, urgency, etc.
            base_score = 100.0
            
            # Apply feature weights
            weights = {
                'skill_match': 0.3,
                'resource_availability': 0.2,
                'historical_performance': 0.2,
                'competition_factor': 0.15,
                'urgency_factor': 0.15
            }
            
            weighted_score = sum(
                features.get(key, 0.5) * weight 
                for key, weight in weights.items()
            )
            
            return base_score * (1.0 / max(0.1, weighted_score))
        
        return optimize_bid
    
    def _create_task_analyzer(self):
        """Create a task complexity analyzer"""
        def analyze_task(task_description: str, requirements: Dict) -> TaskAnalysis:
            # Analyze task complexity based on keywords and requirements
            complexity_keywords = {
                TaskComplexity.TRIVIAL: ['simple', 'basic', 'quick', 'easy'],
                TaskComplexity.SIMPLE: ['standard', 'routine', 'common'],
                TaskComplexity.MODERATE: ['moderate', 'typical', 'normal'],
                TaskComplexity.COMPLEX: ['complex', 'advanced', 'detailed'],
                TaskComplexity.EXPERT: ['expert', 'specialized', 'critical', 'novel']
            }
            
            # Determine complexity
            complexity = TaskComplexity.MODERATE
            task_lower = task_description.lower()
            for level, keywords in complexity_keywords.items():
                if any(keyword in task_lower for keyword in keywords):
                    complexity = level
                    break
            
            # Determine urgency
            urgency = TaskUrgency.MEDIUM
            if 'urgent' in task_lower or 'asap' in task_lower:
                urgency = TaskUrgency.HIGH
            elif 'critical' in task_lower or 'emergency' in task_lower:
                urgency = TaskUrgency.CRITICAL
            
            # Extract required skills
            skill_keywords = ['python', 'javascript', 'ml', 'ai', 'data', 'api', 
                            'database', 'frontend', 'backend', 'devops', 'security']
            required_skills = [skill for skill in skill_keywords if skill in task_lower]
            
            # Estimate duration based on complexity
            duration_map = {
                TaskComplexity.TRIVIAL: 0.5,
                TaskComplexity.SIMPLE: 2.0,
                TaskComplexity.MODERATE: 8.0,
                TaskComplexity.COMPLEX: 24.0,
                TaskComplexity.EXPERT: 40.0
            }
            
            return TaskAnalysis(
                complexity=complexity,
                urgency=urgency,
                required_skills=required_skills or ['general'],
                estimated_duration=duration_map[complexity],
                resource_requirements={'cpu': 0.5, 'memory': 0.3},
                dependencies=[],
                risk_factors=[],
                potential_value=100.0 * complexity.value
            )
        
        return analyze_task
    
    async def get_agent_capability_profile(self, agent_id: str) -> AgentCapabilityProfile:
        """Retrieve or build agent capability profile"""
        # Check cache first
        if agent_id in self.capability_cache:
            cached = self.capability_cache[agent_id]
            if cached['timestamp'] > time.time() - 300:  # 5 min cache
                return cached['profile']
        
        # Fetch from Ignite
        try:
            profile_data = await shared_ignite_client.get(f"agent_profile:{agent_id}")
            if profile_data:
                profile = AgentCapabilityProfile(**json.loads(profile_data))
            else:
                # Build default profile
                profile = self._build_default_profile(agent_id)
        except Exception as e:
            logger.error(f"Failed to fetch agent profile: {e}")
            profile = self._build_default_profile(agent_id)
        
        # Cache the profile
        self.capability_cache[agent_id] = {
            'profile': profile,
            'timestamp': time.time()
        }
        
        return profile
    
    def _build_default_profile(self, agent_id: str) -> AgentCapabilityProfile:
        """Build a default capability profile for new agents"""
        return AgentCapabilityProfile(
            agent_id=agent_id,
            specializations=['general'],
            skill_levels={'general': 0.7},
            historical_performance={
                'tasks_completed': 0,
                'average_rating': 4.0,
                'on_time_delivery': 0.8
            },
            resource_constraints={
                'cpu': 1.0,
                'memory': 1.0,
                'concurrent_tasks': 3
            },
            success_rate=0.8,
            average_completion_time=8.0,
            reliability_score=0.75
        )
    
    async def get_market_context(self, task_type: str) -> MarketContext:
        """Get current market conditions for bidding"""
        # Check cache
        cache_key = f"market:{task_type}"
        if cache_key in self.market_data_cache:
            cached = self.market_data_cache[cache_key]
            if cached['timestamp'] > time.time() - 60:  # 1 min cache
                return cached['context']
        
        # Fetch recent bids from Ignite
        try:
            recent_bids_data = await shared_ignite_client.get(f"recent_bids:{task_type}")
            if recent_bids_data:
                recent_bids = json.loads(recent_bids_data)
                winning_bids = [b['value'] for b in recent_bids if b.get('won', False)]
                avg_bid = np.mean(winning_bids) if winning_bids else 100.0
                
                context = MarketContext(
                    average_bid_value=avg_bid,
                    competition_level=min(1.0, len(recent_bids) / 10),
                    demand_supply_ratio=1.2,  # Would calculate from actual data
                    recent_winning_bids=winning_bids[-10:],
                    agent_availability=5  # Would query actual availability
                )
            else:
                # Default market context
                context = MarketContext(
                    average_bid_value=100.0,
                    competition_level=0.5,
                    demand_supply_ratio=1.0,
                    recent_winning_bids=[],
                    agent_availability=10
                )
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            context = MarketContext(
                average_bid_value=100.0,
                competition_level=0.5,
                demand_supply_ratio=1.0,
                recent_winning_bids=[],
                agent_availability=10
            )
        
        # Cache the context
        self.market_data_cache[cache_key] = {
            'context': context,
            'timestamp': time.time()
        }
        
        return context
    
    def calculate_skill_match_score(self, required_skills: List[str], 
                                  agent_skills: Dict[str, float]) -> float:
        """Calculate how well agent skills match task requirements"""
        if not required_skills:
            return 0.8  # Default match for general tasks
        
        match_scores = []
        for skill in required_skills:
            if skill in agent_skills:
                match_scores.append(agent_skills[skill])
            else:
                # Check for related skills
                related_score = 0.0
                for agent_skill, proficiency in agent_skills.items():
                    similarity = self._calculate_skill_similarity(skill, agent_skill)
                    related_score = max(related_score, similarity * proficiency)
                match_scores.append(related_score)
        
        return np.mean(match_scores) if match_scores else 0.0
    
    def _calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity between two skills"""
        # Simple heuristic - in production, use embeddings or knowledge graph
        skill_groups = {
            'programming': ['python', 'javascript', 'java', 'c++', 'coding'],
            'data': ['data', 'analytics', 'ml', 'ai', 'statistics'],
            'web': ['frontend', 'backend', 'api', 'web', 'http'],
            'infrastructure': ['devops', 'cloud', 'docker', 'kubernetes', 'aws']
        }
        
        for group, skills in skill_groups.items():
            if skill1 in skills and skill2 in skills:
                return 0.8
        
        return 0.0
    
    def calculate_resource_availability_score(self, required_resources: Dict[str, float],
                                           available_resources: Dict[str, float]) -> float:
        """Calculate resource availability score"""
        scores = []
        for resource, required in required_resources.items():
            available = available_resources.get(resource, 0.0)
            if available >= required:
                scores.append(1.0)
            else:
                scores.append(available / required if required > 0 else 0.0)
        
        return np.mean(scores) if scores else 1.0
    
    def apply_strategic_adjustments(self, base_bid: float, task_analysis: TaskAnalysis,
                                  market_context: MarketContext, agent_profile: AgentCapabilityProfile) -> float:
        """Apply strategic adjustments to the bid"""
        adjusted_bid = base_bid
        
        # Urgency adjustment
        urgency_multipliers = {
            TaskUrgency.LOW: 1.1,
            TaskUrgency.MEDIUM: 1.0,
            TaskUrgency.HIGH: 0.9,
            TaskUrgency.CRITICAL: 0.8
        }
        adjusted_bid *= urgency_multipliers[task_analysis.urgency]
        
        # Competition adjustment
        if market_context.competition_level > 0.7:
            # High competition - be more aggressive
            adjusted_bid *= 0.95
        elif market_context.competition_level < 0.3:
            # Low competition - can bid higher
            adjusted_bid *= 1.05
        
        # Historical performance adjustment
        if agent_profile.success_rate > 0.9:
            # High success rate - can command premium
            adjusted_bid *= 1.1
        elif agent_profile.success_rate < 0.7:
            # Lower success rate - need to be competitive
            adjusted_bid *= 0.9
        
        # Market dynamics adjustment
        if market_context.recent_winning_bids:
            avg_winning = np.mean(market_context.recent_winning_bids)
            if adjusted_bid > avg_winning * 1.2:
                # Too high compared to market
                adjusted_bid = avg_winning * 1.1
            elif adjusted_bid < avg_winning * 0.5:
                # Too low - might indicate error
                adjusted_bid = avg_winning * 0.8
        
        return adjusted_bid
    
    async def generate_bid(self, task_announcement: TaskAnnouncement, 
                          agent_context: Dict[str, Any]) -> Tuple[AgentBid, Dict[str, Any]]:
        """Generate a sophisticated bid for the task"""
        agent_id = agent_context.get("agent_id", "unknown_agent")
        
        # Analyze the task
        task_analysis = self.task_analyzer(
            task_announcement.task_prompt,
            task_announcement.requirements
        )
        
        # Get agent capability profile
        agent_profile = await self.get_agent_capability_profile(agent_id)
        
        # Get market context
        market_context = await self.get_market_context(task_announcement.task_personality)
        
        # Calculate skill match score
        skill_match = self.calculate_skill_match_score(
            task_analysis.required_skills,
            agent_profile.skill_levels
        )
        
        # Calculate resource availability
        resource_availability = self.calculate_resource_availability_score(
            task_analysis.resource_requirements,
            agent_profile.resource_constraints
        )
        
        # Prepare features for bid optimization
        features = {
            'skill_match': skill_match,
            'resource_availability': resource_availability,
            'historical_performance': agent_profile.success_rate,
            'competition_factor': 1.0 - market_context.competition_level,
            'urgency_factor': task_analysis.urgency.value / 4.0
        }
        
        # Calculate base bid using optimizer
        base_bid = self.bid_optimizer(features)
        
        # Apply strategic adjustments
        final_bid = self.apply_strategic_adjustments(
            base_bid, task_analysis, market_context, agent_profile
        )
        
        # Calculate confidence score
        confidence = (skill_match * 0.4 + 
                     resource_availability * 0.3 + 
                     agent_profile.reliability_score * 0.3)
        
        # Determine if we can meet requirements
        can_meet = (skill_match >= 0.6 and 
                   resource_availability >= 0.7 and
                   agent_context.get("load_factor", 0.5) < 0.8)
        
        # Create bid object
        bid = AgentBid(
            task_id=task_announcement.task_id,
            agent_id=agent_id,
            bid_value=round(final_bid, 2),
            can_meet_requirements=str(can_meet),
            confidence_score=round(confidence, 3),
            current_load_factor=agent_context.get("load_factor", 0.5),
            timestamp=int(time.time() * 1000)
        )
        
        # Prepare detailed analysis for logging
        analysis = {
            'task_complexity': task_analysis.complexity.name,
            'task_urgency': task_analysis.urgency.name,
            'skill_match_score': round(skill_match, 3),
            'resource_availability_score': round(resource_availability, 3),
            'market_competition': round(market_context.competition_level, 3),
            'base_bid': round(base_bid, 2),
            'final_bid': round(final_bid, 2),
            'adjustments_applied': round(final_bid - base_bid, 2)
        }
        
        # Store bid history for learning
        await self._store_bid_history(bid, analysis)
        
        return bid, analysis
    
    async def _store_bid_history(self, bid: AgentBid, analysis: Dict[str, Any]):
        """Store bid history for future learning and optimization"""
        try:
            bid_record = {
                'bid': bid.dict(),
                'analysis': analysis,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store in Ignite
            key = f"bid_history:{bid.agent_id}:{bid.task_id}"
            await shared_ignite_client.put(key, json.dumps(bid_record))
            
            # Send to analytics topic
            await shared_pulsar_client.send_message(
                "persistent://public/default/bid-analytics",
                bid_record
            )
        except Exception as e:
            logger.error(f"Failed to store bid history: {e}")

# Global instance
bidding_engine = SophisticatedBiddingEngine()

async def estimate_cost_and_generate_bid(task_announcement: TaskAnnouncement, 
                                        agent_context: Dict[str, Any]) -> AgentBid:
    """
    Analyzes a task announcement and generates a sophisticated bid based on:
    - Agent capabilities and historical performance
    - Task complexity and requirements
    - Market conditions and competition
    - Resource availability and constraints
    
    Args:
        task_announcement: The task details broadcast by the manager
        agent_context: The current context of the agent
        
    Returns:
        AgentBid: A sophisticated bid object
    """
    agent_id = agent_context.get("agent_id", "unknown_agent")
    logger.info(f"Agent {agent_id} generating sophisticated bid for task {task_announcement.task_id}")
    
    try:
        bid, analysis = await bidding_engine.generate_bid(task_announcement, agent_context)
        
        logger.info(
            f"Agent {agent_id} generated bid for task {task_announcement.task_id}: "
            f"value={bid.bid_value}, confidence={bid.confidence_score}, "
            f"complexity={analysis['task_complexity']}, skill_match={analysis['skill_match_score']}"
        )
        
        return bid
        
    except Exception as e:
        logger.error(f"Failed to generate sophisticated bid: {e}", exc_info=True)
        
        # Fallback to simple bid
        base_cost = 100.0
        current_load = agent_context.get("load_factor", 0.5)
        fallback_bid_value = base_cost / (0.7 * (1 - current_load))
        
        return AgentBid(
            task_id=task_announcement.task_id,
            agent_id=agent_id,
            bid_value=round(fallback_bid_value, 2),
            can_meet_requirements="True",
            confidence_score=0.5,
            current_load_factor=current_load,
            timestamp=int(time.time() * 1000)
        )

# Synchronous wrapper for backward compatibility
def estimate_cost_and_generate_bid_sync(task_announcement: TaskAnnouncement, 
                                       agent_context: Dict[str, Any]) -> AgentBid:
    """Synchronous wrapper for the async bid generation function"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        estimate_cost_and_generate_bid(task_announcement, agent_context)
    )

# --- Tool Registration Object ---
estimate_cost_tool = Tool(
    name="estimate_task_cost",
    description="Analyzes tasks and generates sophisticated cost-based bids using ML-driven optimization.",
    func=estimate_cost_and_generate_bid_sync
) 