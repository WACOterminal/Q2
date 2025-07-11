"""
Knowledge Graph-Driven Agent Planning Service

This service provides advanced agent planning capabilities using knowledge graph reasoning:
- Context-aware planning using graph traversal
- Dynamic strategy selection based on graph analysis
- Complex pattern matching for similar scenarios
- Dependency-aware task decomposition
- Risk assessment using historical data
- Resource optimization through graph insights
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import uuid
from collections import defaultdict
import networkx as nx

# Q Platform imports
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.q_workflow_schemas.models import Workflow, WorkflowStep, WorkflowStatus
from shared.q_memory_schemas.memory_models import AgentMemory, MemoryType
from app.services.memory_service import MemoryService
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.pulsar_service import PulsarService

logger = logging.getLogger(__name__)

class PlanningComplexity(Enum):
    """Complexity levels for planning tasks"""
    SIMPLE = "simple"           # Single-step, straightforward
    MODERATE = "moderate"       # Multi-step, some dependencies
    COMPLEX = "complex"         # Multi-agent, complex dependencies
    EXPERT = "expert"           # Requires specialized knowledge
    CRITICAL = "critical"       # High-stakes, needs careful planning

class PlanningStrategy(Enum):
    """Different planning strategies"""
    SEQUENTIAL = "sequential"           # Linear execution
    PARALLEL = "parallel"               # Parallel execution
    CONDITIONAL = "conditional"         # Branching logic
    ITERATIVE = "iterative"             # Loops and refinement
    COLLABORATIVE = "collaborative"     # Multi-agent coordination
    ADAPTIVE = "adaptive"               # Dynamic adjustment

class RiskLevel(Enum):
    """Risk levels for planning decisions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PlanningContext:
    """Context for planning decisions"""
    request_id: str
    original_prompt: str
    user_id: str
    domain: str
    complexity: PlanningComplexity
    constraints: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    historical_patterns: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PlanningInsight:
    """Insight from knowledge graph analysis"""
    insight_type: str
    confidence: float
    data: Dict[str, Any]
    source_query: str
    reasoning: str
    impact_score: float

@dataclass
class PlanningRecommendation:
    """Planning recommendation based on analysis"""
    strategy: PlanningStrategy
    confidence: float
    reasoning: str
    estimated_duration: int
    resource_requirements: Dict[str, Any]
    risk_level: RiskLevel
    success_probability: float
    alternative_strategies: List[str]

@dataclass
class EnhancedPlan:
    """Enhanced plan with knowledge graph insights"""
    plan_id: str
    context: PlanningContext
    strategy: PlanningStrategy
    workflow: Workflow
    insights: List[PlanningInsight]
    recommendations: List[PlanningRecommendation]
    risk_assessment: Dict[str, Any]
    success_metrics: Dict[str, Any]
    adaptive_triggers: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

class KnowledgeGraphPlanner:
    """
    Advanced agent planning service using knowledge graph reasoning
    """
    
    def __init__(self, 
                 kg_client: KnowledgeGraphClient,
                 memory_service: MemoryService,
                 pulsar_service: PulsarService):
        self.kg_client = kg_client
        self.memory_service = memory_service
        self.pulsar_service = pulsar_service
        
        # Planning caches
        self.pattern_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.strategy_cache: Dict[str, PlanningStrategy] = {}
        self.insight_cache: Dict[str, List[PlanningInsight]] = {}
        
        # Planning statistics
        self.planning_stats = {
            "total_plans": 0,
            "successful_plans": 0,
            "failed_plans": 0,
            "average_planning_time": 0.0,
            "strategy_distribution": defaultdict(int)
        }
        
        logger.info("Knowledge Graph Planner initialized")
    
    async def initialize(self):
        """Initialize the planning service"""
        logger.info("Initializing Knowledge Graph Planner")
        
        # Build initial pattern cache
        await self._build_pattern_cache()
        
        # Subscribe to planning events
        await self._subscribe_to_planning_events()
        
        logger.info("Knowledge Graph Planner initialized successfully")
    
    # ===== CORE PLANNING METHODS =====
    
    async def create_enhanced_plan(
        self,
        prompt: str,
        user_id: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict[str, Any]] = None
    ) -> EnhancedPlan:
        """
        Create an enhanced plan using knowledge graph reasoning
        
        Args:
            prompt: The planning request
            user_id: User making the request
            domain: Domain context (e.g., "devops", "data", "security")
            constraints: Planning constraints
            resources: Available resources
            
        Returns:
            Enhanced plan with KG insights
        """
        start_time = datetime.utcnow()
        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Creating enhanced plan {plan_id} for domain: {domain}")
        
        try:
            # 1. Analyze the request and build context
            context = await self._analyze_request(
                plan_id, prompt, user_id, domain, constraints, resources
            )
            
            # 2. Extract insights from knowledge graph
            insights = await self._extract_planning_insights(context)
            
            # 3. Determine optimal strategy
            strategy = await self._determine_strategy(context, insights)
            
            # 4. Generate recommendations
            recommendations = await self._generate_recommendations(context, insights, strategy)
            
            # 5. Create workflow based on strategy
            workflow = await self._create_workflow(context, strategy, insights)
            
            # 6. Perform risk assessment
            risk_assessment = await self._assess_risks(context, workflow, insights)
            
            # 7. Define success metrics
            success_metrics = await self._define_success_metrics(context, workflow)
            
            # 8. Create adaptive triggers
            adaptive_triggers = await self._create_adaptive_triggers(context, workflow)
            
            # Create enhanced plan
            enhanced_plan = EnhancedPlan(
                plan_id=plan_id,
                context=context,
                strategy=strategy,
                workflow=workflow,
                insights=insights,
                recommendations=recommendations,
                risk_assessment=risk_assessment,
                success_metrics=success_metrics,
                adaptive_triggers=adaptive_triggers,
                created_at=start_time,
                updated_at=start_time
            )
            
            # Store plan in knowledge graph
            await self._store_plan_in_kg(enhanced_plan)
            
            # Update statistics
            self.planning_stats["total_plans"] += 1
            self.planning_stats["strategy_distribution"][strategy.value] += 1
            
            planning_time = (datetime.utcnow() - start_time).total_seconds()
            self.planning_stats["average_planning_time"] = (
                (self.planning_stats["average_planning_time"] * (self.planning_stats["total_plans"] - 1) + planning_time) 
                / self.planning_stats["total_plans"]
            )
            
            logger.info(f"Enhanced plan {plan_id} created successfully in {planning_time:.2f}s")
            
            return enhanced_plan
            
        except Exception as e:
            logger.error(f"Error creating enhanced plan {plan_id}: {e}", exc_info=True)
            raise
    
    # ===== ANALYSIS METHODS =====
    
    async def _analyze_request(
        self,
        plan_id: str,
        prompt: str,
        user_id: str,
        domain: str,
        constraints: Optional[Dict[str, Any]],
        resources: Optional[Dict[str, Any]]
    ) -> PlanningContext:
        """Analyze the planning request and build context"""
        
        # Determine complexity using graph analysis
        complexity = await self._assess_complexity(prompt, domain)
        
        # Find historical patterns
        historical_patterns = await self._find_historical_patterns(prompt, domain)
        
        # Identify dependencies
        dependencies = await self._identify_dependencies(prompt, domain)
        
        # Assess risk factors
        risk_factors = await self._identify_risk_factors(prompt, domain, historical_patterns)
        
        # Define success criteria
        success_criteria = await self._define_success_criteria(prompt, domain)
        
        context = PlanningContext(
            request_id=plan_id,
            original_prompt=prompt,
            user_id=user_id,
            domain=domain,
            complexity=complexity,
            constraints=constraints or {},
            resources=resources or {},
            historical_patterns=historical_patterns,
            dependencies=dependencies,
            risk_factors=risk_factors,
            success_criteria=success_criteria
        )
        
        return context
    
    async def _assess_complexity(self, prompt: str, domain: str) -> PlanningComplexity:
        """Assess the complexity of the planning request"""
        
        # Use knowledge graph to analyze complexity patterns
        query = f"""
        g.V().hasLabel('Task')
         .has('domain', '{domain}')
         .has('description', textContains('{prompt[:50]}'))
         .values('complexity', 'execution_time', 'resource_count')
         .fold()
        """
        
        try:
            result = await self.kg_client.execute_gremlin_query(query)
            
            if result.get("result"):
                # Analyze patterns to determine complexity
                data = result["result"]
                avg_time = sum(d.get("execution_time", 0) for d in data) / len(data)
                avg_resources = sum(d.get("resource_count", 0) for d in data) / len(data)
                
                if avg_time > 3600 or avg_resources > 5:  # > 1 hour or > 5 resources
                    return PlanningComplexity.COMPLEX
                elif avg_time > 900 or avg_resources > 2:  # > 15 min or > 2 resources
                    return PlanningComplexity.MODERATE
                else:
                    return PlanningComplexity.SIMPLE
                    
        except Exception as e:
            logger.warning(f"Error assessing complexity: {e}")
        
        # Fallback to keyword analysis
        complex_keywords = ["deploy", "migrate", "scale", "secure", "analyze", "optimize"]
        critical_keywords = ["production", "critical", "emergency", "security"]
        
        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in critical_keywords):
            return PlanningComplexity.CRITICAL
        elif any(keyword in prompt_lower for keyword in complex_keywords):
            return PlanningComplexity.COMPLEX
        else:
            return PlanningComplexity.MODERATE
    
    async def _find_historical_patterns(self, prompt: str, domain: str) -> List[Dict[str, Any]]:
        """Find historical patterns similar to current request"""
        
        if f"{domain}_{prompt[:30]}" in self.pattern_cache:
            return self.pattern_cache[f"{domain}_{prompt[:30]}"]
        
        query = f"""
        g.V().hasLabel('Workflow')
         .has('domain', '{domain}')
         .has('status', 'completed')
         .where(values('description').is(textContains('{prompt[:50]}')))
         .project('workflow_id', 'strategy', 'duration', 'success_rate', 'resources')
         .by('workflow_id')
         .by('strategy')
         .by('duration')
         .by('success_rate')
         .by(out('USED_RESOURCE').count())
         .order().by('success_rate', desc)
         .limit(10)
        """
        
        try:
            result = await self.kg_client.execute_gremlin_query(query)
            patterns = result.get("result", [])
            
            # Cache the results
            self.pattern_cache[f"{domain}_{prompt[:30]}"] = patterns
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error finding historical patterns: {e}")
            return []
    
    async def _identify_dependencies(self, prompt: str, domain: str) -> List[str]:
        """Identify dependencies for the planning request"""
        
        query = f"""
        g.V().hasLabel('Service', 'Component')
         .has('domain', '{domain}')
         .where(values('name', 'description').is(textContains('{prompt[:50]}')))
         .bothE('DEPENDS_ON', 'REQUIRES')
         .otherV()
         .values('name')
         .dedup()
         .limit(20)
        """
        
        try:
            result = await self.kg_client.execute_gremlin_query(query)
            return result.get("result", [])
            
        except Exception as e:
            logger.warning(f"Error identifying dependencies: {e}")
            return []
    
    async def _identify_risk_factors(
        self,
        prompt: str,
        domain: str,
        historical_patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify risk factors based on historical data"""
        
        risk_factors = []
        
        # Analyze historical failures
        query = f"""
        g.V().hasLabel('Workflow')
         .has('domain', '{domain}')
         .has('status', 'failed')
         .where(values('description').is(textContains('{prompt[:50]}')))
         .values('failure_reason')
         .groupCount()
         .order(local).by(values, desc)
        """
        
        try:
            result = await self.kg_client.execute_gremlin_query(query)
            failure_patterns = result.get("result", {})
            
            for reason, count in failure_patterns.items():
                if count > 2:  # Recurring failure pattern
                    risk_factors.append(f"Historical failure: {reason}")
                    
        except Exception as e:
            logger.warning(f"Error identifying risk factors: {e}")
        
        # Add domain-specific risk factors
        domain_risks = {
            "devops": ["deployment_failure", "rollback_required", "service_downtime"],
            "data": ["data_corruption", "pipeline_failure", "privacy_breach"],
            "security": ["unauthorized_access", "data_leak", "compliance_violation"]
        }
        
        risk_factors.extend(domain_risks.get(domain, []))
        
        return risk_factors
    
    async def _define_success_criteria(self, prompt: str, domain: str) -> Dict[str, Any]:
        """Define success criteria for the planning request"""
        
        # Extract success criteria from similar successful workflows
        query = f"""
        g.V().hasLabel('Workflow')
         .has('domain', '{domain}')
         .has('status', 'completed')
         .has('success_rate', gte(0.8))
         .where(values('description').is(textContains('{prompt[:50]}')))
         .values('success_metrics')
         .fold()
        """
        
        try:
            result = await self.kg_client.execute_gremlin_query(query)
            metrics_data = result.get("result", [])
            
            if metrics_data:
                # Aggregate common success metrics
                common_metrics = {}
                for metrics in metrics_data:
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            if key not in common_metrics:
                                common_metrics[key] = []
                            common_metrics[key].append(value)
                
                # Calculate average/target values
                success_criteria = {}
                for key, values in common_metrics.items():
                    if all(isinstance(v, (int, float)) for v in values):
                        success_criteria[key] = sum(values) / len(values)
                    else:
                        success_criteria[key] = max(set(values), key=values.count)
                
                return success_criteria
                
        except Exception as e:
            logger.warning(f"Error defining success criteria: {e}")
        
        # Default success criteria
        return {
            "completion_time": 3600,  # 1 hour
            "success_rate": 0.95,
            "resource_efficiency": 0.8,
            "error_rate": 0.05
        }
    
    # ===== INSIGHT EXTRACTION =====
    
    async def _extract_planning_insights(self, context: PlanningContext) -> List[PlanningInsight]:
        """Extract insights from knowledge graph for planning"""
        
        insights = []
        
        # Get domain expertise insights
        domain_insights = await self._get_domain_expertise_insights(context)
        insights.extend(domain_insights)
        
        # Get resource optimization insights
        resource_insights = await self._get_resource_optimization_insights(context)
        insights.extend(resource_insights)
        
        # Get timing insights
        timing_insights = await self._get_timing_insights(context)
        insights.extend(timing_insights)
        
        # Get collaboration insights
        collaboration_insights = await self._get_collaboration_insights(context)
        insights.extend(collaboration_insights)
        
        return insights
    
    async def _get_domain_expertise_insights(self, context: PlanningContext) -> List[PlanningInsight]:
        """Get domain expertise insights"""
        
        query = f"""
        g.V().hasLabel('Agent')
         .has('domain', '{context.domain}')
         .has('expertise_level', gte(0.8))
         .project('agent_id', 'capabilities', 'success_rate', 'avg_completion_time')
         .by('agent_id')
         .by('capabilities')
         .by('success_rate')
         .by('avg_completion_time')
         .order().by('success_rate', desc)
         .limit(5)
        """
        
        try:
            result = await self.kg_client.execute_gremlin_query(query)
            expert_data = result.get("result", [])
            
            if expert_data:
                insight = PlanningInsight(
                    insight_type="domain_expertise",
                    confidence=0.9,
                    data={"experts": expert_data},
                    source_query=query,
                    reasoning=f"Found {len(expert_data)} expert agents in {context.domain}",
                    impact_score=0.8
                )
                return [insight]
                
        except Exception as e:
            logger.warning(f"Error getting domain expertise insights: {e}")
        
        return []
    
    async def _get_resource_optimization_insights(self, context: PlanningContext) -> List[PlanningInsight]:
        """Get resource optimization insights"""
        
        query = f"""
        g.V().hasLabel('Resource')
         .has('domain', '{context.domain}')
         .where(out('USED_BY').hasLabel('Workflow').has('status', 'completed'))
         .project('resource_id', 'type', 'utilization', 'cost_efficiency')
         .by('resource_id')
         .by('type')
         .by('utilization')
         .by('cost_efficiency')
         .order().by('cost_efficiency', desc)
         .limit(10)
        """
        
        try:
            result = await self.kg_client.execute_gremlin_query(query)
            resource_data = result.get("result", [])
            
            if resource_data:
                insight = PlanningInsight(
                    insight_type="resource_optimization",
                    confidence=0.85,
                    data={"optimal_resources": resource_data},
                    source_query=query,
                    reasoning="Identified cost-efficient resources based on historical usage",
                    impact_score=0.7
                )
                return [insight]
                
        except Exception as e:
            logger.warning(f"Error getting resource optimization insights: {e}")
        
        return []
    
    async def _get_timing_insights(self, context: PlanningContext) -> List[PlanningInsight]:
        """Get timing insights for optimal execution"""
        
        query = f"""
        g.V().hasLabel('Workflow')
         .has('domain', '{context.domain}')
         .has('status', 'completed')
         .project('hour', 'avg_duration', 'success_rate')
         .by('start_hour')
         .by('duration')
         .by('success_rate')
         .groupCount()
         .order(local).by(values, desc)
        """
        
        try:
            result = await self.kg_client.execute_gremlin_query(query)
            timing_data = result.get("result", {})
            
            if timing_data:
                best_time = max(timing_data.keys(), key=lambda x: timing_data[x])
                insight = PlanningInsight(
                    insight_type="optimal_timing",
                    confidence=0.7,
                    data={"optimal_hour": best_time, "timing_patterns": timing_data},
                    source_query=query,
                    reasoning=f"Historical data shows optimal execution time at hour {best_time}",
                    impact_score=0.5
                )
                return [insight]
                
        except Exception as e:
            logger.warning(f"Error getting timing insights: {e}")
        
        return []
    
    async def _get_collaboration_insights(self, context: PlanningContext) -> List[PlanningInsight]:
        """Get collaboration insights for multi-agent scenarios"""
        
        query = f"""
        g.V().hasLabel('Workflow')
         .has('domain', '{context.domain}')
         .has('agent_count', gte(2))
         .has('status', 'completed')
         .project('agent_combination', 'collaboration_score', 'efficiency')
         .by(out('EXECUTED_BY').values('agent_id').fold())
         .by('collaboration_score')
         .by('efficiency')
         .order().by('collaboration_score', desc)
         .limit(5)
        """
        
        try:
            result = await self.kg_client.execute_gremlin_query(query)
            collaboration_data = result.get("result", [])
            
            if collaboration_data:
                insight = PlanningInsight(
                    insight_type="collaboration_patterns",
                    confidence=0.8,
                    data={"successful_combinations": collaboration_data},
                    source_query=query,
                    reasoning="Found successful agent collaboration patterns",
                    impact_score=0.9
                )
                return [insight]
                
        except Exception as e:
            logger.warning(f"Error getting collaboration insights: {e}")
        
        return []
    
    # ===== STRATEGY DETERMINATION =====
    
    async def _determine_strategy(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight]
    ) -> PlanningStrategy:
        """Determine optimal planning strategy"""
        
        # Check cache first
        cache_key = f"{context.domain}_{context.complexity.value}_{len(context.dependencies)}"
        if cache_key in self.strategy_cache:
            return self.strategy_cache[cache_key]
        
        # Analyze patterns to determine strategy
        strategy_scores = defaultdict(float)
        
        # Factor in complexity
        if context.complexity == PlanningComplexity.SIMPLE:
            strategy_scores[PlanningStrategy.SEQUENTIAL] += 0.8
        elif context.complexity == PlanningComplexity.MODERATE:
            strategy_scores[PlanningStrategy.PARALLEL] += 0.7
            strategy_scores[PlanningStrategy.CONDITIONAL] += 0.6
        elif context.complexity == PlanningComplexity.COMPLEX:
            strategy_scores[PlanningStrategy.COLLABORATIVE] += 0.8
            strategy_scores[PlanningStrategy.ADAPTIVE] += 0.7
        elif context.complexity == PlanningComplexity.CRITICAL:
            strategy_scores[PlanningStrategy.ADAPTIVE] += 0.9
            strategy_scores[PlanningStrategy.COLLABORATIVE] += 0.8
        
        # Factor in dependencies
        if len(context.dependencies) > 5:
            strategy_scores[PlanningStrategy.COLLABORATIVE] += 0.3
        
        # Factor in risk factors
        if len(context.risk_factors) > 3:
            strategy_scores[PlanningStrategy.ADAPTIVE] += 0.4
        
        # Factor in historical patterns
        for pattern in context.historical_patterns:
            if pattern.get("strategy"):
                strategy_scores[PlanningStrategy(pattern["strategy"])] += 0.2
        
        # Factor in insights
        for insight in insights:
            if insight.insight_type == "collaboration_patterns":
                strategy_scores[PlanningStrategy.COLLABORATIVE] += 0.3
            elif insight.insight_type == "resource_optimization":
                strategy_scores[PlanningStrategy.PARALLEL] += 0.2
        
        # Select strategy with highest score
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        # Cache the result
        self.strategy_cache[cache_key] = best_strategy
        
        return best_strategy
    
    # ===== RECOMMENDATION GENERATION =====
    
    async def _generate_recommendations(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight],
        strategy: PlanningStrategy
    ) -> List[PlanningRecommendation]:
        """Generate planning recommendations"""
        
        recommendations = []
        
        # Primary recommendation
        primary_rec = await self._create_primary_recommendation(context, insights, strategy)
        recommendations.append(primary_rec)
        
        # Alternative recommendations
        alternative_recs = await self._create_alternative_recommendations(context, insights, strategy)
        recommendations.extend(alternative_recs)
        
        return recommendations
    
    async def _create_primary_recommendation(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight],
        strategy: PlanningStrategy
    ) -> PlanningRecommendation:
        """Create primary planning recommendation"""
        
        # Estimate duration based on historical data
        estimated_duration = await self._estimate_duration(context, strategy)
        
        # Determine resource requirements
        resource_requirements = await self._determine_resource_requirements(context, insights)
        
        # Assess risk level
        risk_level = await self._assess_risk_level(context, strategy)
        
        # Calculate success probability
        success_probability = await self._calculate_success_probability(context, insights, strategy)
        
        # Get alternative strategies
        alternative_strategies = await self._get_alternative_strategies(context, strategy)
        
        recommendation = PlanningRecommendation(
            strategy=strategy,
            confidence=0.85,
            reasoning=f"Strategy {strategy.value} selected based on complexity {context.complexity.value} and {len(insights)} insights",
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements,
            risk_level=risk_level,
            success_probability=success_probability,
            alternative_strategies=alternative_strategies
        )
        
        return recommendation
    
    async def _create_alternative_recommendations(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight],
        primary_strategy: PlanningStrategy
    ) -> List[PlanningRecommendation]:
        """Create alternative planning recommendations"""
        
        alternatives = []
        
        # Get alternative strategies
        alt_strategies = await self._get_alternative_strategies(context, primary_strategy)
        
        for alt_strategy in alt_strategies[:2]:  # Limit to 2 alternatives
            try:
                strategy_enum = PlanningStrategy(alt_strategy)
                
                alt_rec = PlanningRecommendation(
                    strategy=strategy_enum,
                    confidence=0.6,
                    reasoning=f"Alternative strategy {alt_strategy} for fallback",
                    estimated_duration=await self._estimate_duration(context, strategy_enum),
                    resource_requirements=await self._determine_resource_requirements(context, insights),
                    risk_level=await self._assess_risk_level(context, strategy_enum),
                    success_probability=await self._calculate_success_probability(context, insights, strategy_enum),
                    alternative_strategies=[]
                )
                
                alternatives.append(alt_rec)
                
            except ValueError:
                continue
        
        return alternatives
    
    # ===== WORKFLOW CREATION =====
    
    async def _create_workflow(
        self,
        context: PlanningContext,
        strategy: PlanningStrategy,
        insights: List[PlanningInsight]
    ) -> Workflow:
        """Create workflow based on planning strategy"""
        
        # Generate workflow steps based on strategy
        steps = await self._generate_workflow_steps(context, strategy, insights)
        
        # Create workflow
        workflow = Workflow(
            workflow_id=f"workflow_{uuid.uuid4().hex[:12]}",
            name=f"Enhanced Plan: {context.original_prompt[:50]}...",
            description=context.original_prompt,
            steps=steps,
            created_by=context.user_id,
            status=WorkflowStatus.PENDING,
            metadata={
                "planning_strategy": strategy.value,
                "complexity": context.complexity.value,
                "domain": context.domain,
                "insights_count": len(insights)
            }
        )
        
        return workflow
    
    async def _generate_workflow_steps(
        self,
        context: PlanningContext,
        strategy: PlanningStrategy,
        insights: List[PlanningInsight]
    ) -> List[WorkflowStep]:
        """Generate workflow steps based on strategy"""
        
        steps = []
        
        if strategy == PlanningStrategy.SEQUENTIAL:
            steps = await self._create_sequential_steps(context, insights)
        elif strategy == PlanningStrategy.PARALLEL:
            steps = await self._create_parallel_steps(context, insights)
        elif strategy == PlanningStrategy.CONDITIONAL:
            steps = await self._create_conditional_steps(context, insights)
        elif strategy == PlanningStrategy.COLLABORATIVE:
            steps = await self._create_collaborative_steps(context, insights)
        elif strategy == PlanningStrategy.ADAPTIVE:
            steps = await self._create_adaptive_steps(context, insights)
        elif strategy == PlanningStrategy.ITERATIVE:
            steps = await self._create_iterative_steps(context, insights)
        
        return steps
    
    async def _create_sequential_steps(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight]
    ) -> List[WorkflowStep]:
        """Create sequential workflow steps"""
        
        steps = []
        
        # Step 1: Initialize
        steps.append(WorkflowStep(
            step_id="init",
            name="Initialize",
            description="Initialize the workflow execution",
            agent_type="coordinator",
            dependencies=[],
            parameters={"context": asdict(context)}
        ))
        
        # Step 2: Execute main task
        steps.append(WorkflowStep(
            step_id="execute",
            name="Execute Main Task",
            description=context.original_prompt,
            agent_type=self._determine_agent_type(context.domain),
            dependencies=["init"],
            parameters={"task": context.original_prompt}
        ))
        
        # Step 3: Validate
        steps.append(WorkflowStep(
            step_id="validate",
            name="Validate Results",
            description="Validate execution results",
            agent_type="validator",
            dependencies=["execute"],
            parameters={"success_criteria": context.success_criteria}
        ))
        
        return steps
    
    async def _create_parallel_steps(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight]
    ) -> List[WorkflowStep]:
        """Create parallel workflow steps"""
        
        steps = []
        
        # Initialize
        steps.append(WorkflowStep(
            step_id="init",
            name="Initialize",
            description="Initialize parallel execution",
            agent_type="coordinator",
            dependencies=[],
            parameters={"context": asdict(context)}
        ))
        
        # Parallel execution steps
        parallel_tasks = await self._decompose_into_parallel_tasks(context)
        parallel_step_ids = []
        
        for i, task in enumerate(parallel_tasks):
            step_id = f"parallel_{i}"
            steps.append(WorkflowStep(
                step_id=step_id,
                name=f"Parallel Task {i+1}",
                description=task["description"],
                agent_type=task["agent_type"],
                dependencies=["init"],
                parameters=task["parameters"]
            ))
            parallel_step_ids.append(step_id)
        
        # Consolidate results
        steps.append(WorkflowStep(
            step_id="consolidate",
            name="Consolidate Results",
            description="Consolidate parallel execution results",
            agent_type="coordinator",
            dependencies=parallel_step_ids,
            parameters={"consolidation_strategy": "merge"}
        ))
        
        return steps
    
    async def _create_collaborative_steps(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight]
    ) -> List[WorkflowStep]:
        """Create collaborative workflow steps"""
        
        steps = []
        
        # Find optimal agent combinations from insights
        collaboration_insight = next(
            (insight for insight in insights if insight.insight_type == "collaboration_patterns"),
            None
        )
        
        if collaboration_insight:
            successful_combinations = collaboration_insight.data.get("successful_combinations", [])
            if successful_combinations:
                best_combination = successful_combinations[0]["agent_combination"]
                
                # Create collaborative steps
                steps.append(WorkflowStep(
                    step_id="setup_collaboration",
                    name="Setup Collaboration",
                    description="Setup multi-agent collaboration",
                    agent_type="coordinator",
                    dependencies=[],
                    parameters={"agents": best_combination}
                ))
                
                for i, agent in enumerate(best_combination):
                    steps.append(WorkflowStep(
                        step_id=f"collab_{i}",
                        name=f"Collaborative Task {i+1}",
                        description=f"Collaborative task for {agent}",
                        agent_type=agent,
                        dependencies=["setup_collaboration"],
                        parameters={"collaboration_mode": True}
                    ))
        
        return steps
    
    async def _create_adaptive_steps(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight]
    ) -> List[WorkflowStep]:
        """Create adaptive workflow steps"""
        
        steps = []
        
        # Adaptive execution with monitoring
        steps.append(WorkflowStep(
            step_id="adaptive_init",
            name="Adaptive Initialization",
            description="Initialize adaptive execution with monitoring",
            agent_type="coordinator",
            dependencies=[],
            parameters={
                "context": asdict(context),
                "monitoring_enabled": True,
                "adaptation_triggers": await self._create_adaptation_triggers(context)
            }
        ))
        
        # Main execution with adaptation capability
        steps.append(WorkflowStep(
            step_id="adaptive_execute",
            name="Adaptive Execution",
            description="Execute with adaptive behavior",
            agent_type=self._determine_agent_type(context.domain),
            dependencies=["adaptive_init"],
            parameters={
                "task": context.original_prompt,
                "adaptive_mode": True,
                "fallback_strategies": await self._get_fallback_strategies(context)
            }
        ))
        
        return steps
    
    async def _create_conditional_steps(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight]
    ) -> List[WorkflowStep]:
        """Create conditional workflow steps"""
        
        steps = []
        
        # Initial assessment
        steps.append(WorkflowStep(
            step_id="assess",
            name="Initial Assessment",
            description="Assess conditions for execution",
            agent_type="assessor",
            dependencies=[],
            parameters={"assessment_criteria": context.success_criteria}
        ))
        
        # Conditional execution paths
        steps.append(WorkflowStep(
            step_id="conditional_execute",
            name="Conditional Execution",
            description="Execute based on assessment results",
            agent_type=self._determine_agent_type(context.domain),
            dependencies=["assess"],
            parameters={
                "task": context.original_prompt,
                "conditions": await self._create_execution_conditions(context)
            }
        ))
        
        return steps
    
    async def _create_iterative_steps(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight]
    ) -> List[WorkflowStep]:
        """Create iterative workflow steps"""
        
        steps = []
        
        # Iterative execution with refinement
        steps.append(WorkflowStep(
            step_id="iterative_init",
            name="Iterative Initialization",
            description="Initialize iterative execution",
            agent_type="coordinator",
            dependencies=[],
            parameters={
                "max_iterations": 5,
                "improvement_threshold": 0.1
            }
        ))
        
        steps.append(WorkflowStep(
            step_id="iterative_execute",
            name="Iterative Execution",
            description="Execute with iterative refinement",
            agent_type=self._determine_agent_type(context.domain),
            dependencies=["iterative_init"],
            parameters={
                "task": context.original_prompt,
                "iterative_mode": True,
                "refinement_criteria": context.success_criteria
            }
        ))
        
        return steps
    
    # ===== HELPER METHODS =====
    
    def _determine_agent_type(self, domain: str) -> str:
        """Determine agent type based on domain"""
        agent_mapping = {
            "devops": "devops_agent",
            "data": "data_analyst_agent",
            "security": "security_agent",
            "finance": "finops_agent",
            "ml": "ml_agent",
            "general": "default_agent"
        }
        return agent_mapping.get(domain, "default_agent")
    
    async def _decompose_into_parallel_tasks(self, context: PlanningContext) -> List[Dict[str, Any]]:
        """Decompose request into parallel tasks"""
        
        # Simple decomposition for now
        tasks = [
            {
                "description": f"Parallel task 1: {context.original_prompt}",
                "agent_type": self._determine_agent_type(context.domain),
                "parameters": {"task_part": 1}
            },
            {
                "description": f"Parallel task 2: {context.original_prompt}",
                "agent_type": self._determine_agent_type(context.domain),
                "parameters": {"task_part": 2}
            }
        ]
        
        return tasks
    
    async def _create_adaptation_triggers(self, context: PlanningContext) -> List[Dict[str, Any]]:
        """Create adaptation triggers for adaptive execution"""
        
        triggers = [
            {
                "type": "performance_degradation",
                "threshold": 0.5,
                "action": "switch_strategy"
            },
            {
                "type": "resource_constraint",
                "threshold": 0.8,
                "action": "optimize_resources"
            },
            {
                "type": "error_rate",
                "threshold": 0.1,
                "action": "activate_fallback"
            }
        ]
        
        return triggers
    
    async def _get_fallback_strategies(self, context: PlanningContext) -> List[str]:
        """Get fallback strategies for adaptive execution"""
        
        return ["sequential", "simplified", "human_assisted"]
    
    async def _create_execution_conditions(self, context: PlanningContext) -> List[Dict[str, Any]]:
        """Create execution conditions for conditional workflow"""
        
        conditions = [
            {
                "condition": "success_rate > 0.8",
                "action": "continue_execution",
                "parameters": {"confidence_level": "high"}
            },
            {
                "condition": "error_rate > 0.1",
                "action": "switch_to_fallback",
                "parameters": {"fallback_strategy": "sequential"}
            }
        ]
        
        return conditions
    
    async def _estimate_duration(self, context: PlanningContext, strategy: PlanningStrategy) -> int:
        """Estimate execution duration"""
        
        # Base duration by complexity
        base_duration = {
            PlanningComplexity.SIMPLE: 300,    # 5 minutes
            PlanningComplexity.MODERATE: 900,  # 15 minutes
            PlanningComplexity.COMPLEX: 3600,  # 1 hour
            PlanningComplexity.EXPERT: 7200,   # 2 hours
            PlanningComplexity.CRITICAL: 14400 # 4 hours
        }
        
        duration = base_duration.get(context.complexity, 900)
        
        # Adjust for strategy
        strategy_multipliers = {
            PlanningStrategy.SEQUENTIAL: 1.0,
            PlanningStrategy.PARALLEL: 0.6,
            PlanningStrategy.CONDITIONAL: 1.2,
            PlanningStrategy.ITERATIVE: 1.8,
            PlanningStrategy.COLLABORATIVE: 1.3,
            PlanningStrategy.ADAPTIVE: 1.5
        }
        
        duration *= strategy_multipliers.get(strategy, 1.0)
        
        return int(duration)
    
    async def _determine_resource_requirements(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight]
    ) -> Dict[str, Any]:
        """Determine resource requirements"""
        
        # Base requirements
        requirements = {
            "cpu": 2,
            "memory": "4GB",
            "storage": "10GB",
            "agents": 1
        }
        
        # Adjust based on complexity
        complexity_multipliers = {
            PlanningComplexity.SIMPLE: 1.0,
            PlanningComplexity.MODERATE: 1.5,
            PlanningComplexity.COMPLEX: 2.0,
            PlanningComplexity.EXPERT: 2.5,
            PlanningComplexity.CRITICAL: 3.0
        }
        
        multiplier = complexity_multipliers.get(context.complexity, 1.0)
        requirements["cpu"] = int(requirements["cpu"] * multiplier)
        requirements["memory"] = f"{int(4 * multiplier)}GB"
        
        # Check for resource optimization insights
        for insight in insights:
            if insight.insight_type == "resource_optimization":
                optimal_resources = insight.data.get("optimal_resources", [])
                if optimal_resources:
                    # Use most efficient resource configuration
                    best_resource = optimal_resources[0]
                    requirements.update({
                        "cpu": best_resource.get("cpu", requirements["cpu"]),
                        "memory": best_resource.get("memory", requirements["memory"])
                    })
        
        return requirements
    
    async def _assess_risk_level(self, context: PlanningContext, strategy: PlanningStrategy) -> RiskLevel:
        """Assess risk level for the plan"""
        
        risk_score = 0
        
        # Risk from complexity
        complexity_risks = {
            PlanningComplexity.SIMPLE: 0.1,
            PlanningComplexity.MODERATE: 0.3,
            PlanningComplexity.COMPLEX: 0.5,
            PlanningComplexity.EXPERT: 0.7,
            PlanningComplexity.CRITICAL: 0.9
        }
        
        risk_score += complexity_risks.get(context.complexity, 0.3)
        
        # Risk from strategy
        strategy_risks = {
            PlanningStrategy.SEQUENTIAL: 0.1,
            PlanningStrategy.PARALLEL: 0.3,
            PlanningStrategy.CONDITIONAL: 0.4,
            PlanningStrategy.ITERATIVE: 0.5,
            PlanningStrategy.COLLABORATIVE: 0.6,
            PlanningStrategy.ADAPTIVE: 0.7
        }
        
        risk_score += strategy_risks.get(strategy, 0.3)
        
        # Risk from dependencies
        risk_score += min(len(context.dependencies) * 0.1, 0.4)
        
        # Risk from historical factors
        risk_score += min(len(context.risk_factors) * 0.05, 0.2)
        
        # Convert to risk level
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    async def _calculate_success_probability(
        self,
        context: PlanningContext,
        insights: List[PlanningInsight],
        strategy: PlanningStrategy
    ) -> float:
        """Calculate success probability"""
        
        base_probability = 0.7
        
        # Adjust based on historical patterns
        if context.historical_patterns:
            avg_success = sum(p.get("success_rate", 0.5) for p in context.historical_patterns) / len(context.historical_patterns)
            base_probability = (base_probability + avg_success) / 2
        
        # Adjust based on insights
        insight_boost = sum(insight.confidence * insight.impact_score for insight in insights) / max(len(insights), 1)
        base_probability += insight_boost * 0.2
        
        # Adjust based on complexity
        complexity_adjustments = {
            PlanningComplexity.SIMPLE: 0.1,
            PlanningComplexity.MODERATE: 0.05,
            PlanningComplexity.COMPLEX: -0.1,
            PlanningComplexity.EXPERT: -0.2,
            PlanningComplexity.CRITICAL: -0.3
        }
        
        base_probability += complexity_adjustments.get(context.complexity, 0)
        
        return max(0.1, min(0.95, base_probability))
    
    async def _get_alternative_strategies(self, context: PlanningContext, current_strategy: PlanningStrategy) -> List[str]:
        """Get alternative strategies"""
        
        all_strategies = [s.value for s in PlanningStrategy if s != current_strategy]
        
        # Filter based on complexity
        if context.complexity == PlanningComplexity.SIMPLE:
            return ["sequential", "parallel"]
        elif context.complexity == PlanningComplexity.MODERATE:
            return ["parallel", "conditional"]
        else:
            return ["collaborative", "adaptive"]
    
    # ===== RISK ASSESSMENT =====
    
    async def _assess_risks(
        self,
        context: PlanningContext,
        workflow: Workflow,
        insights: List[PlanningInsight]
    ) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        
        risks = {
            "overall_risk_level": await self._assess_risk_level(context, PlanningStrategy.SEQUENTIAL),
            "specific_risks": [],
            "mitigation_strategies": [],
            "contingency_plans": []
        }
        
        # Identify specific risks
        for risk_factor in context.risk_factors:
            risk_info = await self._analyze_risk_factor(risk_factor, context)
            risks["specific_risks"].append(risk_info)
        
        # Generate mitigation strategies
        for risk in risks["specific_risks"]:
            mitigation = await self._generate_mitigation_strategy(risk, context)
            risks["mitigation_strategies"].append(mitigation)
        
        # Create contingency plans
        contingency_plans = await self._create_contingency_plans(context, workflow)
        risks["contingency_plans"] = contingency_plans
        
        return risks
    
    async def _analyze_risk_factor(self, risk_factor: str, context: PlanningContext) -> Dict[str, Any]:
        """Analyze a specific risk factor"""
        
        return {
            "risk_factor": risk_factor,
            "probability": 0.3,  # Would be calculated based on historical data
            "impact": 0.5,
            "severity": "medium",
            "description": f"Risk factor: {risk_factor}"
        }
    
    async def _generate_mitigation_strategy(self, risk: Dict[str, Any], context: PlanningContext) -> Dict[str, Any]:
        """Generate mitigation strategy for a risk"""
        
        return {
            "risk_factor": risk["risk_factor"],
            "strategy": "Monitor and alert",
            "actions": ["Set up monitoring", "Define alert thresholds", "Prepare response plan"],
            "responsible_agent": "monitoring_agent"
        }
    
    async def _create_contingency_plans(self, context: PlanningContext, workflow: Workflow) -> List[Dict[str, Any]]:
        """Create contingency plans"""
        
        plans = [
            {
                "trigger": "execution_failure",
                "plan": "Switch to fallback strategy",
                "actions": ["Pause execution", "Analyze failure", "Switch strategy", "Resume execution"]
            },
            {
                "trigger": "resource_exhaustion",
                "plan": "Scale resources or simplify approach",
                "actions": ["Request additional resources", "Simplify workflow", "Prioritize critical tasks"]
            }
        ]
        
        return plans
    
    # ===== SUCCESS METRICS =====
    
    async def _define_success_metrics(self, context: PlanningContext, workflow: Workflow) -> Dict[str, Any]:
        """Define success metrics for the plan"""
        
        metrics = {
            "completion_metrics": {
                "target_completion_time": await self._estimate_duration(context, PlanningStrategy.SEQUENTIAL),
                "quality_threshold": 0.9,
                "success_rate_target": 0.95
            },
            "efficiency_metrics": {
                "resource_utilization_target": 0.8,
                "cost_efficiency_target": 0.85,
                "time_efficiency_target": 0.9
            },
            "quality_metrics": {
                "accuracy_target": 0.95,
                "reliability_target": 0.9,
                "user_satisfaction_target": 0.85
            }
        }
        
        return metrics
    
    # ===== ADAPTIVE TRIGGERS =====
    
    async def _create_adaptive_triggers(self, context: PlanningContext, workflow: Workflow) -> List[Dict[str, Any]]:
        """Create adaptive triggers for dynamic plan adjustment"""
        
        triggers = [
            {
                "trigger_type": "performance_degradation",
                "condition": "success_rate < 0.7",
                "action": "switch_strategy",
                "parameters": {"fallback_strategy": "sequential"}
            },
            {
                "trigger_type": "resource_constraint",
                "condition": "resource_utilization > 0.9",
                "action": "optimize_resources",
                "parameters": {"optimization_level": "aggressive"}
            },
            {
                "trigger_type": "time_overrun",
                "condition": "execution_time > estimated_time * 1.5",
                "action": "escalate_priority",
                "parameters": {"escalation_level": "high"}
            }
        ]
        
        return triggers
    
    # ===== STORAGE =====
    
    async def _store_plan_in_kg(self, plan: EnhancedPlan):
        """Store the enhanced plan in knowledge graph"""
        
        # Create plan vertex
        plan_query = f"""
        g.addV('EnhancedPlan')
         .property('plan_id', '{plan.plan_id}')
         .property('strategy', '{plan.strategy.value}')
         .property('complexity', '{plan.context.complexity.value}')
         .property('domain', '{plan.context.domain}')
         .property('success_probability', {plan.recommendations[0].success_probability if plan.recommendations else 0.0})
         .property('created_at', '{plan.created_at.isoformat()}')
         .property('original_prompt', '{plan.context.original_prompt.replace("'", "\\'")[:500]}')
        """
        
        try:
            await self.kg_client.execute_gremlin_query(plan_query)
            
            # Store insights as connected vertices
            for insight in plan.insights:
                insight_query = f"""
                g.addV('PlanningInsight')
                 .property('insight_type', '{insight.insight_type}')
                 .property('confidence', {insight.confidence})
                 .property('impact_score', {insight.impact_score})
                 .property('reasoning', '{insight.reasoning.replace("'", "\\'")[:200]}')
                 .as_('insight')
                 .V().has('EnhancedPlan', 'plan_id', '{plan.plan_id}')
                 .addE('HAS_INSIGHT').to('insight')
                """
                
                await self.kg_client.execute_gremlin_query(insight_query)
                
        except Exception as e:
            logger.error(f"Error storing plan in knowledge graph: {e}")
    
    # ===== BACKGROUND TASKS =====
    
    async def _build_pattern_cache(self):
        """Build pattern cache from historical data"""
        
        logger.info("Building pattern cache...")
        
        # Cache common patterns
        domains = ["devops", "data", "security", "ml", "general"]
        
        for domain in domains:
            query = f"""
            g.V().hasLabel('Workflow')
             .has('domain', '{domain}')
             .has('status', 'completed')
             .project('pattern', 'success_rate', 'avg_duration')
             .by(values('description').map({{it.get().substring(0, Math.min(50, it.get().length()))}})
             .by('success_rate')
             .by('avg_duration')
             .groupBy('pattern')
             .cap()
            """
            
            try:
                result = await self.kg_client.execute_gremlin_query(query)
                patterns = result.get("result", {})
                
                for pattern, data in patterns.items():
                    cache_key = f"{domain}_{pattern}"
                    self.pattern_cache[cache_key] = data
                    
            except Exception as e:
                logger.warning(f"Error building pattern cache for domain {domain}: {e}")
        
        logger.info(f"Pattern cache built with {len(self.pattern_cache)} entries")
    
    async def _subscribe_to_planning_events(self):
        """Subscribe to planning events for learning"""
        
        try:
            # Subscribe to workflow completion events
            await self.pulsar_service.subscribe(
                "q.workflow.completed",
                self._handle_workflow_completion,
                subscription_name="kg_planner_learning"
            )
            
            # Subscribe to planning feedback events
            await self.pulsar_service.subscribe(
                "q.planning.feedback",
                self._handle_planning_feedback,
                subscription_name="kg_planner_feedback"
            )
            
        except Exception as e:
            logger.error(f"Error subscribing to planning events: {e}")
    
    async def _handle_workflow_completion(self, event_data: Dict[str, Any]):
        """Handle workflow completion events for learning"""
        
        try:
            workflow_id = event_data.get("workflow_id")
            success = event_data.get("success", False)
            
            if success:
                self.planning_stats["successful_plans"] += 1
            else:
                self.planning_stats["failed_plans"] += 1
                
            # Update pattern cache with new data
            await self._update_pattern_cache(event_data)
            
        except Exception as e:
            logger.error(f"Error handling workflow completion: {e}")
    
    async def _handle_planning_feedback(self, feedback_data: Dict[str, Any]):
        """Handle planning feedback for improvement"""
        
        try:
            plan_id = feedback_data.get("plan_id")
            rating = feedback_data.get("rating", 0)
            
            # Update planning strategies based on feedback
            if rating > 4:  # Good feedback
                strategy = feedback_data.get("strategy")
                if strategy:
                    # Increase confidence in this strategy
                    pass
            elif rating < 3:  # Poor feedback
                # Analyze what went wrong and adjust
                pass
                
        except Exception as e:
            logger.error(f"Error handling planning feedback: {e}")
    
    async def _update_pattern_cache(self, event_data: Dict[str, Any]):
        """Update pattern cache with new workflow data"""
        
        try:
            domain = event_data.get("domain", "general")
            pattern = event_data.get("description", "")[:50]
            
            cache_key = f"{domain}_{pattern}"
            
            # Update cache with new data
            if cache_key in self.pattern_cache:
                # Update existing pattern
                pass
            else:
                # Add new pattern
                self.pattern_cache[cache_key] = [event_data]
                
        except Exception as e:
            logger.error(f"Error updating pattern cache: {e}")
    
    # ===== MONITORING =====
    
    async def get_planning_statistics(self) -> Dict[str, Any]:
        """Get planning statistics"""
        
        return {
            "statistics": self.planning_stats,
            "cache_size": len(self.pattern_cache),
            "active_plans": len(self.strategy_cache),
            "insights_cached": len(self.insight_cache)
        }
    
    async def get_plan_insights(self, plan_id: str) -> List[PlanningInsight]:
        """Get insights for a specific plan"""
        
        return self.insight_cache.get(plan_id, [])
    
    async def update_plan_strategy(self, plan_id: str, new_strategy: PlanningStrategy) -> bool:
        """Update strategy for an existing plan"""
        
        try:
            # Update in knowledge graph
            update_query = f"""
            g.V().has('EnhancedPlan', 'plan_id', '{plan_id}')
             .property('strategy', '{new_strategy.value}')
             .property('updated_at', '{datetime.utcnow().isoformat()}')
            """
            
            await self.kg_client.execute_gremlin_query(update_query)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating plan strategy: {e}")
            return False

# Create global instance
knowledge_graph_planner = KnowledgeGraphPlanner(
    kg_client=KnowledgeGraphClient(base_url="http://localhost:8003"),
    memory_service=MemoryService(),
    pulsar_service=PulsarService()
) 