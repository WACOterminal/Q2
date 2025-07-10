"""
AI Explanation Service

This service provides transparency and justification for AI decisions through:
- Multi-level explanations (from simple to technical)
- Different explanation types (decision rationale, process steps, etc.)
- Confidence and uncertainty analysis
- Visual explanations and diagrams
- Interactive Q&A about decisions
- Audit trails for decision making
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
from enum import Enum
import json
import uuid

# Q Platform imports
from shared.q_collaboration_schemas.models import (
    AIExplanation, ExplanationType, CollaborationSession
)
from shared.q_memory_schemas.memory_models import AgentMemory, MemoryType
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.memory_service import MemoryService
from app.services.pulsar_service import PulsarService
from app.services.ignite_service import IgniteService

logger = logging.getLogger(__name__)

class ExplanationLevel(Enum):
    """Levels of explanation detail"""
    SIMPLE = "simple"          # For end users
    DETAILED = "detailed"      # For business users
    TECHNICAL = "technical"    # For technical users
    EXPERT = "expert"          # For AI experts

class ExplanationFormat(Enum):
    """Format of explanations"""
    TEXT = "text"
    STRUCTURED = "structured"
    VISUAL = "visual"
    INTERACTIVE = "interactive"
    DIAGRAM = "diagram"

class ConfidenceLevel(Enum):
    """Confidence levels for decisions"""
    VERY_LOW = "very_low"      # < 0.3
    LOW = "low"                # 0.3-0.5
    MEDIUM = "medium"          # 0.5-0.7
    HIGH = "high"              # 0.7-0.9
    VERY_HIGH = "very_high"    # > 0.9

class AIExplanationService:
    """
    Service for generating AI explanations and transparency reports
    """
    
    def __init__(self):
        self.knowledge_graph = KnowledgeGraphService()
        self.memory_service = MemoryService()
        self.pulsar_service = PulsarService()
        self.ignite_service = IgniteService()
        
        # Active explanations
        self.active_explanations: Dict[str, AIExplanation] = {}
        
        # Explanation templates
        self.explanation_templates = {}
        
        # Model metadata for explanations
        self.model_registry = {}
        
    async def initialize(self):
        """Initialize the AI explanation service"""
        logger.info("Initializing AI Explanation Service")
        
        # Load explanation templates
        await self._load_explanation_templates()
        
        # Load model registry
        await self._load_model_registry()
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        logger.info("AI Explanation Service initialized successfully")
    
    # ===== EXPLANATION GENERATION =====
    
    async def generate_explanation(
        self,
        agent_id: str,
        requester_id: str,
        explanation_type: ExplanationType,
        decision_context: Dict[str, Any],
        explanation_level: ExplanationLevel = ExplanationLevel.DETAILED,
        explanation_format: ExplanationFormat = ExplanationFormat.TEXT,
        specific_questions: List[str] = None
    ) -> AIExplanation:
        """
        Generate an AI explanation for a decision
        
        Args:
            agent_id: ID of the agent that made the decision
            requester_id: ID of the user requesting explanation
            explanation_type: Type of explanation requested
            decision_context: Context of the decision
            explanation_level: Level of detail for explanation
            explanation_format: Format of the explanation
            specific_questions: Specific questions about the decision
            
        Returns:
            Generated AI explanation
        """
        logger.info(f"Generating {explanation_type.value} explanation for agent {agent_id}")
        
        explanation_id = f"explain_{uuid.uuid4().hex[:12]}"
        session_id = decision_context.get("session_id", f"session_{explanation_id}")
        
        # Extract decision details
        decision_data = await self._extract_decision_data(decision_context)
        
        # Generate explanation content based on type
        explanation_content = await self._generate_explanation_content(
            explanation_type,
            decision_data,
            explanation_level,
            explanation_format
        )
        
        # Calculate confidence scores
        confidence_scores = await self._calculate_confidence_scores(decision_data)
        
        # Identify uncertainty factors
        uncertainty_factors = await self._identify_uncertainty_factors(decision_data)
        
        # Generate alternative options
        alternative_options = await self._generate_alternative_options(decision_data)
        
        # Assess risk factors
        risk_factors = await self._assess_risk_factors(decision_data)
        
        # Get supporting data sources
        data_sources = await self._get_data_sources(decision_data)
        
        # Get model details
        model_details = await self._get_model_details(agent_id, decision_context)
        
        # Create explanation
        explanation = AIExplanation(
            explanation_id=explanation_id,
            session_id=session_id,
            agent_id=agent_id,
            requester_id=requester_id,
            explanation_type=explanation_type,
            decision_context=decision_context,
            specific_questions=specific_questions or [],
            rationale=explanation_content["rationale"],
            process_steps=explanation_content["process_steps"],
            confidence_scores=confidence_scores,
            uncertainty_factors=uncertainty_factors,
            data_sources=data_sources,
            model_details=model_details,
            alternative_options=alternative_options,
            risk_factors=risk_factors,
            human_feedback=None,
            accuracy_verified=None,
            explanation_quality=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store explanation
        self.active_explanations[explanation_id] = explanation
        await self._persist_explanation(explanation)
        
        # Create memory entry for this explanation
        await self._create_explanation_memory(explanation)
        
        # Publish explanation event
        await self.pulsar_service.publish(
            "q.explanation.generated",
            {
                "explanation_id": explanation_id,
                "agent_id": agent_id,
                "requester_id": requester_id,
                "explanation_type": explanation_type.value,
                "confidence_level": self._get_confidence_level(confidence_scores.get("overall", 0)),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"AI explanation generated: {explanation_id}")
        return explanation
    
    # ===== EXPLANATION CONTENT GENERATION =====
    
    async def _generate_explanation_content(
        self,
        explanation_type: ExplanationType,
        decision_data: Dict[str, Any],
        explanation_level: ExplanationLevel,
        explanation_format: ExplanationFormat
    ) -> Dict[str, Any]:
        """
        Generate explanation content based on type and level
        
        Args:
            explanation_type: Type of explanation
            decision_data: Decision data
            explanation_level: Level of detail
            explanation_format: Format of explanation
            
        Returns:
            Generated explanation content
        """
        logger.debug(f"Generating {explanation_type.value} content at {explanation_level.value} level")
        
        if explanation_type == ExplanationType.DECISION_RATIONALE:
            return await self._generate_decision_rationale(decision_data, explanation_level)
        elif explanation_type == ExplanationType.PROCESS_STEPS:
            return await self._generate_process_steps(decision_data, explanation_level)
        elif explanation_type == ExplanationType.CONFIDENCE_FACTORS:
            return await self._generate_confidence_factors(decision_data, explanation_level)
        elif explanation_type == ExplanationType.RISK_ASSESSMENT:
            return await self._generate_risk_assessment(decision_data, explanation_level)
        elif explanation_type == ExplanationType.ALTERNATIVE_OPTIONS:
            return await self._generate_alternative_analysis(decision_data, explanation_level)
        elif explanation_type == ExplanationType.LEARNING_SOURCES:
            return await self._generate_learning_sources(decision_data, explanation_level)
        else:
            return await self._generate_general_explanation(decision_data, explanation_level)
    
    async def _generate_decision_rationale(
        self,
        decision_data: Dict[str, Any],
        explanation_level: ExplanationLevel
    ) -> Dict[str, Any]:
        """Generate decision rationale explanation"""
        decision = decision_data.get("decision", "")
        factors = decision_data.get("decision_factors", [])
        
        if explanation_level == ExplanationLevel.SIMPLE:
            rationale = f"I chose '{decision}' because it best addresses the main requirements."
            if factors:
                rationale += f" The key factors were: {', '.join(factors[:3])}."
        
        elif explanation_level == ExplanationLevel.DETAILED:
            rationale = f"I selected '{decision}' after analyzing multiple factors:\n"
            for i, factor in enumerate(factors[:5], 1):
                rationale += f"{i}. {factor}\n"
            rationale += f"This decision optimizes for the stated objectives while minimizing risks."
        
        elif explanation_level == ExplanationLevel.TECHNICAL:
            rationale = f"Decision: '{decision}'\n"
            rationale += f"Analysis method: {decision_data.get('method', 'Multi-criteria analysis')}\n"
            rationale += f"Factors considered ({len(factors)} total):\n"
            for factor in factors:
                weight = decision_data.get("factor_weights", {}).get(factor, 1.0)
                rationale += f"  - {factor} (weight: {weight})\n"
            rationale += f"Confidence score: {decision_data.get('confidence', 0.0):.2f}"
        
        else:  # EXPERT
            rationale = f"Decision: '{decision}'\n"
            rationale += f"Algorithm: {decision_data.get('algorithm', 'Unknown')}\n"
            rationale += f"Model version: {decision_data.get('model_version', 'Unknown')}\n"
            rationale += f"Feature importance: {decision_data.get('feature_importance', {})}\n"
            rationale += f"Decision boundary: {decision_data.get('decision_boundary', 'Unknown')}\n"
            rationale += f"Prediction probabilities: {decision_data.get('probabilities', {})}"
        
        return {
            "rationale": rationale,
            "process_steps": self._extract_process_steps(decision_data)
        }
    
    async def _generate_process_steps(
        self,
        decision_data: Dict[str, Any],
        explanation_level: ExplanationLevel
    ) -> Dict[str, Any]:
        """Generate process steps explanation"""
        steps = decision_data.get("process_steps", [])
        
        process_steps = []
        
        for i, step in enumerate(steps, 1):
            if explanation_level == ExplanationLevel.SIMPLE:
                process_steps.append({
                    "step": i,
                    "description": step.get("description", ""),
                    "result": step.get("result", "")
                })
            else:
                process_steps.append({
                    "step": i,
                    "description": step.get("description", ""),
                    "inputs": step.get("inputs", []),
                    "outputs": step.get("outputs", []),
                    "method": step.get("method", ""),
                    "confidence": step.get("confidence", 0.0),
                    "duration": step.get("duration", 0),
                    "result": step.get("result", "")
                })
        
        rationale = f"The decision process involved {len(process_steps)} main steps:"
        if explanation_level == ExplanationLevel.SIMPLE:
            rationale += " " + " → ".join([f"Step {s['step']}: {s['description']}" for s in process_steps[:3]])
        else:
            rationale += "\n" + "\n".join([f"{s['step']}. {s['description']}" for s in process_steps])
        
        return {
            "rationale": rationale,
            "process_steps": process_steps
        }
    
    async def _generate_confidence_factors(
        self,
        decision_data: Dict[str, Any],
        explanation_level: ExplanationLevel
    ) -> Dict[str, Any]:
        """Generate confidence factors explanation"""
        confidence_factors = decision_data.get("confidence_factors", {})
        overall_confidence = decision_data.get("confidence", 0.0)
        
        confidence_level = self._get_confidence_level(overall_confidence)
        
        if explanation_level == ExplanationLevel.SIMPLE:
            rationale = f"I am {confidence_level.value.replace('_', ' ')} confident in this decision ({overall_confidence:.1%})."
            if confidence_factors:
                top_factors = sorted(confidence_factors.items(), key=lambda x: x[1], reverse=True)[:2]
                rationale += f" Main confidence drivers: {', '.join([f[0] for f in top_factors])}."
        
        else:
            rationale = f"Overall confidence: {overall_confidence:.1%} ({confidence_level.value.replace('_', ' ')})\n"
            rationale += "Confidence breakdown:\n"
            for factor, score in sorted(confidence_factors.items(), key=lambda x: x[1], reverse=True):
                rationale += f"  - {factor}: {score:.1%}\n"
        
        return {
            "rationale": rationale,
            "process_steps": [
                {
                    "step": 1,
                    "description": "Analyze decision factors",
                    "result": f"Identified {len(confidence_factors)} confidence factors"
                },
                {
                    "step": 2,
                    "description": "Calculate confidence scores",
                    "result": f"Overall confidence: {overall_confidence:.1%}"
                }
            ]
        }
    
    async def _generate_risk_assessment(
        self,
        decision_data: Dict[str, Any],
        explanation_level: ExplanationLevel
    ) -> Dict[str, Any]:
        """Generate risk assessment explanation"""
        risks = decision_data.get("risk_factors", [])
        risk_score = decision_data.get("risk_score", 0.0)
        
        risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
        
        if explanation_level == ExplanationLevel.SIMPLE:
            rationale = f"Risk level: {risk_level} ({risk_score:.1%})"
            if risks:
                rationale += f". Main risks: {', '.join([r.get('name', '') for r in risks[:2]])}."
        
        else:
            rationale = f"Risk Assessment Summary:\n"
            rationale += f"Overall risk score: {risk_score:.1%} ({risk_level})\n"
            rationale += f"Risk factors identified: {len(risks)}\n\n"
            
            for i, risk in enumerate(risks, 1):
                rationale += f"{i}. {risk.get('name', 'Unknown risk')}\n"
                rationale += f"   Probability: {risk.get('probability', 0.0):.1%}\n"
                rationale += f"   Impact: {risk.get('impact', 0.0):.1%}\n"
                rationale += f"   Mitigation: {risk.get('mitigation', 'None specified')}\n\n"
        
        return {
            "rationale": rationale,
            "process_steps": [
                {
                    "step": 1,
                    "description": "Identify potential risks",
                    "result": f"Found {len(risks)} risk factors"
                },
                {
                    "step": 2,
                    "description": "Assess risk probability and impact",
                    "result": f"Calculated risk score: {risk_score:.1%}"
                },
                {
                    "step": 3,
                    "description": "Determine risk level",
                    "result": f"Risk level: {risk_level}"
                }
            ]
        }
    
    async def _generate_alternative_analysis(
        self,
        decision_data: Dict[str, Any],
        explanation_level: ExplanationLevel
    ) -> Dict[str, Any]:
        """Generate alternative options analysis"""
        alternatives = decision_data.get("alternatives", [])
        chosen_option = decision_data.get("decision", "")
        
        if explanation_level == ExplanationLevel.SIMPLE:
            rationale = f"I considered {len(alternatives)} options and chose '{chosen_option}'."
            if alternatives:
                rationale += f" Other options included: {', '.join([a.get('name', '') for a in alternatives[:2]])}."
        
        else:
            rationale = f"Alternative Analysis:\n"
            rationale += f"Options considered: {len(alternatives)}\n"
            rationale += f"Selected option: '{chosen_option}'\n\n"
            
            for i, alt in enumerate(alternatives, 1):
                rationale += f"{i}. {alt.get('name', 'Unknown option')}\n"
                rationale += f"   Score: {alt.get('score', 0.0):.2f}\n"
                rationale += f"   Pros: {', '.join(alt.get('pros', []))}\n"
                rationale += f"   Cons: {', '.join(alt.get('cons', []))}\n\n"
        
        return {
            "rationale": rationale,
            "process_steps": [
                {
                    "step": 1,
                    "description": "Generate alternative options",
                    "result": f"Created {len(alternatives)} alternatives"
                },
                {
                    "step": 2,
                    "description": "Evaluate each option",
                    "result": "Scored all options against criteria"
                },
                {
                    "step": 3,
                    "description": "Select best option",
                    "result": f"Chose '{chosen_option}' with highest score"
                }
            ]
        }
    
    async def _generate_learning_sources(
        self,
        decision_data: Dict[str, Any],
        explanation_level: ExplanationLevel
    ) -> Dict[str, Any]:
        """Generate learning sources explanation"""
        training_data = decision_data.get("training_data", [])
        knowledge_sources = decision_data.get("knowledge_sources", [])
        
        if explanation_level == ExplanationLevel.SIMPLE:
            rationale = f"My decision is based on learning from {len(training_data)} examples"
            if knowledge_sources:
                rationale += f" and knowledge from {len(knowledge_sources)} sources."
        
        else:
            rationale = f"Learning Sources:\n"
            rationale += f"Training examples: {len(training_data)}\n"
            rationale += f"Knowledge sources: {len(knowledge_sources)}\n\n"
            
            if training_data:
                rationale += "Training data:\n"
                for source in training_data[:5]:
                    rationale += f"  - {source.get('description', 'Unknown source')}\n"
            
            if knowledge_sources:
                rationale += "\nKnowledge sources:\n"
                for source in knowledge_sources[:5]:
                    rationale += f"  - {source.get('name', 'Unknown source')}: {source.get('description', '')}\n"
        
        return {
            "rationale": rationale,
            "process_steps": [
                {
                    "step": 1,
                    "description": "Access training data",
                    "result": f"Retrieved {len(training_data)} training examples"
                },
                {
                    "step": 2,
                    "description": "Consult knowledge sources",
                    "result": f"Referenced {len(knowledge_sources)} knowledge sources"
                },
                {
                    "step": 3,
                    "description": "Apply learned patterns",
                    "result": "Applied patterns to current decision"
                }
            ]
        }
    
    async def _generate_general_explanation(
        self,
        decision_data: Dict[str, Any],
        explanation_level: ExplanationLevel
    ) -> Dict[str, Any]:
        """Generate general explanation"""
        decision = decision_data.get("decision", "")
        method = decision_data.get("method", "analysis")
        
        rationale = f"I made the decision '{decision}' using {method}."
        
        return {
            "rationale": rationale,
            "process_steps": [
                {
                    "step": 1,
                    "description": "Analyze the situation",
                    "result": "Identified key factors and constraints"
                },
                {
                    "step": 2,
                    "description": f"Apply {method}",
                    "result": f"Determined optimal decision: '{decision}'"
                }
            ]
        }
    
    # ===== FEEDBACK AND VALIDATION =====
    
    async def collect_explanation_feedback(
        self,
        explanation_id: str,
        user_id: str,
        feedback: Dict[str, Any]
    ) -> bool:
        """
        Collect feedback on explanation quality
        
        Args:
            explanation_id: Explanation ID
            user_id: User providing feedback
            feedback: Feedback data
            
        Returns:
            True if feedback collected successfully
        """
        logger.info(f"Collecting feedback for explanation: {explanation_id}")
        
        explanation = self.active_explanations.get(explanation_id)
        if not explanation:
            explanation = await self._load_explanation(explanation_id)
            if not explanation:
                return False
        
        # Store feedback
        explanation.human_feedback = feedback
        explanation.explanation_quality = feedback.get("quality_rating", 0.0)
        explanation.accuracy_verified = feedback.get("accuracy_verified", False)
        explanation.updated_at = datetime.utcnow()
        
        # Persist updated explanation
        await self._persist_explanation(explanation)
        
        # Learn from feedback
        await self._learn_from_feedback(explanation, feedback)
        
        return True
    
    async def _learn_from_feedback(
        self,
        explanation: AIExplanation,
        feedback: Dict[str, Any]
    ):
        """Learn from explanation feedback to improve future explanations"""
        # Create learning memory
        learning_memory = AgentMemory(
            memory_id=f"explain_feedback_{explanation.explanation_id}",
            agent_id=explanation.agent_id,
            memory_type=MemoryType.FEEDBACK,
            content=f"Explanation feedback: {feedback.get('comments', '')}",
            context={
                "explanation_type": explanation.explanation_type.value,
                "quality_rating": feedback.get("quality_rating", 0.0),
                "accuracy_verified": feedback.get("accuracy_verified", False),
                "improvement_suggestions": feedback.get("suggestions", [])
            },
            importance=0.7,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1
        )
        
        await self.memory_service.store_memory(learning_memory)
    
    # ===== HELPER METHODS =====
    
    async def _extract_decision_data(
        self,
        decision_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract decision data from context"""
        return {
            "decision": decision_context.get("decision", ""),
            "decision_factors": decision_context.get("factors", []),
            "confidence": decision_context.get("confidence", 0.0),
            "method": decision_context.get("method", "analysis"),
            "alternatives": decision_context.get("alternatives", []),
            "risk_factors": decision_context.get("risks", []),
            "risk_score": decision_context.get("risk_score", 0.0),
            "process_steps": decision_context.get("steps", []),
            "training_data": decision_context.get("training_data", []),
            "knowledge_sources": decision_context.get("knowledge_sources", [])
        }
    
    async def _calculate_confidence_scores(
        self,
        decision_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate confidence scores for different aspects"""
        base_confidence = decision_data.get("confidence", 0.0)
        
        confidence_scores = {
            "overall": base_confidence,
            "data_quality": min(1.0, base_confidence + 0.1),
            "model_certainty": base_confidence * 0.9,
            "domain_expertise": base_confidence * 0.8,
            "historical_accuracy": base_confidence * 0.85
        }
        
        return confidence_scores
    
    async def _identify_uncertainty_factors(
        self,
        decision_data: Dict[str, Any]
    ) -> List[str]:
        """Identify factors that contribute to uncertainty"""
        uncertainty_factors = []
        
        confidence = decision_data.get("confidence", 0.0)
        
        if confidence < 0.5:
            uncertainty_factors.append("Low overall confidence in decision")
        
        if len(decision_data.get("alternatives", [])) > 3:
            uncertainty_factors.append("Many alternative options available")
        
        if decision_data.get("risk_score", 0.0) > 0.7:
            uncertainty_factors.append("High risk factors identified")
        
        if len(decision_data.get("decision_factors", [])) < 3:
            uncertainty_factors.append("Limited decision factors considered")
        
        return uncertainty_factors
    
    async def _generate_alternative_options(
        self,
        decision_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate alternative options that were considered"""
        alternatives = decision_data.get("alternatives", [])
        
        # Format alternatives for explanation
        formatted_alternatives = []
        for alt in alternatives:
            formatted_alternatives.append({
                "name": alt.get("name", ""),
                "description": alt.get("description", ""),
                "score": alt.get("score", 0.0),
                "pros": alt.get("pros", []),
                "cons": alt.get("cons", []),
                "feasibility": alt.get("feasibility", 0.0)
            })
        
        return formatted_alternatives
    
    async def _assess_risk_factors(
        self,
        decision_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Assess risk factors for the decision"""
        risk_factors = decision_data.get("risk_factors", [])
        
        # Format risk factors for explanation
        formatted_risks = []
        for risk in risk_factors:
            formatted_risks.append({
                "name": risk.get("name", ""),
                "description": risk.get("description", ""),
                "probability": risk.get("probability", 0.0),
                "impact": risk.get("impact", 0.0),
                "mitigation": risk.get("mitigation", ""),
                "severity": risk.get("severity", "medium")
            })
        
        return formatted_risks
    
    async def _get_data_sources(
        self,
        decision_data: Dict[str, Any]
    ) -> List[str]:
        """Get data sources used in the decision"""
        sources = []
        
        # Add training data sources
        training_data = decision_data.get("training_data", [])
        for data in training_data:
            sources.append(data.get("source", "Unknown source"))
        
        # Add knowledge sources
        knowledge_sources = decision_data.get("knowledge_sources", [])
        for source in knowledge_sources:
            sources.append(source.get("name", "Unknown source"))
        
        return list(set(sources))  # Remove duplicates
    
    async def _get_model_details(
        self,
        agent_id: str,
        decision_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get model details for the agent"""
        model_details = self.model_registry.get(agent_id, {})
        
        return {
            "model_type": model_details.get("type", "Unknown"),
            "model_version": model_details.get("version", "Unknown"),
            "training_date": model_details.get("training_date", "Unknown"),
            "parameters": model_details.get("parameters", {}),
            "performance_metrics": model_details.get("performance", {})
        }
    
    def _extract_process_steps(
        self,
        decision_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract process steps from decision data"""
        return decision_data.get("process_steps", [])
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level from confidence score"""
        if confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    # ===== PERSISTENCE =====
    
    async def _persist_explanation(self, explanation: AIExplanation):
        """Persist explanation to storage"""
        await self.ignite_service.put(
            f"ai_explanation:{explanation.explanation_id}",
            asdict(explanation)
        )
    
    async def _load_explanation(self, explanation_id: str) -> Optional[AIExplanation]:
        """Load explanation from storage"""
        explanation_data = await self.ignite_service.get(f"ai_explanation:{explanation_id}")
        if explanation_data:
            return AIExplanation(**explanation_data)
        return None
    
    async def _create_explanation_memory(self, explanation: AIExplanation):
        """Create memory entry for explanation"""
        memory = AgentMemory(
            memory_id=f"explanation_{explanation.explanation_id}",
            agent_id=explanation.agent_id,
            memory_type=MemoryType.EXPLANATION,
            content=f"Generated {explanation.explanation_type.value} explanation",
            context={
                "explanation_id": explanation.explanation_id,
                "requester_id": explanation.requester_id,
                "explanation_type": explanation.explanation_type.value,
                "confidence_scores": explanation.confidence_scores
            },
            importance=0.6,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1
        )
        
        await self.memory_service.store_memory(memory)
    
    # ===== INITIALIZATION =====
    
    async def _load_explanation_templates(self):
        """Load explanation templates"""
        # This would load templates from configuration
        self.explanation_templates = {
            "decision_rationale": {
                "simple": "I chose {decision} because {main_reason}.",
                "detailed": "Decision: {decision}\nRationale: {detailed_rationale}",
                "technical": "Decision: {decision}\nMethod: {method}\nFactors: {factors}"
            }
        }
    
    async def _load_model_registry(self):
        """Load model registry"""
        # This would load model metadata from registry
        self.model_registry = {}
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics"""
        topics = [
            "q.explanation.generated",
            "q.explanation.feedback",
            "q.explanation.quality"
        ]
        
        for topic in topics:
            await self.pulsar_service.ensure_topic(topic)

# Global service instance
ai_explanation_service = AIExplanationService() 