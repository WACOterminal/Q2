# shared/q_memory_schemas/memory_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from enum import Enum
import uuid


class MemoryType(str, Enum):
    """Types of agent memories"""
    EPISODIC = "episodic"      # Specific events/interactions
    SEMANTIC = "semantic"       # Facts and knowledge
    PROCEDURAL = "procedural"   # How-to knowledge
    REFLECTION = "reflection"   # Meta-cognition about past actions
    

class MemoryImportance(BaseModel):
    """Calculates and stores memory importance metrics"""
    recency_score: float = Field(..., description="How recent the memory is (0-1)")
    frequency_score: float = Field(..., description="How often referenced (0-1)")
    relevance_score: float = Field(..., description="Relevance to current context (0-1)")
    emotional_weight: float = Field(0.5, description="Emotional significance (0-1)")
    
    @property
    def total_score(self) -> float:
        """Calculate weighted importance score"""
        return (
            self.recency_score * 0.3 +
            self.frequency_score * 0.2 +
            self.relevance_score * 0.4 +
            self.emotional_weight * 0.1
        )


class AgentMemory(BaseModel):
    """Core memory structure for agents"""
    memory_id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4()}")
    agent_id: str = Field(..., description="The agent that created this memory")
    conversation_id: Optional[str] = Field(None, description="Associated conversation")
    workflow_id: Optional[str] = Field(None, description="Associated workflow")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Memory content
    type: MemoryType
    content: str = Field(..., description="The actual memory content")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    
    # Extracted information
    entities: List[str] = Field(default_factory=list, description="Named entities in the memory")
    relationships: Dict[str, List[str]] = Field(
        default_factory=dict, 
        description="Relationships between entities"
    )
    keywords: List[str] = Field(default_factory=list, description="Key terms for retrieval")
    
    # Memory metadata
    importance: MemoryImportance
    access_count: int = Field(0, description="Number of times accessed")
    last_accessed: Optional[str] = None
    embedding_vector: Optional[List[float]] = Field(None, description="Vector representation")
    
    # For procedural memories
    success_rate: Optional[float] = Field(None, description="Success rate for procedural memories")
    
    # For reflections
    source_memories: List[str] = Field(default_factory=list, description="Memories that led to this reflection")
    

class MemoryQuery(BaseModel):
    """Query structure for retrieving memories"""
    agent_id: str
    query_text: Optional[str] = None
    memory_types: List[MemoryType] = Field(default_factory=list)
    time_range: Optional[Dict[str, str]] = None  # {"start": iso_date, "end": iso_date}
    entities: List[str] = Field(default_factory=list)
    min_importance: float = 0.0
    max_results: int = 10
    include_consolidated: bool = True
    

class MemoryConsolidation(BaseModel):
    """Represents consolidated/summarized memories"""
    consolidation_id: str = Field(default_factory=lambda: f"consol_{uuid.uuid4()}")
    agent_id: str
    source_memory_ids: List[str]
    consolidation_type: str  # "daily", "weekly", "topic-based"
    
    summary: str = Field(..., description="Consolidated summary of memories")
    key_insights: List[str] = Field(default_factory=list)
    patterns_identified: List[Dict[str, Any]] = Field(default_factory=list)
    
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    importance_score: float
    
    # Preserved important details
    preserved_entities: List[str] = Field(default_factory=list)
    preserved_relationships: Dict[str, List[str]] = Field(default_factory=dict)


class MemoryFeedback(BaseModel):
    """Feedback on memory usefulness"""
    memory_id: str
    was_helpful: bool
    context: str  # What task/query it was used for
    feedback_text: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat()) 