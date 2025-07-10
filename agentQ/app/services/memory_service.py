# agentQ/app/services/memory_service.py
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, timezone
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from shared.q_memory_schemas.memory_models import (
    AgentMemory, MemoryType, MemoryImportance, 
    MemoryQuery, MemoryConsolidation, MemoryFeedback
)
from shared.q_vectorstore_client.client import VectorStoreClient
from shared.q_vectorstore_client.models import Vector, Query
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from pyignite import Client
from pyignite.exceptions import CacheError

logger = logging.getLogger(__name__)


class MemoryService:
    """Manages agent memories with storage, retrieval, and consolidation"""
    
    MEMORY_COLLECTION = "agent_memories"
    MEMORY_CACHE = "agent_memory_cache"
    CONSOLIDATION_CACHE = "memory_consolidations"
    
    def __init__(
        self, 
        vector_client: VectorStoreClient,
        kg_client: KnowledgeGraphClient,
        ignite_addresses: List[str]
    ):
        self.vector_client = vector_client
        self.kg_client = kg_client
        self.ignite_client = Client()
        self.ignite_addresses = ignite_addresses
        
        # Load embedding model for memory vectorization
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Connect to Ignite
        self._connect_ignite()
        
    def _connect_ignite(self):
        """Connect to Ignite cluster"""
        try:
            self.ignite_client.connect(self.ignite_addresses)
            self.memory_cache = self.ignite_client.get_or_create_cache(self.MEMORY_CACHE)
            self.consolidation_cache = self.ignite_client.get_or_create_cache(self.CONSOLIDATION_CACHE)
            logger.info("Connected to Ignite for memory caching")
        except Exception as e:
            logger.error(f"Failed to connect to Ignite: {e}")
            raise
            
    async def save_memory(self, memory: AgentMemory) -> str:
        """Save a memory to all storage systems"""
        try:
            # Calculate importance if not provided
            if not memory.importance:
                memory.importance = self._calculate_importance(memory)
                
            # Generate embedding
            memory.embedding_vector = self.embedding_model.encode(memory.content).tolist()
            
            # Store in vector database
            vector = Vector(
                id=memory.memory_id,
                values=memory.embedding_vector,
                metadata={
                    "agent_id": memory.agent_id,
                    "type": memory.type.value,
                    "timestamp": memory.timestamp,
                    "entities": memory.entities,
                    "importance": memory.importance.total_score
                }
            )
            await self.vector_client.upsert(self.MEMORY_COLLECTION, [vector])
            
            # Store in knowledge graph
            await self._store_in_knowledge_graph(memory)
            
            # Cache in Ignite for fast access
            cache_key = f"{memory.agent_id}:{memory.memory_id}"
            self.memory_cache.put(cache_key, memory.dict())
            
            # Update agent's memory index
            await self._update_memory_index(memory.agent_id, memory.memory_id)
            
            logger.info(f"Saved memory {memory.memory_id} for agent {memory.agent_id}")
            return memory.memory_id
            
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            raise
            
    async def retrieve_memories(self, query: MemoryQuery) -> List[AgentMemory]:
        """Retrieve relevant memories based on query"""
        memories = []
        
        try:
            # Semantic search if query text provided
            if query.query_text:
                query_vector = self.embedding_model.encode(query.query_text).tolist()
                
                search_query = Query(
                    values=query_vector,
                    top_k=query.max_results * 2,  # Get more for filtering
                    filter={"agent_id": query.agent_id}
                )
                
                search_response = await self.vector_client.search(
                    self.MEMORY_COLLECTION, 
                    [search_query]
                )
                
                # Get full memories from cache
                for hit in search_response.results[0].hits:
                    if hit.score >= query.min_importance:
                        memory_dict = self.memory_cache.get(f"{query.agent_id}:{hit.id}")
                        if memory_dict:
                            memory = AgentMemory(**memory_dict)
                            
                            # Apply type filter
                            if not query.memory_types or memory.type in query.memory_types:
                                memories.append(memory)
                                
            # Also get recent memories if no query text
            else:
                memories.extend(await self._get_recent_memories(query))
                
            # Apply time range filter
            if query.time_range:
                memories = self._filter_by_time_range(memories, query.time_range)
                
            # Include consolidated memories if requested
            if query.include_consolidated:
                consolidated = await self._get_consolidated_memories(
                    query.agent_id, 
                    query.query_text
                )
                memories.extend(consolidated)
                
            # Update access counts
            for memory in memories:
                await self._update_access_count(memory)
                
            # Sort by importance and limit results
            memories.sort(key=lambda m: m.importance.total_score, reverse=True)
            return memories[:query.max_results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
            
    async def consolidate_memories(
        self, 
        agent_id: str, 
        consolidation_type: str = "daily"
    ) -> Optional[MemoryConsolidation]:
        """Consolidate memories to prevent unbounded growth"""
        try:
            # Get memories to consolidate based on type
            if consolidation_type == "daily":
                cutoff_time = datetime.now(timezone.utc) - timedelta(days=1)
                memories = await self._get_memories_since(agent_id, cutoff_time)
            elif consolidation_type == "weekly":
                cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
                memories = await self._get_memories_since(agent_id, cutoff_time)
            else:
                return None
                
            if len(memories) < 5:  # Don't consolidate too few memories
                return None
                
            # Group by type and importance
            memory_groups = self._group_memories_for_consolidation(memories)
            
            # Generate consolidation summary
            summary = await self._generate_consolidation_summary(memory_groups)
            
            # Extract key information to preserve
            entities = set()
            relationships = {}
            insights = []
            
            for memory in memories:
                entities.update(memory.entities)
                for rel_type, targets in memory.relationships.items():
                    if rel_type not in relationships:
                        relationships[rel_type] = []
                    relationships[rel_type].extend(targets)
                    
            # Create consolidation
            consolidation = MemoryConsolidation(
                agent_id=agent_id,
                source_memory_ids=[m.memory_id for m in memories],
                consolidation_type=consolidation_type,
                summary=summary,
                key_insights=insights,
                patterns_identified=self._identify_patterns(memories),
                importance_score=np.mean([m.importance.total_score for m in memories]),
                preserved_entities=list(entities),
                preserved_relationships=relationships
            )
            
            # Store consolidation
            self.consolidation_cache.put(
                f"{agent_id}:{consolidation.consolidation_id}",
                consolidation.dict()
            )
            
            # Mark source memories as consolidated
            for memory in memories:
                memory.context["consolidated"] = True
                memory.context["consolidation_id"] = consolidation.consolidation_id
                await self._update_memory(memory)
                
            logger.info(f"Created {consolidation_type} consolidation for agent {agent_id}")
            return consolidation
            
        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            return None
            
    async def create_reflection(
        self, 
        agent_id: str, 
        reflection_content: str,
        source_memories: List[str]
    ) -> AgentMemory:
        """Create a reflection memory based on past experiences"""
        reflection = AgentMemory(
            agent_id=agent_id,
            type=MemoryType.REFLECTION,
            content=reflection_content,
            source_memories=source_memories,
            importance=MemoryImportance(
                recency_score=1.0,
                frequency_score=0.5,
                relevance_score=0.8,
                emotional_weight=0.3
            )
        )
        
        # Extract entities and relationships from reflection
        reflection.entities = self._extract_entities(reflection_content)
        
        memory_id = await self.save_memory(reflection)
        reflection.memory_id = memory_id
        
        return reflection
        
    async def provide_feedback(self, feedback: MemoryFeedback):
        """Update memory importance based on feedback"""
        try:
            cache_key = f"{feedback.memory_id.split(':')[0]}:{feedback.memory_id}"
            memory_dict = self.memory_cache.get(cache_key)
            
            if memory_dict:
                memory = AgentMemory(**memory_dict)
                
                # Adjust importance based on feedback
                if feedback.was_helpful:
                    memory.importance.relevance_score = min(1.0, memory.importance.relevance_score + 0.1)
                    memory.importance.frequency_score = min(1.0, memory.importance.frequency_score + 0.05)
                else:
                    memory.importance.relevance_score = max(0.0, memory.importance.relevance_score - 0.05)
                    
                await self._update_memory(memory)
                logger.info(f"Updated memory {feedback.memory_id} based on feedback")
                
        except Exception as e:
            logger.error(f"Failed to process memory feedback: {e}")
            
    # Helper methods
    
    def _calculate_importance(self, memory: AgentMemory) -> MemoryImportance:
        """Calculate initial importance scores"""
        # Recency: newer memories are more important
        recency_score = 1.0  # New memories start with max recency
        
        # Frequency: based on entities and keywords
        frequency_score = min(1.0, len(memory.entities) * 0.1 + len(memory.keywords) * 0.05)
        
        # Relevance: based on type and content length
        relevance_weights = {
            MemoryType.REFLECTION: 0.9,
            MemoryType.PROCEDURAL: 0.8,
            MemoryType.SEMANTIC: 0.7,
            MemoryType.EPISODIC: 0.6
        }
        relevance_score = relevance_weights.get(memory.type, 0.5)
        
        # Emotional weight: check for emotional keywords
        emotional_keywords = ['important', 'critical', 'failure', 'success', 'breakthrough']
        emotional_weight = 0.3
        for keyword in emotional_keywords:
            if keyword in memory.content.lower():
                emotional_weight = 0.8
                break
                
        return MemoryImportance(
            recency_score=recency_score,
            frequency_score=frequency_score,
            relevance_score=relevance_score,
            emotional_weight=emotional_weight
        )
        
    async def _store_in_knowledge_graph(self, memory: AgentMemory):
        """Store memory relationships in knowledge graph"""
        operations = []
        
        # Create memory node
        operations.append({
            "operation": "upsert_vertex",
            "label": "Memory",
            "id_key": "memory_id",
            "properties": {
                "memory_id": memory.memory_id,
                "agent_id": memory.agent_id,
                "type": memory.type.value,
                "timestamp": memory.timestamp,
                "importance": memory.importance.total_score
            }
        })
        
        # Create entity nodes and relationships
        for entity in memory.entities:
            operations.append({
                "operation": "upsert_vertex",
                "label": "Entity",
                "id_key": "name",
                "properties": {"name": entity}
            })
            
            operations.append({
                "operation": "upsert_edge",
                "label": "MENTIONS",
                "from_vertex_id": memory.memory_id,
                "to_vertex_id": entity,
                "from_vertex_label": "Memory",
                "to_vertex_label": "Entity",
                "id_key": "memory_id"
            })
            
        await self.kg_client.ingest_operations(operations)
        
    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction (should use NER in production)"""
        # This is a placeholder - in production, use spaCy or similar
        import re
        
        # Extract capitalized words as potential entities
        pattern = r'\b[A-Z][a-z]+\b'
        entities = re.findall(pattern, text)
        
        # Also extract technical terms
        tech_terms = ['API', 'HTTP', 'SQL', 'Docker', 'Kubernetes', 'Python']
        for term in tech_terms:
            if term.lower() in text.lower():
                entities.append(term)
                
        return list(set(entities))
        
    def _identify_patterns(self, memories: List[AgentMemory]) -> List[Dict[str, Any]]:
        """Identify patterns in memories"""
        patterns = []
        
        # Frequency analysis
        entity_counts = {}
        for memory in memories:
            for entity in memory.entities:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
                
        # Find frequently mentioned entities
        frequent_entities = [e for e, c in entity_counts.items() if c >= 3]
        if frequent_entities:
            patterns.append({
                "type": "frequent_entities",
                "entities": frequent_entities,
                "description": f"Frequently mentioned: {', '.join(frequent_entities[:5])}"
            })
            
        # Success/failure patterns for procedural memories
        procedural_memories = [m for m in memories if m.type == MemoryType.PROCEDURAL]
        if procedural_memories:
            success_rate = np.mean([m.success_rate or 0 for m in procedural_memories])
            patterns.append({
                "type": "procedural_success_rate",
                "rate": success_rate,
                "description": f"Average success rate: {success_rate:.2%}"
            })
            
        return patterns
        
    async def _generate_consolidation_summary(self, memory_groups: Dict[str, List[AgentMemory]]) -> str:
        """Generate a summary of grouped memories"""
        # This is a simplified version - in production, use LLM for summarization
        summary_parts = []
        
        for memory_type, memories in memory_groups.items():
            if memories:
                summary_parts.append(f"{memory_type} memories ({len(memories)} total):")
                
                # Get most important memories
                important_memories = sorted(
                    memories, 
                    key=lambda m: m.importance.total_score, 
                    reverse=True
                )[:3]
                
                for memory in important_memories:
                    summary_parts.append(f"- {memory.content[:100]}...")
                    
        return "\n".join(summary_parts)
        
    def _group_memories_for_consolidation(
        self, 
        memories: List[AgentMemory]
    ) -> Dict[str, List[AgentMemory]]:
        """Group memories by type for consolidation"""
        groups = {}
        for memory in memories:
            if memory.type.value not in groups:
                groups[memory.type.value] = []
            groups[memory.type.value].append(memory)
        return groups
        
    async def _update_memory(self, memory: AgentMemory):
        """Update an existing memory"""
        cache_key = f"{memory.agent_id}:{memory.memory_id}"
        self.memory_cache.put(cache_key, memory.dict())
        
    async def _update_access_count(self, memory: AgentMemory):
        """Update access count and timestamp"""
        memory.access_count += 1
        memory.last_accessed = datetime.now(timezone.utc).isoformat()
        await self._update_memory(memory)
        
    async def _update_memory_index(self, agent_id: str, memory_id: str):
        """Update agent's memory index"""
        index_key = f"{agent_id}:memory_index"
        memory_ids = self.memory_cache.get(index_key) or []
        memory_ids.append(memory_id)
        self.memory_cache.put(index_key, memory_ids)
        
    async def _get_recent_memories(self, query: MemoryQuery) -> List[AgentMemory]:
        """Get recent memories for an agent"""
        index_key = f"{query.agent_id}:memory_index"
        memory_ids = self.memory_cache.get(index_key) or []
        
        memories = []
        for memory_id in memory_ids[-query.max_results*2:]:  # Get recent memories
            cache_key = f"{query.agent_id}:{memory_id}"
            memory_dict = self.memory_cache.get(cache_key)
            if memory_dict:
                memories.append(AgentMemory(**memory_dict))
                
        return memories
        
    async def _get_memories_since(
        self, 
        agent_id: str, 
        since: datetime
    ) -> List[AgentMemory]:
        """Get memories since a specific time"""
        index_key = f"{agent_id}:memory_index"
        memory_ids = self.memory_cache.get(index_key) or []
        
        memories = []
        for memory_id in memory_ids:
            cache_key = f"{agent_id}:{memory_id}"
            memory_dict = self.memory_cache.get(cache_key)
            if memory_dict:
                memory = AgentMemory(**memory_dict)
                memory_time = datetime.fromisoformat(memory.timestamp)
                if memory_time >= since:
                    memories.append(memory)
                    
        return memories
        
    def _filter_by_time_range(
        self, 
        memories: List[AgentMemory], 
        time_range: Dict[str, str]
    ) -> List[AgentMemory]:
        """Filter memories by time range"""
        filtered = []
        
        start_time = datetime.fromisoformat(time_range.get("start", "1970-01-01T00:00:00+00:00"))
        end_time = datetime.fromisoformat(time_range.get("end", datetime.now(timezone.utc).isoformat()))
        
        for memory in memories:
            memory_time = datetime.fromisoformat(memory.timestamp)
            if start_time <= memory_time <= end_time:
                filtered.append(memory)
                
        return filtered
        
    async def _get_consolidated_memories(
        self, 
        agent_id: str, 
        query_text: Optional[str]
    ) -> List[AgentMemory]:
        """Get consolidated memories as AgentMemory objects"""
        # This would search through consolidations and convert them to memories
        # For now, return empty list
        return [] 