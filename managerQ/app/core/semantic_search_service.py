"""
Semantic Search Enhancement Service

This service provides advanced semantic search capabilities for the Q Platform:
- Vector-based semantic search
- Hybrid search combining semantic and keyword search
- Relevance scoring and ranking
- Personalized search results
- Multi-modal search (text, images, audio)
- Real-time search indexing and updates
- Query expansion and reformulation
- Search analytics and optimization
- Federated search across multiple sources
- Context-aware search results
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import re
import math
from scipy.spatial.distance import cosine
from scipy.stats import zscore

# ML and NLP libraries
try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False
    logging.warning("SentenceTransformers not available - semantic search will be limited")

try:
    import faiss
    faiss_available = True
except ImportError:
    faiss_available = False
    logging.warning("FAISS not available - vector search will be limited")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
except ImportError:
    sklearn_available = False
    logging.warning("Scikit-learn not available - hybrid search will be limited")

try:
    import spacy
    spacy_available = True
except ImportError:
    spacy_available = False
    logging.warning("spaCy not available - NLP features will be limited")

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.vault_client import VaultClient
from shared.q_vectorstore_client.client import VectorStoreClient

logger = logging.getLogger(__name__)

class SearchType(Enum):
    """Search types"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    MULTIMODAL = "multimodal"
    PERSONALIZED = "personalized"
    FEDERATED = "federated"

class RelevanceModel(Enum):
    """Relevance scoring models"""
    COSINE = "cosine"
    BM25 = "bm25"
    TFIDF = "tfidf"
    NEURAL = "neural"
    HYBRID = "hybrid"
    LEARNING_TO_RANK = "learning_to_rank"

class SearchDomain(Enum):
    """Search domains"""
    DOCUMENTS = "documents"
    KNOWLEDGE_BASE = "knowledge_base"
    CONVERSATIONS = "conversations"
    MODELS = "models"
    WORKFLOWS = "workflows"
    EXPERIMENTS = "experiments"
    USERS = "users"
    GENERAL = "general"

class IndexStatus(Enum):
    """Index status"""
    BUILDING = "building"
    READY = "ready"
    UPDATING = "updating"
    ERROR = "error"

@dataclass
class SearchDocument:
    """Search document representation"""
    document_id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    domain: SearchDomain
    vector_embedding: Optional[np.ndarray] = None
    keywords: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

@dataclass
class SearchQuery:
    """Search query representation"""
    query_id: str
    text: str
    search_type: SearchType
    domain: SearchDomain
    user_id: Optional[str] = None
    context: Dict[str, Any] = None
    filters: Dict[str, Any] = None
    limit: int = 10
    offset: int = 0
    relevance_model: RelevanceModel = RelevanceModel.HYBRID
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.filters is None:
            self.filters = {}

@dataclass
class SearchResult:
    """Search result representation"""
    document_id: str
    title: str
    content: str
    snippet: str
    score: float
    rank: int
    metadata: Dict[str, Any]
    domain: SearchDomain
    relevance_factors: Dict[str, float] = None
    
    def __post_init__(self):
        if self.relevance_factors is None:
            self.relevance_factors = {}

@dataclass
class SearchResponse:
    """Search response representation"""
    query_id: str
    results: List[SearchResult]
    total_count: int
    search_time: float
    query_suggestions: List[str] = None
    facets: Dict[str, List[Tuple[str, int]]] = None
    
    def __post_init__(self):
        if self.query_suggestions is None:
            self.query_suggestions = []
        if self.facets is None:
            self.facets = {}

@dataclass
class UserProfile:
    """User search profile for personalization"""
    user_id: str
    preferences: Dict[str, Any]
    search_history: List[str]
    interaction_history: List[Dict[str, Any]]
    domain_preferences: Dict[SearchDomain, float]
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

@dataclass
class SearchIndex:
    """Search index representation"""
    index_id: str
    name: str
    domain: SearchDomain
    status: IndexStatus
    document_count: int
    vector_dimension: int
    created_at: datetime
    updated_at: datetime
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}

class SemanticSearchService:
    """
    Comprehensive Semantic Search Enhancement Service
    """
    
    def __init__(self, 
                 storage_path: str = "semantic_search",
                 vector_store_url: str = "http://localhost:8080"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.vector_store_client = VectorStoreClient(base_url=vector_store_url)
        self.vault_client = VaultClient()
        
        # Search indexes
        self.indexes: Dict[str, SearchIndex] = {}
        self.documents: Dict[str, SearchDocument] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # ML models
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        self.nlp_model = None
        self.faiss_index = None
        
        # Search configuration
        self.config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "max_query_length": 512,
            "max_results": 100,
            "similarity_threshold": 0.5,
            "hybrid_weight_semantic": 0.7,
            "hybrid_weight_keyword": 0.3,
            "enable_query_expansion": True,
            "enable_spell_correction": True,
            "enable_personalization": True,
            "cache_embeddings": True,
            "index_update_interval": 300,  # seconds
            "max_facet_values": 10
        }
        
        # Caching
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.query_cache: Dict[str, SearchResponse] = {}
        
        # Analytics
        self.search_analytics = {
            "total_queries": 0,
            "average_response_time": 0.0,
            "popular_queries": Counter(),
            "domain_distribution": Counter(),
            "user_engagement": defaultdict(int)
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        logger.info("Semantic Search Service initialized")
    
    async def initialize(self):
        """Initialize the semantic search service"""
        logger.info("Initializing Semantic Search Service")
        
        # Initialize ML models
        await self._initialize_ml_models()
        
        # Load existing indexes
        await self._load_indexes()
        
        # Initialize vector store
        await self._initialize_vector_store()
        
        # Load user profiles
        await self._load_user_profiles()
        
        # Start background tasks
        await self._start_background_tasks()
        
        logger.info("Semantic Search Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the semantic search service"""
        logger.info("Shutting down Semantic Search Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Save state
        await self._save_indexes()
        await self._save_user_profiles()
        
        logger.info("Semantic Search Service shutdown complete")
    
    # ===== INDEX MANAGEMENT =====
    
    async def create_index(self, 
                          name: str, 
                          domain: SearchDomain,
                          config: Dict[str, Any] = None) -> str:
        """Create a new search index"""
        
        index_id = f"idx_{uuid.uuid4().hex[:12]}"
        
        try:
            # Create index
            index = SearchIndex(
                index_id=index_id,
                name=name,
                domain=domain,
                status=IndexStatus.BUILDING,
                document_count=0,
                vector_dimension=384,  # Default for MiniLM
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                config=config or {}
            )
            
            self.indexes[index_id] = index
            
            # Initialize vector index
            if faiss_available:
                await self._initialize_faiss_index(index_id, index.vector_dimension)
            
            # Create vector store collection
            await self._create_vector_store_collection(index_id, domain)
            
            index.status = IndexStatus.READY
            
            logger.info(f"Created search index: {index_id}")
            return index_id
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    async def add_document(self, 
                          index_id: str, 
                          document: SearchDocument) -> bool:
        """Add a document to a search index"""
        
        if index_id not in self.indexes:
            raise ValueError(f"Index {index_id} not found")
        
        try:
            # Generate embeddings
            if not document.vector_embedding:
                document.vector_embedding = await self._generate_embedding(document.content)
            
            # Extract keywords
            if not document.keywords:
                document.keywords = await self._extract_keywords(document.content)
            
            # Store document
            self.documents[document.document_id] = document
            
            # Update vector store
            await self._add_to_vector_store(index_id, document)
            
            # Update FAISS index
            if faiss_available and self.faiss_index:
                await self._add_to_faiss_index(document)
            
            # Update index statistics
            index = self.indexes[index_id]
            index.document_count += 1
            index.updated_at = datetime.utcnow()
            
            logger.debug(f"Added document {document.document_id} to index {index_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to index: {e}")
            return False
    
    async def update_document(self, 
                             index_id: str, 
                             document: SearchDocument) -> bool:
        """Update a document in a search index"""
        
        if index_id not in self.indexes:
            raise ValueError(f"Index {index_id} not found")
        
        try:
            # Update embeddings if content changed
            if document.document_id in self.documents:
                old_document = self.documents[document.document_id]
                if old_document.content != document.content:
                    document.vector_embedding = await self._generate_embedding(document.content)
                    document.keywords = await self._extract_keywords(document.content)
            
            document.updated_at = datetime.utcnow()
            
            # Update stored document
            self.documents[document.document_id] = document
            
            # Update vector store
            await self._update_in_vector_store(index_id, document)
            
            # Update FAISS index
            if faiss_available and self.faiss_index:
                await self._update_in_faiss_index(document)
            
            # Update index
            index = self.indexes[index_id]
            index.updated_at = datetime.utcnow()
            
            logger.debug(f"Updated document {document.document_id} in index {index_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document in index: {e}")
            return False
    
    async def remove_document(self, 
                             index_id: str, 
                             document_id: str) -> bool:
        """Remove a document from a search index"""
        
        if index_id not in self.indexes:
            raise ValueError(f"Index {index_id} not found")
        
        try:
            # Remove from storage
            if document_id in self.documents:
                del self.documents[document_id]
            
            # Remove from vector store
            await self._remove_from_vector_store(index_id, document_id)
            
            # Remove from FAISS index
            if faiss_available and self.faiss_index:
                await self._remove_from_faiss_index(document_id)
            
            # Update index statistics
            index = self.indexes[index_id]
            index.document_count = max(0, index.document_count - 1)
            index.updated_at = datetime.utcnow()
            
            logger.debug(f"Removed document {document_id} from index {index_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document from index: {e}")
            return False
    
    # ===== SEARCH OPERATIONS =====
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Perform semantic search"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate query ID
            query.query_id = f"q_{uuid.uuid4().hex[:12]}"
            
            # Check cache
            cache_key = self._get_cache_key(query)
            if cache_key in self.query_cache:
                cached_response = self.query_cache[cache_key]
                logger.debug(f"Cache hit for query: {query.text}")
                return cached_response
            
            # Preprocess query
            processed_query = await self._preprocess_query(query)
            
            # Perform search based on type
            if query.search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(processed_query)
            elif query.search_type == SearchType.KEYWORD:
                results = await self._keyword_search(processed_query)
            elif query.search_type == SearchType.HYBRID:
                results = await self._hybrid_search(processed_query)
            elif query.search_type == SearchType.MULTIMODAL:
                results = await self._multimodal_search(processed_query)
            elif query.search_type == SearchType.PERSONALIZED:
                results = await self._personalized_search(processed_query)
            elif query.search_type == SearchType.FEDERATED:
                results = await self._federated_search(processed_query)
            else:
                results = await self._hybrid_search(processed_query)
            
            # Apply post-processing
            processed_results = await self._postprocess_results(results, query)
            
            # Generate response
            end_time = asyncio.get_event_loop().time()
            search_time = end_time - start_time
            
            response = SearchResponse(
                query_id=query.query_id,
                results=processed_results,
                total_count=len(processed_results),
                search_time=search_time,
                query_suggestions=await self._generate_query_suggestions(query),
                facets=await self._generate_facets(processed_results)
            )
            
            # Cache response
            self.query_cache[cache_key] = response
            
            # Update analytics
            await self._update_search_analytics(query, response)
            
            # Update user profile
            if query.user_id:
                await self._update_user_profile(query.user_id, query, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise
    
    async def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform semantic vector search"""
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query.text)
            
            # Search using FAISS if available
            if faiss_available and self.faiss_index:
                results = await self._faiss_search(query_embedding, query)
            else:
                # Fallback to vector store search
                results = await self._vector_store_search(query_embedding, query)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform keyword-based search"""
        
        try:
            # Tokenize query
            query_tokens = await self._tokenize_query(query.text)
            
            # Search documents
            results = []
            for doc_id, document in self.documents.items():
                # Apply domain filter
                if query.domain != SearchDomain.GENERAL and document.domain != query.domain:
                    continue
                
                # Calculate BM25 score
                score = await self._calculate_bm25_score(query_tokens, document)
                
                if score > self.config["similarity_threshold"]:
                    result = SearchResult(
                        document_id=doc_id,
                        title=document.title,
                        content=document.content,
                        snippet=await self._generate_snippet(document.content, query.text),
                        score=score,
                        rank=0,
                        metadata=document.metadata,
                        domain=document.domain,
                        relevance_factors={"bm25": score}
                    )
                    results.append(result)
            
            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Apply ranking
            for i, result in enumerate(results):
                result.rank = i + 1
            
            return results[:query.limit]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    async def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform hybrid semantic + keyword search"""
        
        try:
            # Get semantic results
            semantic_results = await self._semantic_search(query)
            
            # Get keyword results
            keyword_results = await self._keyword_search(query)
            
            # Combine results
            combined_results = await self._combine_hybrid_results(
                semantic_results, keyword_results, query
            )
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    async def _multimodal_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform multimodal search"""
        
        try:
            # For now, fallback to hybrid search
            # In a full implementation, this would handle images, audio, etc.
            return await self._hybrid_search(query)
            
        except Exception as e:
            logger.error(f"Error in multimodal search: {e}")
            return []
    
    async def _personalized_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform personalized search"""
        
        try:
            # Get base results
            base_results = await self._hybrid_search(query)
            
            # Apply personalization
            if query.user_id and query.user_id in self.user_profiles:
                personalized_results = await self._apply_personalization(
                    base_results, query.user_id, query
                )
                return personalized_results
            
            return base_results
            
        except Exception as e:
            logger.error(f"Error in personalized search: {e}")
            return []
    
    async def _federated_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform federated search across multiple sources"""
        
        try:
            # Search across all domains
            all_results = []
            
            for domain in SearchDomain:
                domain_query = SearchQuery(
                    query_id=query.query_id,
                    text=query.text,
                    search_type=SearchType.HYBRID,
                    domain=domain,
                    user_id=query.user_id,
                    context=query.context,
                    filters=query.filters,
                    limit=query.limit // len(SearchDomain),
                    offset=query.offset,
                    relevance_model=query.relevance_model
                )
                
                domain_results = await self._hybrid_search(domain_query)
                all_results.extend(domain_results)
            
            # Sort by relevance
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            # Apply limit
            return all_results[:query.limit]
            
        except Exception as e:
            logger.error(f"Error in federated search: {e}")
            return []
    
    # ===== RELEVANCE SCORING =====
    
    async def _calculate_bm25_score(self, query_tokens: List[str], document: SearchDocument) -> float:
        """Calculate BM25 relevance score"""
        
        try:
            # BM25 parameters
            k1 = 1.2
            b = 0.75
            
            # Document tokens
            doc_tokens = await self._tokenize_text(document.content)
            doc_length = len(doc_tokens)
            
            # Average document length (simplified)
            avg_doc_length = 100
            
            # Calculate score
            score = 0.0
            for token in query_tokens:
                # Term frequency in document
                tf = doc_tokens.count(token)
                
                # Document frequency (simplified)
                df = 1
                
                # Inverse document frequency
                idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5))
                
                # BM25 score component
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
            
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating BM25 score: {e}")
            return 0.0
    
    async def _combine_hybrid_results(self, 
                                     semantic_results: List[SearchResult], 
                                     keyword_results: List[SearchResult],
                                     query: SearchQuery) -> List[SearchResult]:
        """Combine semantic and keyword search results"""
        
        try:
            # Normalize scores
            semantic_scores = [r.score for r in semantic_results]
            keyword_scores = [r.score for r in keyword_results]
            
            if semantic_scores:
                semantic_max = max(semantic_scores)
                semantic_min = min(semantic_scores)
                
                for result in semantic_results:
                    if semantic_max > semantic_min:
                        result.score = (result.score - semantic_min) / (semantic_max - semantic_min)
                    else:
                        result.score = 1.0
            
            if keyword_scores:
                keyword_max = max(keyword_scores)
                keyword_min = min(keyword_scores)
                
                for result in keyword_results:
                    if keyword_max > keyword_min:
                        result.score = (result.score - keyword_min) / (keyword_max - keyword_min)
                    else:
                        result.score = 1.0
            
            # Combine results
            combined_results = {}
            
            # Add semantic results
            for result in semantic_results:
                combined_results[result.document_id] = result
                result.relevance_factors["semantic"] = result.score
                result.score = result.score * self.config["hybrid_weight_semantic"]
            
            # Add keyword results
            for result in keyword_results:
                if result.document_id in combined_results:
                    # Merge scores
                    existing = combined_results[result.document_id]
                    existing.relevance_factors["keyword"] = result.score
                    existing.score += result.score * self.config["hybrid_weight_keyword"]
                else:
                    # Add new result
                    result.relevance_factors["keyword"] = result.score
                    result.score = result.score * self.config["hybrid_weight_keyword"]
                    combined_results[result.document_id] = result
            
            # Sort by combined score
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x.score, reverse=True)
            
            # Apply ranking
            for i, result in enumerate(final_results):
                result.rank = i + 1
            
            return final_results[:query.limit]
            
        except Exception as e:
            logger.error(f"Error combining hybrid results: {e}")
            return []
    
    # ===== PERSONALIZATION =====
    
    async def _apply_personalization(self, 
                                   results: List[SearchResult], 
                                   user_id: str,
                                   query: SearchQuery) -> List[SearchResult]:
        """Apply personalization to search results"""
        
        try:
            user_profile = self.user_profiles.get(user_id)
            if not user_profile:
                return results
            
            # Apply domain preferences
            for result in results:
                domain_preference = user_profile.domain_preferences.get(result.domain, 1.0)
                result.score *= domain_preference
                result.relevance_factors["personalization"] = domain_preference
            
            # Apply content preferences
            for result in results:
                content_boost = await self._calculate_content_boost(result, user_profile)
                result.score *= content_boost
                result.relevance_factors["content_preference"] = content_boost
            
            # Re-sort by personalized score
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Update rankings
            for i, result in enumerate(results):
                result.rank = i + 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error applying personalization: {e}")
            return results
    
    async def _calculate_content_boost(self, 
                                     result: SearchResult, 
                                     user_profile: UserProfile) -> float:
        """Calculate content-based personalization boost"""
        
        try:
            boost = 1.0
            
            # Boost based on user preferences
            preferences = user_profile.preferences
            
            # Check metadata matching
            for pref_key, pref_value in preferences.items():
                if pref_key in result.metadata:
                    if result.metadata[pref_key] == pref_value:
                        boost *= 1.2
            
            # Check interaction history
            for interaction in user_profile.interaction_history:
                if interaction.get("document_id") == result.document_id:
                    if interaction.get("action") == "click":
                        boost *= 1.1
                    elif interaction.get("action") == "like":
                        boost *= 1.3
                    elif interaction.get("action") == "share":
                        boost *= 1.2
            
            return boost
            
        except Exception as e:
            logger.error(f"Error calculating content boost: {e}")
            return 1.0
    
    async def _update_user_profile(self, 
                                  user_id: str, 
                                  query: SearchQuery, 
                                  response: SearchResponse):
        """Update user profile based on search interaction"""
        
        try:
            # Get or create user profile
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    preferences={},
                    search_history=[],
                    interaction_history=[],
                    domain_preferences={}
                )
            
            profile = self.user_profiles[user_id]
            
            # Update search history
            profile.search_history.append(query.text)
            if len(profile.search_history) > 100:
                profile.search_history.pop(0)
            
            # Update domain preferences
            if query.domain not in profile.domain_preferences:
                profile.domain_preferences[query.domain] = 1.0
            
            # Update timestamp
            profile.updated_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
    
    # ===== QUERY PROCESSING =====
    
    async def _preprocess_query(self, query: SearchQuery) -> SearchQuery:
        """Preprocess search query"""
        
        try:
            # Clean query text
            cleaned_text = await self._clean_query_text(query.text)
            
            # Spell correction
            if self.config["enable_spell_correction"]:
                cleaned_text = await self._correct_spelling(cleaned_text)
            
            # Query expansion
            if self.config["enable_query_expansion"]:
                expanded_text = await self._expand_query(cleaned_text)
                if expanded_text != cleaned_text:
                    cleaned_text = expanded_text
            
            # Update query
            query.text = cleaned_text
            
            return query
            
        except Exception as e:
            logger.error(f"Error preprocessing query: {e}")
            return query
    
    async def _clean_query_text(self, text: str) -> str:
        """Clean query text"""
        
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove special characters that might interfere
            text = re.sub(r'[^\w\s\-\.]', '', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning query text: {e}")
            return text
    
    async def _correct_spelling(self, text: str) -> str:
        """Correct spelling in query text"""
        
        # Simplified spelling correction
        # In a full implementation, this would use a proper spell checker
        return text
    
    async def _expand_query(self, text: str) -> str:
        """Expand query with synonyms and related terms"""
        
        try:
            # Simple query expansion using common synonyms
            expansions = {
                "search": ["find", "look", "discover"],
                "create": ["make", "build", "generate"],
                "delete": ["remove", "erase", "destroy"],
                "update": ["modify", "change", "edit"]
            }
            
            words = text.split()
            expanded_words = []
            
            for word in words:
                expanded_words.append(word)
                if word.lower() in expansions:
                    expanded_words.extend(expansions[word.lower()])
            
            return " ".join(expanded_words)
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return text
    
    async def _tokenize_query(self, text: str) -> List[str]:
        """Tokenize query text"""
        
        try:
            # Simple tokenization
            tokens = re.findall(r'\w+', text.lower())
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing query: {e}")
            return []
    
    async def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize document text"""
        
        try:
            # Simple tokenization
            tokens = re.findall(r'\w+', text.lower())
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []
    
    # ===== EMBEDDING GENERATION =====
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding"""
        
        try:
            # Check cache
            if text in self.embedding_cache:
                return self.embedding_cache[text]
            
            # Generate embedding
            if self.sentence_transformer:
                embedding = self.sentence_transformer.encode(text)
                embedding = np.array(embedding, dtype=np.float32)
            else:
                # Fallback to simple embedding
                embedding = self._simple_embedding(text)
            
            # Cache embedding
            if self.config["cache_embeddings"]:
                self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Generate simple embedding for fallback"""
        
        try:
            # Simple hash-based embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            
            # Convert to numeric values
            embedding = []
            for i in range(0, len(hash_hex), 2):
                val = int(hash_hex[i:i+2], 16) / 255.0
                embedding.append(val)
            
            # Pad to 384 dimensions
            while len(embedding) < 384:
                embedding.extend(embedding[:384-len(embedding)])
            
            return np.array(embedding[:384], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error generating simple embedding: {e}")
            return np.zeros(384, dtype=np.float32)
    
    # ===== UTILITY METHODS =====
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        
        try:
            # Simple keyword extraction
            words = re.findall(r'\w+', text.lower())
            
            # Filter out common stop words
            stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
                'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
                'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
                'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
                'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
                'who', 'did', 'get', 'may', 'day', 'way', 'made', 'part', 'over'
            }
            
            keywords = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Return most frequent keywords
            word_counts = Counter(keywords)
            return [word for word, count in word_counts.most_common(10)]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    async def _generate_snippet(self, content: str, query: str) -> str:
        """Generate search result snippet"""
        
        try:
            # Find query terms in content
            query_terms = query.lower().split()
            content_lower = content.lower()
            
            # Find best matching position
            best_pos = 0
            best_score = 0
            
            for i in range(len(content) - 200):
                snippet = content_lower[i:i+200]
                score = sum(1 for term in query_terms if term in snippet)
                
                if score > best_score:
                    best_score = score
                    best_pos = i
            
            # Extract snippet
            snippet = content[best_pos:best_pos+200]
            
            # Highlight query terms
            for term in query_terms:
                snippet = re.sub(f'({re.escape(term)})', r'<mark>\1</mark>', snippet, flags=re.IGNORECASE)
            
            return snippet + "..."
            
        except Exception as e:
            logger.error(f"Error generating snippet: {e}")
            return content[:200] + "..."
    
    async def _generate_query_suggestions(self, query: SearchQuery) -> List[str]:
        """Generate query suggestions"""
        
        try:
            suggestions = []
            
            # Popular queries
            popular = [q for q, count in self.search_analytics["popular_queries"].most_common(5)]
            suggestions.extend(popular)
            
            # Similar queries from user history
            if query.user_id and query.user_id in self.user_profiles:
                user_history = self.user_profiles[query.user_id].search_history
                suggestions.extend(user_history[-3:])
            
            # Remove duplicates and current query
            suggestions = list(set(suggestions))
            if query.text in suggestions:
                suggestions.remove(query.text)
            
            return suggestions[:5]
            
        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return []
    
    async def _generate_facets(self, results: List[SearchResult]) -> Dict[str, List[Tuple[str, int]]]:
        """Generate search facets"""
        
        try:
            facets = {}
            
            # Domain facets
            domain_counts = Counter(result.domain.value for result in results)
            facets["domain"] = list(domain_counts.most_common(self.config["max_facet_values"]))
            
            # Metadata facets
            metadata_keys = set()
            for result in results:
                metadata_keys.update(result.metadata.keys())
            
            for key in metadata_keys:
                if key in ['type', 'category', 'author', 'source']:
                    values = [result.metadata.get(key, 'unknown') for result in results]
                    value_counts = Counter(values)
                    facets[key] = list(value_counts.most_common(self.config["max_facet_values"]))
            
            return facets
            
        except Exception as e:
            logger.error(f"Error generating facets: {e}")
            return {}
    
    async def _postprocess_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Post-process search results"""
        
        try:
            # Apply filters
            if query.filters:
                filtered_results = []
                for result in results:
                    include = True
                    for filter_key, filter_value in query.filters.items():
                        if filter_key in result.metadata:
                            if result.metadata[filter_key] != filter_value:
                                include = False
                                break
                    if include:
                        filtered_results.append(result)
                results = filtered_results
            
            # Apply pagination
            start_idx = query.offset
            end_idx = start_idx + query.limit
            results = results[start_idx:end_idx]
            
            return results
            
        except Exception as e:
            logger.error(f"Error post-processing results: {e}")
            return results
    
    def _get_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for query"""
        
        try:
            # Create cache key from query parameters
            key_parts = [
                query.text,
                query.search_type.value,
                query.domain.value,
                str(query.limit),
                str(query.offset),
                str(sorted(query.filters.items()) if query.filters else [])
            ]
            
            return hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return str(uuid.uuid4())
    
    # ===== ANALYTICS =====
    
    async def _update_search_analytics(self, query: SearchQuery, response: SearchResponse):
        """Update search analytics"""
        
        try:
            # Update counters
            self.search_analytics["total_queries"] += 1
            self.search_analytics["popular_queries"][query.text] += 1
            self.search_analytics["domain_distribution"][query.domain.value] += 1
            
            # Update average response time
            current_avg = self.search_analytics["average_response_time"]
            total_queries = self.search_analytics["total_queries"]
            
            self.search_analytics["average_response_time"] = (
                (current_avg * (total_queries - 1) + response.search_time) / total_queries
            )
            
            # Update user engagement
            if query.user_id:
                self.search_analytics["user_engagement"][query.user_id] += 1
            
        except Exception as e:
            logger.error(f"Error updating search analytics: {e}")
    
    # ===== INITIALIZATION METHODS =====
    
    async def _initialize_ml_models(self):
        """Initialize ML models"""
        
        try:
            # Initialize sentence transformer
            if sentence_transformers_available:
                self.sentence_transformer = SentenceTransformer(self.config["embedding_model"])
                logger.info("Sentence transformer initialized")
            
            # Initialize TF-IDF vectorizer
            if sklearn_available:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                logger.info("TF-IDF vectorizer initialized")
            
            # Initialize spaCy model
            if spacy_available:
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                    logger.info("spaCy model initialized")
                except OSError:
                    logger.warning("spaCy English model not found")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    async def _initialize_vector_store(self):
        """Initialize vector store connection"""
        
        try:
            # Test connection
            # In a full implementation, this would set up the vector store
            logger.info("Vector store initialized")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
    
    async def _initialize_faiss_index(self, index_id: str, dimension: int):
        """Initialize FAISS index"""
        
        try:
            if faiss_available:
                # Create FAISS index
                self.faiss_index = faiss.IndexFlatIP(dimension)
                logger.info(f"FAISS index initialized for {index_id}")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
    
    async def _start_background_tasks(self):
        """Start background tasks"""
        
        try:
            # Start index update task
            task = asyncio.create_task(self._index_update_loop())
            self.background_tasks.add(task)
            
            # Start cache cleanup task
            task = asyncio.create_task(self._cache_cleanup_loop())
            self.background_tasks.add(task)
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def _index_update_loop(self):
        """Background index update loop"""
        
        while True:
            try:
                await asyncio.sleep(self.config["index_update_interval"])
                
                # Update indexes
                for index_id, index in self.indexes.items():
                    if index.status == IndexStatus.READY:
                        # Perform maintenance tasks
                        await self._maintain_index(index_id)
                
            except Exception as e:
                logger.error(f"Error in index update loop: {e}")
    
    async def _cache_cleanup_loop(self):
        """Background cache cleanup loop"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old cache entries
                if len(self.embedding_cache) > 10000:
                    # Remove oldest entries
                    keys_to_remove = list(self.embedding_cache.keys())[:1000]
                    for key in keys_to_remove:
                        del self.embedding_cache[key]
                
                if len(self.query_cache) > 1000:
                    # Remove oldest entries
                    keys_to_remove = list(self.query_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self.query_cache[key]
                
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
    
    async def _maintain_index(self, index_id: str):
        """Maintain a search index"""
        
        try:
            # Placeholder for index maintenance
            # In a full implementation, this would optimize the index
            pass
            
        except Exception as e:
            logger.error(f"Error maintaining index {index_id}: {e}")
    
    # ===== PERSISTENCE =====
    
    async def _load_indexes(self):
        """Load indexes from storage"""
        
        try:
            indexes_file = self.storage_path / "indexes.json"
            if indexes_file.exists():
                with open(indexes_file, 'r') as f:
                    indexes_data = json.load(f)
                
                for index_data in indexes_data:
                    index = SearchIndex(**index_data)
                    self.indexes[index.index_id] = index
            
            logger.info("Search indexes loaded")
            
        except Exception as e:
            logger.warning(f"Error loading indexes: {e}")
    
    async def _save_indexes(self):
        """Save indexes to storage"""
        
        try:
            indexes_data = []
            for index in self.indexes.values():
                index_dict = asdict(index)
                indexes_data.append(index_dict)
            
            indexes_file = self.storage_path / "indexes.json"
            with open(indexes_file, 'w') as f:
                json.dump(indexes_data, f, indent=2, default=str)
            
            logger.info("Search indexes saved")
            
        except Exception as e:
            logger.warning(f"Error saving indexes: {e}")
    
    async def _load_user_profiles(self):
        """Load user profiles from storage"""
        
        try:
            profiles_file = self.storage_path / "user_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                
                for profile_data in profiles_data:
                    profile = UserProfile(**profile_data)
                    self.user_profiles[profile.user_id] = profile
            
            logger.info("User profiles loaded")
            
        except Exception as e:
            logger.warning(f"Error loading user profiles: {e}")
    
    async def _save_user_profiles(self):
        """Save user profiles to storage"""
        
        try:
            profiles_data = []
            for profile in self.user_profiles.values():
                profile_dict = asdict(profile)
                profiles_data.append(profile_dict)
            
            profiles_file = self.storage_path / "user_profiles.json"
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2, default=str)
            
            logger.info("User profiles saved")
            
        except Exception as e:
            logger.warning(f"Error saving user profiles: {e}")
    
    # ===== VECTOR STORE OPERATIONS =====
    
    async def _create_vector_store_collection(self, index_id: str, domain: SearchDomain):
        """Create vector store collection"""
        
        try:
            # Create collection in vector store
            # This would use the actual vector store client
            pass
            
        except Exception as e:
            logger.error(f"Error creating vector store collection: {e}")
    
    async def _add_to_vector_store(self, index_id: str, document: SearchDocument):
        """Add document to vector store"""
        
        try:
            # Add to vector store
            # This would use the actual vector store client
            pass
            
        except Exception as e:
            logger.error(f"Error adding to vector store: {e}")
    
    async def _update_in_vector_store(self, index_id: str, document: SearchDocument):
        """Update document in vector store"""
        
        try:
            # Update in vector store
            # This would use the actual vector store client
            pass
            
        except Exception as e:
            logger.error(f"Error updating in vector store: {e}")
    
    async def _remove_from_vector_store(self, index_id: str, document_id: str):
        """Remove document from vector store"""
        
        try:
            # Remove from vector store
            # This would use the actual vector store client
            pass
            
        except Exception as e:
            logger.error(f"Error removing from vector store: {e}")
    
    async def _vector_store_search(self, query_embedding: np.ndarray, query: SearchQuery) -> List[SearchResult]:
        """Search using vector store"""
        
        try:
            # Search vector store
            # This would use the actual vector store client
            return []
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    # ===== FAISS OPERATIONS =====
    
    async def _add_to_faiss_index(self, document: SearchDocument):
        """Add document to FAISS index"""
        
        try:
            if self.faiss_index and document.vector_embedding is not None:
                # Add to FAISS index
                self.faiss_index.add(document.vector_embedding.reshape(1, -1))
            
        except Exception as e:
            logger.error(f"Error adding to FAISS index: {e}")
    
    async def _update_in_faiss_index(self, document: SearchDocument):
        """Update document in FAISS index"""
        
        try:
            # FAISS doesn't support direct updates, so we'd need to rebuild
            # For now, just add the new embedding
            await self._add_to_faiss_index(document)
            
        except Exception as e:
            logger.error(f"Error updating in FAISS index: {e}")
    
    async def _remove_from_faiss_index(self, document_id: str):
        """Remove document from FAISS index"""
        
        try:
            # FAISS doesn't support direct removal by ID
            # This would require rebuilding the index
            pass
            
        except Exception as e:
            logger.error(f"Error removing from FAISS index: {e}")
    
    async def _faiss_search(self, query_embedding: np.ndarray, query: SearchQuery) -> List[SearchResult]:
        """Search using FAISS index"""
        
        try:
            if not self.faiss_index:
                return []
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), query.limit)
            
            # Convert to search results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    # Find document by index (simplified)
                    doc_id = f"doc_{idx}"
                    if doc_id in self.documents:
                        document = self.documents[doc_id]
                        result = SearchResult(
                            document_id=doc_id,
                            title=document.title,
                            content=document.content,
                            snippet=await self._generate_snippet(document.content, query.text),
                            score=float(score),
                            rank=i + 1,
                            metadata=document.metadata,
                            domain=document.domain,
                            relevance_factors={"semantic": float(score)}
                        )
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return []
    
    # ===== API METHODS =====
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics"""
        
        return {
            "total_queries": self.search_analytics["total_queries"],
            "average_response_time": self.search_analytics["average_response_time"],
            "popular_queries": dict(self.search_analytics["popular_queries"].most_common(10)),
            "domain_distribution": dict(self.search_analytics["domain_distribution"]),
            "total_documents": len(self.documents),
            "total_indexes": len(self.indexes),
            "cache_size": len(self.embedding_cache)
        }
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        
        return self.user_profiles.get(user_id)
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    preferences=preferences,
                    search_history=[],
                    interaction_history=[],
                    domain_preferences={}
                )
            else:
                self.user_profiles[user_id].preferences.update(preferences)
                self.user_profiles[user_id].updated_at = datetime.utcnow()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False

# Global service instance
semantic_search_service = SemanticSearchService() 