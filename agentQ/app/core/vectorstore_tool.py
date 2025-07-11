import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from shared.q_vectorstore_client.client import VectorStoreClient
from shared.q_vectorstore_client.models import Query
from agentQ.app.core.toolbox import Tool

logger = logging.getLogger(__name__)

# Initialize embedding model with a more advanced model
try:
    # Use a state-of-the-art embedding model
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    logger.info("Advanced embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embedding_model = None

# Collection names
COLLECTION_NAME = "rag_document_chunks"
CODE_COLLECTION = "code_documentation"
INSIGHTS_COLLECTION = "insights"
LESSONS_COLLECTION = "lessons_learned"

class SemanticSearchEngine:
    """Advanced semantic search engine with query optimization and reranking"""
    
    def __init__(self):
        self.embedding_model = embedding_model
        self.query_cache = {}
        self.collection_metadata = {}
    
    def optimize_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Optimize query for better semantic search results"""
        # Query expansion based on context
        expanded_query = query
        
        if context:
            # Add relevant context to improve search
            if context.get('domain'):
                expanded_query = f"{context['domain']} {query}"
            if context.get('intent'):
                intent_keywords = {
                    'code': 'implementation source code function class',
                    'concept': 'explanation theory concept definition',
                    'troubleshoot': 'error problem issue solution fix',
                    'architecture': 'design pattern structure system'
                }
                if context['intent'] in intent_keywords:
                    expanded_query += f" {intent_keywords[context['intent']]}"
        
        # Query reformulation for better embedding
        # Remove stop words and focus on key terms
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be'}
        words = expanded_query.lower().split()
        filtered_words = [w for w in words if w not in stop_words or len(words) < 5]
        
        return ' '.join(filtered_words)
    
    def generate_embeddings(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embeddings with caching"""
        if use_cache and text in self.query_cache:
            return self.query_cache[text]
        
        if not self.embedding_model:
            raise ValueError("Embedding model not available")
        
        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        
        # Cache the embedding
        if use_cache:
            self.query_cache[text] = embedding.tolist()
        
        return embedding.tolist()
    
    async def hybrid_search(self, query: str, collection: str, top_k: int = 10,
                          filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword matching"""
        results = []
        
        # Generate query embedding
        query_embedding = self.generate_embeddings(query)
        
        # Perform semantic search
        vector_store_url = filters.get("vector_store_url") if filters else None
        if not vector_store_url:
            raise ValueError("vector_store_url not configured")
        
        vs_client = VectorStoreClient(base_url=vector_store_url)
        
        try:
            # Create search query
            search_query = Query(values=query_embedding, top_k=top_k * 2)  # Get more for reranking
            
            # Add metadata filters if provided
            if filters and 'metadata_filters' in filters:
                search_query.filter = filters['metadata_filters']
            
            # Execute search
            search_response = await vs_client.search(
                collection_name=collection,
                queries=[search_query]
            )
            
            if search_response.results and search_response.results[0].hits:
                # Process and rerank results
                for hit in search_response.results[0].hits:
                    result = {
                        'score': hit.score,
                        'content': hit.metadata.get('text_chunk', hit.metadata.get('content', '')),
                        'metadata': hit.metadata,
                        'id': hit.id
                    }
                    
                    # Calculate additional relevance scores
                    keyword_score = self._calculate_keyword_relevance(query, result['content'])
                    result['combined_score'] = 0.7 * hit.score + 0.3 * keyword_score
                    
                    results.append(result)
                
                # Sort by combined score
                results.sort(key=lambda x: x['combined_score'], reverse=True)
                
        finally:
            await vs_client.close()
        
        return results[:top_k]
    
    def _calculate_keyword_relevance(self, query: str, content: str) -> float:
        """Calculate keyword-based relevance score"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def multi_collection_search(self, query: str, collections: List[str], 
                                    top_k_per_collection: int = 5,
                                    config: Dict[str, Any] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Search across multiple collections and aggregate results"""
        all_results = {}
        
        # Search each collection in parallel
        search_tasks = []
        for collection in collections:
            task = self.hybrid_search(query, collection, top_k_per_collection, config)
            search_tasks.append(task)
        
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        for collection, result in zip(collections, results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for collection {collection}: {result}")
                all_results[collection] = []
            else:
                all_results[collection] = result
        
        return all_results
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]], 
                      rerank_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Rerank results using a cross-encoder model"""
        if not results:
            return results
        
        if rerank_model and self.embedding_model:
            # In production, use a cross-encoder model for reranking
            # For now, we'll use a simple heuristic
            for result in results:
                # Boost results that contain exact query phrases
                if query.lower() in result['content'].lower():
                    result['rerank_score'] = result.get('combined_score', result['score']) * 1.2
                else:
                    result['rerank_score'] = result.get('combined_score', result['score'])
            
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return results

# Global search engine instance
semantic_search_engine = SemanticSearchEngine()

# --- Enhanced Tool Functions ---

async def search_knowledge_base_async(query: str, top_k: int = 5, 
                                    collection: str = COLLECTION_NAME,
                                    config: dict = None) -> str:
    """
    Advanced semantic search across the knowledge base with query optimization and reranking.
    
    Args:
        query: The search query
        top_k: Number of results to return
        collection: Collection to search in
        config: Configuration including vector_store_url
        
    Returns:
        Formatted search results or error message
    """
    if not embedding_model:
        return "Error: Embedding model is not available."
    
    if not config or "vector_store_url" not in config:
        return "Error: vector_store_url not found in configuration."
    
    try:
        # Detect query intent and optimize
        context = {
            'intent': _detect_query_intent(query),
            'domain': _detect_domain(query)
        }
        
        optimized_query = semantic_search_engine.optimize_query(query, context)
        logger.info(f"Optimized query: '{query}' -> '{optimized_query}'")
        
        # Perform hybrid search
        results = await semantic_search_engine.hybrid_search(
            optimized_query, 
            collection, 
            top_k,
            config
        )
        
        if not results:
            return f"No relevant information found in {collection} for query: {query}"
        
        # Rerank results
        reranked_results = semantic_search_engine.rerank_results(query, results)
        
        # Format results with rich metadata
        formatted_results = []
        for i, result in enumerate(reranked_results):
            metadata = result['metadata']
            source = metadata.get('source', metadata.get('file_path', 'Unknown'))
            
            # Add context information
            context_info = []
            if metadata.get('timestamp'):
                context_info.append(f"Updated: {metadata['timestamp']}")
            if metadata.get('author'):
                context_info.append(f"Author: {metadata['author']}")
            if metadata.get('tags'):
                context_info.append(f"Tags: {', '.join(metadata['tags'])}")
            
            context_str = " | ".join(context_info) if context_info else ""
            
            formatted_results.append(
                f"Result {i+1} (Score: {result['combined_score']:.3f}):\n"
                f"Source: {source}\n"
                f"{context_str}\n"
                f"Content: {result['content'][:500]}..."
                if len(result['content']) > 500 else result['content']
            )
        
        return "\n\n---\n\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}", exc_info=True)
        return f"Error: Failed to search knowledge base: {str(e)}"

def search_knowledge_base(query: str, top_k: int = 5, collection: str = COLLECTION_NAME, 
                         config: dict = None) -> str:
    """Synchronous wrapper for search_knowledge_base_async"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        search_knowledge_base_async(query, top_k, collection, config)
    )

async def search_codebase_async(query: str, top_k: int = 5, config: dict = None) -> str:
    """
    Search the codebase with code-specific optimizations.
    
    Args:
        query: Code search query
        top_k: Number of results
        config: Configuration
        
    Returns:
        Formatted code search results
    """
    if not config:
        config = {}
    
    # Add code-specific context
    code_context = {
        'intent': 'code',
        'domain': 'implementation'
    }
    
    # Optimize query for code search
    optimized_query = semantic_search_engine.optimize_query(query, code_context)
    
    # Add programming language keywords if detected
    lang_keywords = {
        'python': ['def', 'class', 'import', 'async', 'await'],
        'javascript': ['function', 'const', 'let', 'var', 'export'],
        'java': ['public', 'private', 'class', 'interface', 'extends'],
        'go': ['func', 'type', 'struct', 'interface', 'package']
    }
    
    for lang, keywords in lang_keywords.items():
        if any(kw in query.lower() for kw in keywords):
            optimized_query = f"{lang} {optimized_query}"
            break
    
    # Search in code collection
    result = await search_knowledge_base_async(
        optimized_query, 
        top_k, 
        CODE_COLLECTION,
        config
    )
    
    # Format code results with syntax highlighting hints
    if "Result" in result:
        # Add code formatting
        result = result.replace("Content:", "Code:\n```")
        result = result.replace("\n\n---", "```\n\n---")
    
    return result

def search_codebase(query: str, top_k: int = 5, config: dict = None) -> str:
    """Synchronous wrapper for search_codebase_async"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        search_codebase_async(query, top_k, config)
    )

async def multi_collection_search_async(query: str, collections: Optional[List[str]] = None,
                                      top_k: int = 3, config: dict = None) -> str:
    """
    Search across multiple collections and aggregate results.
    
    Args:
        query: Search query
        collections: List of collections to search (None = all)
        top_k: Results per collection
        config: Configuration
        
    Returns:
        Aggregated search results
    """
    if not collections:
        collections = [COLLECTION_NAME, CODE_COLLECTION, INSIGHTS_COLLECTION, LESSONS_COLLECTION]
    
    if not config:
        return "Error: Configuration required"
    
    try:
        # Search all collections
        all_results = await semantic_search_engine.multi_collection_search(
            query, collections, top_k, config
        )
        
        # Format aggregated results
        formatted_sections = []
        
        collection_titles = {
            COLLECTION_NAME: "📚 Documentation",
            CODE_COLLECTION: "💻 Code",
            INSIGHTS_COLLECTION: "💡 Insights",
            LESSONS_COLLECTION: "🎓 Lessons Learned"
        }
        
        for collection, results in all_results.items():
            if results:
                title = collection_titles.get(collection, collection)
                section = f"\n=== {title} ===\n"
                
                for i, result in enumerate(results[:top_k]):
                    section += f"\n{i+1}. (Score: {result['combined_score']:.3f}) "
                    section += result['content'][:200] + "...\n"
                
                formatted_sections.append(section)
        
        if not formatted_sections:
            return "No results found across any collection."
        
        return "\n".join(formatted_sections)
        
    except Exception as e:
        logger.error(f"Multi-collection search failed: {e}", exc_info=True)
        return f"Error: Multi-collection search failed: {str(e)}"

# --- Helper Functions ---

def _detect_query_intent(query: str) -> str:
    """Detect the intent of the search query"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['code', 'function', 'class', 'implement']):
        return 'code'
    elif any(word in query_lower for word in ['error', 'fix', 'issue', 'problem']):
        return 'troubleshoot'
    elif any(word in query_lower for word in ['how', 'what', 'explain', 'why']):
        return 'concept'
    elif any(word in query_lower for word in ['design', 'architecture', 'pattern']):
        return 'architecture'
    else:
        return 'general'

def _detect_domain(query: str) -> str:
    """Detect the domain of the search query"""
    domains = {
        'quantum': ['quantum', 'qubit', 'qgan', 'quantum circuit'],
        'ml': ['machine learning', 'ml', 'neural', 'model', 'training'],
        'infrastructure': ['kubernetes', 'docker', 'deployment', 'service'],
        'data': ['data', 'database', 'storage', 'query'],
        'api': ['api', 'endpoint', 'rest', 'graphql']
    }
    
    query_lower = query.lower()
    for domain, keywords in domains.items():
        if any(keyword in query_lower for keyword in keywords):
            return domain
    
    return 'general'

# --- Tool Registration ---
search_knowledge_base_tool = Tool(
    name="search_knowledge_base",
    description="Advanced semantic search across the knowledge base with query optimization and reranking.",
    func=search_knowledge_base
)

search_codebase_tool = Tool(
    name="search_codebase", 
    description="Search the codebase with code-specific optimizations and formatting.",
    func=search_codebase
) 