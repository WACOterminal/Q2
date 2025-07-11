"""
Semantic Search API

This API provides endpoints for enhanced semantic search capabilities:
- Advanced search operations (semantic, keyword, hybrid, personalized)
- Search index management
- Document indexing and updates
- User personalization and preferences
- Search analytics and insights
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from ..core.semantic_search_service import (
    SemanticSearchService,
    semantic_search_service,
    SearchQuery,
    SearchDocument,
    SearchType,
    SearchDomain,
    RelevanceModel,
    IndexStatus
)
from ..core.auth_service import auth_service
from ..core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Initialize components
router = APIRouter(prefix="/api/v1/semantic-search", tags=["Semantic Search"])
security = HTTPBearer()
rate_limiter = RateLimiter()

# ===== REQUEST/RESPONSE MODELS =====

class SearchRequest(BaseModel):
    """Request model for search operations"""
    query: str = Field(..., description="Search query text")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search to perform")
    domain: SearchDomain = Field(default=SearchDomain.GENERAL, description="Search domain")
    limit: int = Field(default=10, description="Maximum number of results")
    offset: int = Field(default=0, description="Offset for pagination")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    context: Dict[str, Any] = Field(default_factory=dict, description="Search context")
    relevance_model: RelevanceModel = Field(default=RelevanceModel.HYBRID, description="Relevance scoring model")

class SearchResultResponse(BaseModel):
    """Response model for search result"""
    document_id: str
    title: str
    content: str
    snippet: str
    score: float
    rank: int
    metadata: Dict[str, Any]
    domain: str
    relevance_factors: Dict[str, float]

class SearchResponse(BaseModel):
    """Response model for search operations"""
    query_id: str
    results: List[SearchResultResponse]
    total_count: int
    search_time: float
    query_suggestions: List[str]
    facets: Dict[str, List[List[Any]]]

class IndexRequest(BaseModel):
    """Request model for creating search index"""
    name: str = Field(..., description="Index name")
    domain: SearchDomain = Field(..., description="Search domain")
    config: Dict[str, Any] = Field(default_factory=dict, description="Index configuration")

class IndexResponse(BaseModel):
    """Response model for index operations"""
    index_id: str
    name: str
    domain: str
    status: str
    document_count: int
    created_at: datetime
    updated_at: datetime

class DocumentRequest(BaseModel):
    """Request model for adding documents"""
    document_id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    domain: SearchDomain = Field(..., description="Document domain")
    keywords: List[str] = Field(default_factory=list, description="Document keywords")

class DocumentResponse(BaseModel):
    """Response model for document operations"""
    document_id: str
    success: bool
    message: str
    timestamp: datetime

class UserPreferencesRequest(BaseModel):
    """Request model for user preferences"""
    preferences: Dict[str, Any] = Field(..., description="User preferences")
    domain_preferences: Dict[str, float] = Field(default_factory=dict, description="Domain preferences")

class UserPreferencesResponse(BaseModel):
    """Response model for user preferences"""
    user_id: str
    preferences: Dict[str, Any]
    domain_preferences: Dict[str, float]
    updated_at: datetime

class AnalyticsResponse(BaseModel):
    """Response model for search analytics"""
    total_queries: int
    average_response_time: float
    popular_queries: Dict[str, int]
    domain_distribution: Dict[str, int]
    total_documents: int
    total_indexes: int
    cache_size: int

# ===== DEPENDENCY INJECTION =====

def get_semantic_search_service() -> SemanticSearchService:
    """Get the semantic search service instance"""
    return semantic_search_service

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    return auth_service.verify_token(credentials.credentials)

# ===== SEARCH OPERATIONS =====

@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Perform semantic search
    
    Performs advanced semantic search with support for different search types,
    relevance models, and personalization.
    """
    try:
        # Rate limiting
        await rate_limiter.check_rate_limit(f"search:{user['user_id']}", max_requests=100, window_seconds=60)
        
        # Create search query
        query = SearchQuery(
            query_id="",  # Will be generated
            text=request.query,
            search_type=request.search_type,
            domain=request.domain,
            user_id=user["user_id"],
            context=request.context,
            filters=request.filters,
            limit=request.limit,
            offset=request.offset,
            relevance_model=request.relevance_model
        )
        
        # Perform search
        response = await service.search(query)
        
        # Convert to API response format
        results = []
        for result in response.results:
            results.append(SearchResultResponse(
                document_id=result.document_id,
                title=result.title,
                content=result.content,
                snippet=result.snippet,
                score=result.score,
                rank=result.rank,
                metadata=result.metadata,
                domain=result.domain.value,
                relevance_factors=result.relevance_factors
            ))
        
        return SearchResponse(
            query_id=response.query_id,
            results=results,
            total_count=response.total_count,
            search_time=response.search_time,
            query_suggestions=response.query_suggestions,
            facets=response.facets
        )
        
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/suggestions")
async def get_search_suggestions(
    query: str = Query(..., description="Partial query for suggestions"),
    limit: int = Query(5, description="Number of suggestions to return"),
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Get search query suggestions
    
    Returns suggested queries based on popular searches and user history.
    """
    try:
        await rate_limiter.check_rate_limit(f"suggestions:{user['user_id']}", max_requests=50, window_seconds=60)
        
        # Create dummy query for suggestions
        dummy_query = SearchQuery(
            query_id="",
            text=query,
            search_type=SearchType.HYBRID,
            domain=SearchDomain.GENERAL,
            user_id=user["user_id"]
        )
        
        # Get suggestions
        suggestions = await service._generate_query_suggestions(dummy_query)
        
        return {
            "query": query,
            "suggestions": suggestions[:limit],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting search suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== INDEX MANAGEMENT =====

@router.post("/indexes", response_model=IndexResponse)
async def create_index(
    request: IndexRequest,
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Create a new search index
    
    Creates a new search index for a specific domain with custom configuration.
    """
    try:
        await rate_limiter.check_rate_limit(f"create_index:{user['user_id']}", max_requests=10, window_seconds=3600)
        
        # Create index
        index_id = await service.create_index(
            name=request.name,
            domain=request.domain,
            config=request.config
        )
        
        # Get index details
        index = service.indexes[index_id]
        
        logger.info(f"Created search index: {index_id} by user {user['user_id']}")
        
        return IndexResponse(
            index_id=index_id,
            name=index.name,
            domain=index.domain.value,
            status=index.status.value,
            document_count=index.document_count,
            created_at=index.created_at,
            updated_at=index.updated_at
        )
        
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indexes", response_model=List[IndexResponse])
async def list_indexes(
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    List all search indexes
    
    Returns a list of all available search indexes.
    """
    try:
        await rate_limiter.check_rate_limit(f"list_indexes:{user['user_id']}", max_requests=20, window_seconds=60)
        
        indexes = []
        for index in service.indexes.values():
            indexes.append(IndexResponse(
                index_id=index.index_id,
                name=index.name,
                domain=index.domain.value,
                status=index.status.value,
                document_count=index.document_count,
                created_at=index.created_at,
                updated_at=index.updated_at
            ))
        
        return indexes
        
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indexes/{index_id}", response_model=IndexResponse)
async def get_index(
    index_id: str,
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Get search index details
    
    Returns detailed information about a specific search index.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_index:{user['user_id']}", max_requests=30, window_seconds=60)
        
        if index_id not in service.indexes:
            raise HTTPException(status_code=404, detail="Index not found")
        
        index = service.indexes[index_id]
        
        return IndexResponse(
            index_id=index.index_id,
            name=index.name,
            domain=index.domain.value,
            status=index.status.value,
            document_count=index.document_count,
            created_at=index.created_at,
            updated_at=index.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== DOCUMENT MANAGEMENT =====

@router.post("/indexes/{index_id}/documents", response_model=DocumentResponse)
async def add_document(
    index_id: str,
    request: DocumentRequest,
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Add a document to a search index
    
    Adds a new document to the specified search index with automatic
    embedding generation and keyword extraction.
    """
    try:
        await rate_limiter.check_rate_limit(f"add_document:{user['user_id']}", max_requests=100, window_seconds=60)
        
        # Create search document
        document = SearchDocument(
            document_id=request.document_id,
            title=request.title,
            content=request.content,
            metadata={
                **request.metadata,
                "added_by": user["user_id"],
                "added_at": datetime.utcnow().isoformat()
            },
            domain=request.domain,
            keywords=request.keywords if request.keywords else None
        )
        
        # Add to index
        success = await service.add_document(index_id, document)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to add document")
        
        logger.info(f"Added document {request.document_id} to index {index_id} by user {user['user_id']}")
        
        return DocumentResponse(
            document_id=request.document_id,
            success=True,
            message="Document added successfully",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/indexes/{index_id}/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    index_id: str,
    document_id: str,
    request: DocumentRequest,
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Update a document in a search index
    
    Updates an existing document in the specified search index.
    """
    try:
        await rate_limiter.check_rate_limit(f"update_document:{user['user_id']}", max_requests=100, window_seconds=60)
        
        # Create updated document
        document = SearchDocument(
            document_id=document_id,
            title=request.title,
            content=request.content,
            metadata={
                **request.metadata,
                "updated_by": user["user_id"],
                "updated_at": datetime.utcnow().isoformat()
            },
            domain=request.domain,
            keywords=request.keywords if request.keywords else None
        )
        
        # Update in index
        success = await service.update_document(index_id, document)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update document")
        
        logger.info(f"Updated document {document_id} in index {index_id} by user {user['user_id']}")
        
        return DocumentResponse(
            document_id=document_id,
            success=True,
            message="Document updated successfully",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/indexes/{index_id}/documents/{document_id}", response_model=DocumentResponse)
async def remove_document(
    index_id: str,
    document_id: str,
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Remove a document from a search index
    
    Removes a document from the specified search index.
    """
    try:
        await rate_limiter.check_rate_limit(f"remove_document:{user['user_id']}", max_requests=100, window_seconds=60)
        
        # Remove from index
        success = await service.remove_document(index_id, document_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to remove document")
        
        logger.info(f"Removed document {document_id} from index {index_id} by user {user['user_id']}")
        
        return DocumentResponse(
            document_id=document_id,
            success=True,
            message="Document removed successfully",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== USER PERSONALIZATION =====

@router.get("/users/{user_id}/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    user_id: str,
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Get user search preferences
    
    Returns the search preferences and personalization settings for a user.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_preferences:{user['user_id']}", max_requests=20, window_seconds=60)
        
        # Check if user can access these preferences
        if user_id != user["user_id"] and not user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get user profile
        profile = await service.get_user_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return UserPreferencesResponse(
            user_id=user_id,
            preferences=profile.preferences,
            domain_preferences={k.value: v for k, v in profile.domain_preferences.items()},
            updated_at=profile.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/users/{user_id}/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(
    user_id: str,
    request: UserPreferencesRequest,
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Update user search preferences
    
    Updates the search preferences and personalization settings for a user.
    """
    try:
        await rate_limiter.check_rate_limit(f"update_preferences:{user['user_id']}", max_requests=10, window_seconds=60)
        
        # Check if user can update these preferences
        if user_id != user["user_id"] and not user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update preferences
        success = await service.update_user_preferences(user_id, request.preferences)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update preferences")
        
        # Get updated profile
        profile = await service.get_user_profile(user_id)
        
        logger.info(f"Updated preferences for user {user_id}")
        
        return UserPreferencesResponse(
            user_id=user_id,
            preferences=profile.preferences,
            domain_preferences={k.value: v for k, v in profile.domain_preferences.items()},
            updated_at=profile.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/search-history")
async def get_user_search_history(
    user_id: str,
    limit: int = Query(50, description="Number of search history items to return"),
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Get user search history
    
    Returns the search history for a user.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_history:{user['user_id']}", max_requests=20, window_seconds=60)
        
        # Check if user can access this history
        if user_id != user["user_id"] and not user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get user profile
        profile = await service.get_user_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return {
            "user_id": user_id,
            "search_history": profile.search_history[-limit:],
            "interaction_history": profile.interaction_history[-limit:],
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user search history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ANALYTICS =====

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_search_analytics(
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Get search analytics
    
    Returns comprehensive analytics about search usage and performance.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_analytics:{user['user_id']}", max_requests=10, window_seconds=60)
        
        # Get analytics data
        analytics = await service.get_search_analytics()
        
        return AnalyticsResponse(
            total_queries=analytics["total_queries"],
            average_response_time=analytics["average_response_time"],
            popular_queries=analytics["popular_queries"],
            domain_distribution=analytics["domain_distribution"],
            total_documents=analytics["total_documents"],
            total_indexes=analytics["total_indexes"],
            cache_size=analytics["cache_size"]
        )
        
    except Exception as e:
        logger.error(f"Error getting search analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/trends")
async def get_search_trends(
    days: int = Query(7, description="Number of days to analyze"),
    user=Depends(get_current_user),
    service: SemanticSearchService = Depends(get_semantic_search_service)
):
    """
    Get search trends
    
    Returns search trends and patterns over time.
    """
    try:
        await rate_limiter.check_rate_limit(f"get_trends:{user['user_id']}", max_requests=10, window_seconds=60)
        
        # Get analytics data
        analytics = await service.get_search_analytics()
        
        # Generate trends (simplified)
        trends = {
            "query_volume": analytics["total_queries"],
            "popular_domains": analytics["domain_distribution"],
            "top_queries": analytics["popular_queries"],
            "performance_metrics": {
                "average_response_time": analytics["average_response_time"],
                "cache_hit_rate": 0.85,  # Simulated
                "success_rate": 0.95  # Simulated
            },
            "analyzed_period": f"{days} days",
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return trends
        
    except Exception as e:
        logger.error(f"Error getting search trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== HEALTH AND STATUS =====

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns the health status of the semantic search service.
    """
    try:
        # Check service health
        analytics = await semantic_search_service.get_search_analytics()
        
        return {
            "status": "healthy",
            "indexes": len(semantic_search_service.indexes),
            "documents": analytics["total_documents"],
            "cache_size": analytics["cache_size"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {"status": "unhealthy", "message": str(e)}

# ===== INITIALIZATION =====

async def initialize_semantic_search_api():
    """Initialize the semantic search API"""
    
    try:
        await semantic_search_service.initialize()
        logger.info("Semantic search API initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing semantic search API: {e}")
        raise

async def shutdown_semantic_search_api():
    """Shutdown the semantic search API"""
    
    try:
        await semantic_search_service.shutdown()
        logger.info("Semantic search API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error shutting down semantic search API: {e}") 