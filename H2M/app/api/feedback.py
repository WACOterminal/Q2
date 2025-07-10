import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import asyncio
import json

from app.services.h2m_pulsar import h2m_pulsar_client
from shared.q_auth_parser.parser import get_current_user
from shared.q_auth_parser.models import UserClaims
from app.h2m_models import FeedbackEvent
from shared.q_feedback_schemas.models import (
    ExplicitFeedback, ImplicitFeedback, FeedbackAggregation,
    FeedbackPattern, UserPreference, FeedbackLoop,
    FeedbackContext, FeedbackType
)
from app.services.feedback_processor import FeedbackProcessor
import pulsar
import httpx

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize feedback processor (in production, this would be dependency injected)
feedback_processor = None
pulsar_producer = None


def get_feedback_processor():
    """Get or initialize feedback processor"""
    global feedback_processor
    if not feedback_processor:
        feedback_processor = FeedbackProcessor(
            ignite_addresses=["ignite:10800"],
            pulsar_url="pulsar://pulsar:6650",
            knowledge_graph_url="http://knowledgegraphq:8006"
        )
        feedback_processor.connect()
    return feedback_processor


def get_pulsar_producer():
    """Get or initialize Pulsar producer"""
    global pulsar_producer
    if not pulsar_producer:
        client = pulsar.Client("pulsar://pulsar:6650")
        pulsar_producer = client.create_producer(
            "persistent://public/default/q.feedback.events"
        )
    return pulsar_producer


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(
    request: FeedbackEvent,
    user: UserClaims = Depends(get_current_user)
):
    """
    Receives feedback from a user and publishes it to a Pulsar topic for later processing.
    """
    logger.info(f"Received feedback from user '{user.username}' for item '{request.reference_id}' in context '{request.context}'. Score: {request.score}")
    
    feedback_data = request.dict()
    feedback_data['user'] = user.dict() # Add user info to the event payload

    try:
        # Send to the dedicated feedback topic for analytics
        await h2m_pulsar_client.send_feedback(feedback_data)
        
        # Also send a platform event if model feedback is present
        if request.model_version:
            platform_event = {
                "event_type": "MODEL_FEEDBACK_RECEIVED",
                "payload": {
                    "model_version": request.model_version,
                    "score": request.score,
                    "context": request.context
                }
            }
            await h2m_pulsar_client.send_platform_event(platform_event)

        return {"status": "Feedback received"}
    except RuntimeError as e:
        logger.error(f"Failed to send feedback to Pulsar: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The feedback processing service is currently unavailable."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred."
        ) 


@router.post("/explicit")
async def submit_explicit_feedback(
    feedback: ExplicitFeedback,
    user: UserClaims = Depends(get_current_user)
):
    """Submit explicit user feedback (rating, comment, correction)"""
    try:
        # Set user ID from auth
        feedback.user_id = user.sub
        
        # Publish to feedback stream
        producer = get_pulsar_producer()
        producer.send(json.dumps(feedback.dict()).encode('utf-8'))
        
        # Process immediately if processor available
        processor = get_feedback_processor()
        await processor.process_explicit_feedback(feedback)
        
        return {
            "status": "success",
            "feedback_id": feedback.feedback_id,
            "message": "Feedback submitted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.post("/implicit")
async def submit_implicit_feedback(
    feedback: ImplicitFeedback,
    user: UserClaims = Depends(get_current_user)
):
    """Submit implicit user behavior feedback"""
    try:
        # Set user ID from auth
        feedback.user_id = user.sub
        
        # Publish to feedback stream
        producer = get_pulsar_producer()
        producer.send(json.dumps(feedback.dict()).encode('utf-8'))
        
        # Process immediately if processor available
        processor = get_feedback_processor()
        await processor.process_implicit_feedback(feedback)
        
        return {
            "status": "success",
            "feedback_id": feedback.feedback_id,
            "message": "Behavior tracked successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track behavior: {str(e)}")


@router.get("/aggregations/{target_id}")
async def get_feedback_aggregations(
    target_id: str,
    target_type: str,
    period: Optional[str] = Query("daily", regex="^(hourly|daily|weekly)$"),
    user: UserClaims = Depends(get_current_user)
):
    """Get aggregated feedback for a specific target"""
    try:
        processor = get_feedback_processor()
        
        # Get current period key
        now = datetime.now(timezone.utc)
        if period == "hourly":
            key = now.strftime("%Y%m%d%H")
        elif period == "daily":
            key = now.strftime("%Y%m%d")
        else:  # weekly
            key = now.strftime("%Y%W")
            
        agg_key = f"{target_id}:{target_type}:{period}:{key}"
        agg_dict = processor.aggregation_cache.get(agg_key)
        
        if not agg_dict:
            return {
                "target_id": target_id,
                "target_type": target_type,
                "period": period,
                "data": None,
                "message": "No feedback data available for this period"
            }
            
        return {
            "target_id": target_id,
            "target_type": target_type,
            "period": period,
            "data": agg_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get aggregations: {str(e)}")


@router.get("/patterns")
async def get_feedback_patterns(
    pattern_type: Optional[str] = None,
    min_confidence: float = Query(0.5, ge=0, le=1),
    limit: int = Query(10, ge=1, le=100),
    user: UserClaims = Depends(get_current_user)
):
    """Get detected feedback patterns"""
    try:
        processor = get_feedback_processor()
        
        # Get all patterns from cache
        patterns = []
        
        # In production, this would use a proper index/query
        # For now, we'll scan the cache
        pattern_keys = processor.pattern_cache.keys()
        
        for key in pattern_keys:
            if key.startswith("pattern:"):
                pattern_dict = processor.pattern_cache.get(key)
                if pattern_dict:
                    pattern = FeedbackPattern(**pattern_dict)
                    
                    # Apply filters
                    if pattern_type and pattern.pattern_type != pattern_type:
                        continue
                    if pattern.confidence < min_confidence:
                        continue
                        
                    patterns.append(pattern_dict)
                    
        # Sort by impact score and limit
        patterns.sort(key=lambda p: p.get("impact_score", 0), reverse=True)
        patterns = patterns[:limit]
        
        return {
            "total": len(patterns),
            "patterns": patterns,
            "filters": {
                "pattern_type": pattern_type,
                "min_confidence": min_confidence
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get patterns: {str(e)}")


@router.get("/preferences/{user_id}")
async def get_user_preferences(
    user_id: str,
    preference_type: Optional[str] = None,
    current_user: UserClaims = Depends(get_current_user)
):
    """Get user preferences learned from feedback"""
    # Check authorization - users can only see their own preferences unless admin
    if user_id != current_user.sub and "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Not authorized to view these preferences")
        
    try:
        processor = get_feedback_processor()
        
        # Get user preferences
        preferences = []
        
        # Scan preference cache for user
        pref_keys = processor.preference_cache.keys()
        
        for key in pref_keys:
            if key.startswith(f"{user_id}:"):
                pref_dict = processor.preference_cache.get(key)
                if pref_dict:
                    pref = UserPreference(**pref_dict)
                    
                    # Apply filter
                    if preference_type and pref.preference_type != preference_type:
                        continue
                        
                    preferences.append(pref_dict)
                    
        # Sort by confidence
        preferences.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        
        return {
            "user_id": user_id,
            "total": len(preferences),
            "preferences": preferences
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")


@router.get("/loops")
async def get_feedback_loops(
    status: Optional[str] = Query(None, regex="^(open|closed|monitoring)$"),
    limit: int = Query(10, ge=1, le=100),
    user: UserClaims = Depends(get_current_user)
):
    """Get active feedback loops"""
    try:
        processor = get_feedback_processor()
        
        # Get feedback loops
        loops = []
        
        # Scan pattern cache for loops
        loop_keys = processor.pattern_cache.keys()
        
        for key in loop_keys:
            if key.startswith("loop:"):
                loop_dict = processor.pattern_cache.get(key)
                if loop_dict:
                    loop = FeedbackLoop(**loop_dict)
                    
                    # Apply filter
                    if status and loop.loop_status != status:
                        continue
                        
                    loops.append(loop_dict)
                    
        # Sort by opened_at (most recent first)
        loops.sort(key=lambda l: l.get("opened_at", ""), reverse=True)
        loops = loops[:limit]
        
        return {
            "total": len(loops),
            "loops": loops,
            "filters": {
                "status": status
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feedback loops: {str(e)}")


@router.post("/loops/{loop_id}/close")
async def close_feedback_loop(
    loop_id: str,
    outcome: str,
    metrics_after: Dict[str, Any],
    user: UserClaims = Depends(get_current_user)
):
    """Close a feedback loop with results"""
    try:
        processor = get_feedback_processor()
        
        # Get the loop
        loop_dict = processor.pattern_cache.get(f"loop:{loop_id}")
        if not loop_dict:
            raise HTTPException(status_code=404, detail="Feedback loop not found")
            
        loop = FeedbackLoop(**loop_dict)
        
        # Update loop
        loop.loop_status = "closed"
        loop.outcome = outcome
        loop.metrics_after = metrics_after
        loop.closed_at = datetime.now(timezone.utc).isoformat()
        
        # Calculate improvement
        if loop.metrics_before and metrics_after:
            # Simple improvement calculation - can be customized
            before_score = loop.metrics_before.get("average_rating", 0) or loop.metrics_before.get("success_rate", 0)
            after_score = metrics_after.get("average_rating", 0) or metrics_after.get("success_rate", 0)
            
            if before_score > 0:
                loop.improvement_percentage = ((after_score - before_score) / before_score) * 100
                
        # Save updated loop
        processor.pattern_cache.put(f"loop:{loop_id}", loop.dict())
        
        return {
            "status": "success",
            "loop_id": loop_id,
            "improvement_percentage": loop.improvement_percentage,
            "message": "Feedback loop closed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to close feedback loop: {str(e)}")


@router.get("/stats/summary")
async def get_feedback_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user: UserClaims = Depends(get_current_user)
):
    """Get summary statistics of feedback"""
    try:
        processor = get_feedback_processor()
        
        # Calculate summary stats
        total_feedback = 0
        total_explicit = 0
        total_implicit = 0
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
        rating_sum = 0
        rating_count = 0
        
        # Get all feedback from cache (in production, use proper querying)
        feedback_keys = processor.feedback_cache.keys()
        
        for key in feedback_keys:
            fb_dict = processor.feedback_cache.get(key)
            if fb_dict:
                # Apply date filter if provided
                if start_date or end_date:
                    fb_time = datetime.fromisoformat(fb_dict.get("timestamp"))
                    if start_date and fb_time < start_date:
                        continue
                    if end_date and fb_time > end_date:
                        continue
                        
                total_feedback += 1
                
                if fb_dict.get("type") in ["explicit_rating", "explicit_comment", "correction", "preference"]:
                    total_explicit += 1
                    
                    if fb_dict.get("sentiment"):
                        sentiment_counts[fb_dict["sentiment"]] += 1
                        
                    if fb_dict.get("rating"):
                        rating_sum += fb_dict["rating"]
                        rating_count += 1
                else:
                    total_implicit += 1
                    
        # Calculate averages
        avg_rating = rating_sum / rating_count if rating_count > 0 else None
        
        # Get active patterns and loops
        active_patterns = sum(1 for k in processor.pattern_cache.keys() if k.startswith("pattern:"))
        active_loops = sum(
            1 for k in processor.pattern_cache.keys() 
            if k.startswith("loop:") and processor.pattern_cache.get(k, {}).get("loop_status") == "open"
        )
        
        return {
            "total_feedback": total_feedback,
            "explicit_feedback": total_explicit,
            "implicit_feedback": total_implicit,
            "average_rating": avg_rating,
            "sentiment_distribution": sentiment_counts,
            "active_patterns": active_patterns,
            "active_loops": active_loops,
            "date_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feedback summary: {str(e)}")


@router.post("/test/generate")
async def generate_test_feedback(
    count: int = Query(10, ge=1, le=100),
    user: UserClaims = Depends(get_current_user)
):
    """Generate test feedback data (development only)"""
    if "admin" not in user.roles:
        raise HTTPException(status_code=403, detail="Admin access required")
        
    try:
        import random
        
        producer = get_pulsar_producer()
        generated = []
        
        for i in range(count):
            if random.random() < 0.5:
                # Generate explicit feedback
                feedback = ExplicitFeedback(
                    user_id=f"test_user_{random.randint(1, 10)}",
                    session_id=f"session_{random.randint(1, 100)}",
                    context=random.choice(list(FeedbackContext)),
                    target_id=f"target_{random.randint(1, 20)}",
                    target_type=random.choice(["agent", "workflow", "ui_element"]),
                    rating=random.uniform(1, 5),
                    comment=random.choice([
                        "Great response!",
                        "Could be better",
                        "Not what I expected",
                        "Perfect, thank you!",
                        "This needs improvement"
                    ])
                )
            else:
                # Generate implicit feedback
                feedback = ImplicitFeedback(
                    user_id=f"test_user_{random.randint(1, 10)}",
                    session_id=f"session_{random.randint(1, 100)}",
                    context=random.choice(list(FeedbackContext)),
                    target_id=f"target_{random.randint(1, 20)}",
                    target_type=random.choice(["agent", "workflow", "ui_element"]),
                    action=random.choice(["click", "scroll", "copy", "abandon", "retry"]),
                    dwell_time=random.uniform(1, 120),
                    time_to_action=random.uniform(0.5, 10)
                )
                
            producer.send(json.dumps(feedback.dict()).encode('utf-8'))
            generated.append(feedback.dict())
            
        return {
            "status": "success",
            "generated": count,
            "samples": generated[:5]  # Return first 5 as samples
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate test data: {str(e)}") 