# H2M/app/services/feedback_processor.py
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, timezone
import asyncio
import json
from collections import defaultdict, Counter
import numpy as np
from textblob import TextBlob

from shared.q_feedback_schemas.models import (
    ExplicitFeedback, ImplicitFeedback, FeedbackAggregation,
    FeedbackPattern, FeedbackAction, UserPreference,
    FeedbackLoop, FeedbackType, FeedbackSentiment, FeedbackContext
)
from pyignite import Client
from pyignite.exceptions import CacheError
import pulsar
import httpx

logger = logging.getLogger(__name__)


class FeedbackProcessor:
    """
    Processes both explicit and implicit feedback to improve system behavior.
    
    Features:
    - Real-time feedback processing
    - Pattern detection and analysis
    - Automated action triggering
    - User preference learning
    - Feedback loop tracking
    """
    
    def __init__(
        self,
        ignite_addresses: List[str],
        pulsar_url: str,
        knowledge_graph_url: str
    ):
        self.ignite_client = Client()
        self.ignite_addresses = ignite_addresses
        self.pulsar_url = pulsar_url
        self.knowledge_graph_url = knowledge_graph_url
        
        # Caches
        self.feedback_cache = None
        self.aggregation_cache = None
        self.preference_cache = None
        self.pattern_cache = None
        
        # Pulsar
        self.pulsar_client = None
        self.feedback_consumer = None
        self.action_producer = None
        
        # Pattern detection thresholds
        self.NEGATIVE_FEEDBACK_THRESHOLD = 0.3  # 30% negative feedback triggers pattern
        self.LOW_RATING_THRESHOLD = 2.5  # Average rating below this triggers action
        self.ABANDONMENT_THRESHOLD = 0.5  # 50% abandonment rate is concerning
        
    def connect(self):
        """Initialize connections to storage and messaging systems"""
        try:
            # Connect to Ignite
            self.ignite_client.connect(self.ignite_addresses)
            self.feedback_cache = self.ignite_client.get_or_create_cache("feedback_cache")
            self.aggregation_cache = self.ignite_client.get_or_create_cache("feedback_aggregations")
            self.preference_cache = self.ignite_client.get_or_create_cache("user_preferences")
            self.pattern_cache = self.ignite_client.get_or_create_cache("feedback_patterns")
            
            # Connect to Pulsar
            self.pulsar_client = pulsar.Client(self.pulsar_url)
            self.feedback_consumer = self.pulsar_client.subscribe(
                "persistent://public/default/q.feedback.events",
                "feedback-processor"
            )
            self.action_producer = self.pulsar_client.create_producer(
                "persistent://public/default/q.feedback.actions"
            )
            
            logger.info("Feedback processor connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect feedback processor: {e}")
            raise
            
    async def process_explicit_feedback(self, feedback: ExplicitFeedback):
        """Process explicit user feedback"""
        try:
            # Analyze sentiment if comment provided
            if feedback.comment:
                sentiment_result = self._analyze_sentiment(feedback.comment)
                feedback.sentiment = sentiment_result[0]
                feedback.sentiment_score = sentiment_result[1]
                
            # Store feedback
            cache_key = f"{feedback.user_id}:{feedback.feedback_id}"
            self.feedback_cache.put(cache_key, feedback.dict())
            
            # Update user preferences if this is a preference-type feedback
            if feedback.type == FeedbackType.PREFERENCE:
                await self._update_user_preferences(feedback)
                
            # Check for correction patterns
            if feedback.type == FeedbackType.CORRECTION:
                await self._process_correction(feedback)
                
            # Update aggregations
            await self._update_aggregations(feedback)
            
            # Check for action triggers
            await self._check_feedback_triggers(feedback)
            
            logger.info(f"Processed explicit feedback: {feedback.feedback_id}")
            
        except Exception as e:
            logger.error(f"Failed to process explicit feedback: {e}")
            raise
            
    async def process_implicit_feedback(self, feedback: ImplicitFeedback):
        """Process implicit user behavior feedback"""
        try:
            # Store feedback
            cache_key = f"{feedback.user_id}:{feedback.feedback_id}"
            self.feedback_cache.put(cache_key, feedback.dict())
            
            # Analyze behavioral patterns
            if feedback.action == "abandon":
                await self._process_abandonment(feedback)
            elif feedback.action == "retry":
                await self._process_retry(feedback)
            elif feedback.action in ["copy", "share", "bookmark"]:
                await self._process_positive_action(feedback)
                
            # Update aggregations
            await self._update_aggregations(feedback)
            
            # Learn from interaction patterns
            await self._learn_from_behavior(feedback)
            
            logger.info(f"Processed implicit feedback: {feedback.feedback_id}")
            
        except Exception as e:
            logger.error(f"Failed to process implicit feedback: {e}")
            raise
            
    def _analyze_sentiment(self, text: str) -> Tuple[FeedbackSentiment, float]:
        """Analyze sentiment of feedback text"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.3:
            sentiment = FeedbackSentiment.POSITIVE
        elif polarity < -0.3:
            sentiment = FeedbackSentiment.NEGATIVE
        elif -0.1 <= polarity <= 0.1:
            sentiment = FeedbackSentiment.NEUTRAL
        else:
            sentiment = FeedbackSentiment.MIXED
            
        return sentiment, polarity
        
    async def _update_aggregations(self, feedback: Union[ExplicitFeedback, ImplicitFeedback]):
        """Update feedback aggregations"""
        # Determine aggregation periods
        now = datetime.now(timezone.utc)
        hour_key = now.strftime("%Y%m%d%H")
        day_key = now.strftime("%Y%m%d")
        week_key = now.strftime("%Y%W")
        
        for period, key in [("hourly", hour_key), ("daily", day_key), ("weekly", week_key)]:
            agg_key = f"{feedback.target_id}:{feedback.target_type}:{period}:{key}"
            
            # Get or create aggregation
            agg_dict = self.aggregation_cache.get(agg_key)
            if agg_dict:
                agg = FeedbackAggregation(**agg_dict)
            else:
                agg = FeedbackAggregation(
                    target_id=feedback.target_id,
                    target_type=feedback.target_type,
                    aggregation_period=period,
                    period_start=now.isoformat(),
                    period_end=(now + timedelta(hours=1 if period == "hourly" else days=1 if period == "daily" else days=7)).isoformat()
                )
                
            # Update counts
            agg.total_feedback_count += 1
            
            if isinstance(feedback, ExplicitFeedback):
                agg.explicit_feedback_count += 1
                
                # Update ratings
                if feedback.rating:
                    if agg.average_rating:
                        agg.average_rating = (agg.average_rating * (agg.explicit_feedback_count - 1) + feedback.rating) / agg.explicit_feedback_count
                    else:
                        agg.average_rating = feedback.rating
                        
                    rating_int = int(feedback.rating)
                    agg.rating_distribution[rating_int] = agg.rating_distribution.get(rating_int, 0) + 1
                    
                # Update sentiment
                if feedback.sentiment:
                    agg.sentiment_distribution[feedback.sentiment.value] = agg.sentiment_distribution.get(feedback.sentiment.value, 0) + 1
                    
                    if feedback.sentiment_score is not None:
                        if agg.average_sentiment_score:
                            agg.average_sentiment_score = (agg.average_sentiment_score * (agg.explicit_feedback_count - 1) + feedback.sentiment_score) / agg.explicit_feedback_count
                        else:
                            agg.average_sentiment_score = feedback.sentiment_score
                            
            else:  # Implicit feedback
                agg.implicit_feedback_count += 1
                
                # Update behavioral metrics
                if feedback.dwell_time:
                    if agg.average_dwell_time:
                        agg.average_dwell_time = (agg.average_dwell_time * (agg.implicit_feedback_count - 1) + feedback.dwell_time) / agg.implicit_feedback_count
                    else:
                        agg.average_dwell_time = feedback.dwell_time
                        
                # Calculate rates
                if feedback.action == "abandon":
                    agg.abandonment_rate = (agg.abandonment_rate or 0) + (1 / agg.implicit_feedback_count)
                    
            # Save updated aggregation
            self.aggregation_cache.put(agg_key, agg.dict())
            
    async def _check_feedback_triggers(self, feedback: ExplicitFeedback):
        """Check if feedback should trigger immediate action"""
        # Check for low rating
        if feedback.rating and feedback.rating <= 2:
            await self._trigger_low_rating_action(feedback)
            
        # Check for negative sentiment
        if feedback.sentiment == FeedbackSentiment.NEGATIVE:
            await self._check_negative_pattern(feedback)
            
        # Check for repeated issues
        await self._check_repeated_issues(feedback)
        
    async def _trigger_low_rating_action(self, feedback: ExplicitFeedback):
        """Trigger action for low rating"""
        action = FeedbackAction(
            feedback_ids=[feedback.feedback_id],
            action_type="alert",
            description=f"Low rating ({feedback.rating}) received for {feedback.target_type} {feedback.target_id}",
            target_component=feedback.target_type,
            changes_made={
                "alert_sent": True,
                "priority": "high" if feedback.rating == 1 else "medium"
            }
        )
        
        # Send action to queue
        self.action_producer.send(json.dumps(action.dict()).encode('utf-8'))
        
        # Store action
        self.pattern_cache.put(f"action:{action.action_id}", action.dict())
        
    async def _check_negative_pattern(self, feedback: ExplicitFeedback):
        """Check if negative feedback forms a pattern"""
        # Get recent feedback for same target
        day_key = datetime.now(timezone.utc).strftime("%Y%m%d")
        agg_key = f"{feedback.target_id}:{feedback.target_type}:daily:{day_key}"
        
        agg_dict = self.aggregation_cache.get(agg_key)
        if agg_dict:
            agg = FeedbackAggregation(**agg_dict)
            
            negative_count = agg.sentiment_distribution.get(FeedbackSentiment.NEGATIVE.value, 0)
            negative_rate = negative_count / agg.total_feedback_count if agg.total_feedback_count > 0 else 0
            
            if negative_rate >= self.NEGATIVE_FEEDBACK_THRESHOLD and agg.total_feedback_count >= 10:
                # Create pattern
                pattern = FeedbackPattern(
                    pattern_type="recurring_issue",
                    description=f"High negative feedback rate ({negative_rate:.1%}) for {feedback.target_type} {feedback.target_id}",
                    affected_targets=[feedback.target_id],
                    time_range={"start": agg.period_start, "end": agg.period_end},
                    frequency=negative_count,
                    confidence=min(0.9, agg.total_feedback_count / 50),  # Higher confidence with more data
                    impact_score=negative_rate,
                    example_feedback_ids=[feedback.feedback_id],
                    metrics={
                        "negative_rate": negative_rate,
                        "total_feedback": agg.total_feedback_count,
                        "average_rating": agg.average_rating
                    }
                )
                
                # Store pattern
                self.pattern_cache.put(f"pattern:{pattern.pattern_id}", pattern.dict())
                
                # Trigger remediation action
                await self._trigger_pattern_action(pattern)
                
    async def _trigger_pattern_action(self, pattern: FeedbackPattern):
        """Trigger action based on detected pattern"""
        action_type = "agent_retrain" if "agent" in pattern.affected_targets[0] else "workflow_update"
        
        action = FeedbackAction(
            pattern_id=pattern.pattern_id,
            feedback_ids=pattern.example_feedback_ids,
            action_type=action_type,
            description=f"Pattern detected: {pattern.description}",
            target_component=pattern.affected_targets[0],
            changes_made={
                "pattern_type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "impact_score": pattern.impact_score
            },
            expected_impact="Reduce negative feedback rate by 20%"
        )
        
        # Send action
        self.action_producer.send(json.dumps(action.dict()).encode('utf-8'))
        
        # Create feedback loop
        loop = FeedbackLoop(
            trigger_type="pattern",
            trigger_details={"pattern_id": pattern.pattern_id},
            feedback_count=len(pattern.example_feedback_ids),
            feedback_time_range=pattern.time_range,
            patterns_identified=[pattern.pattern_id],
            actions_taken=[action.action_id],
            metrics_before=pattern.metrics
        )
        
        self.pattern_cache.put(f"loop:{loop.loop_id}", loop.dict())
        
    async def _update_user_preferences(self, feedback: ExplicitFeedback):
        """Update user preferences based on feedback"""
        if feedback.type != FeedbackType.PREFERENCE:
            return
            
        # Extract preference from feedback
        pref_key = f"{feedback.user_id}:{feedback.context.value}:{feedback.target_type}"
        
        pref_dict = self.preference_cache.get(pref_key)
        if pref_dict:
            pref = UserPreference(**pref_dict)
            # Update confidence based on consistency
            pref.confidence = min(1.0, pref.confidence + 0.1)
            pref.source_feedback_count += 1
        else:
            pref = UserPreference(
                user_id=feedback.user_id,
                preference_type=feedback.target_type,
                preference_key=feedback.context.value,
                preference_value=feedback.target_id,
                confidence=0.6,
                source_feedback_count=1,
                applicable_contexts=[feedback.context.value]
            )
            
        pref.last_updated = datetime.now(timezone.utc).isoformat()
        
        # Save preference
        self.preference_cache.put(pref_key, pref.dict())
        
        logger.info(f"Updated user preference: {pref_key}")
        
    async def _process_correction(self, feedback: ExplicitFeedback):
        """Process user corrections"""
        if not feedback.corrected_content:
            return
            
        # Store correction pattern
        correction_key = f"correction:{feedback.target_type}:{feedback.target_id}"
        corrections = self.pattern_cache.get(correction_key) or []
        
        corrections.append({
            "feedback_id": feedback.feedback_id,
            "original": feedback.original_content,
            "corrected": feedback.corrected_content,
            "timestamp": feedback.timestamp
        })
        
        self.pattern_cache.put(correction_key, corrections)
        
        # If multiple similar corrections, trigger action
        if len(corrections) >= 3:
            action = FeedbackAction(
                feedback_ids=[c["feedback_id"] for c in corrections],
                action_type="agent_retrain",
                description=f"Multiple corrections received for {feedback.target_type} {feedback.target_id}",
                target_component=feedback.target_type,
                changes_made={
                    "corrections": corrections,
                    "correction_count": len(corrections)
                }
            )
            
            self.action_producer.send(json.dumps(action.dict()).encode('utf-8'))
            
    async def _process_abandonment(self, feedback: ImplicitFeedback):
        """Process abandonment feedback"""
        # Check abandonment rate for target
        hour_key = datetime.now(timezone.utc).strftime("%Y%m%d%H")
        agg_key = f"{feedback.target_id}:{feedback.target_type}:hourly:{hour_key}"
        
        agg_dict = self.aggregation_cache.get(agg_key)
        if agg_dict:
            agg = FeedbackAggregation(**agg_dict)
            
            if agg.abandonment_rate and agg.abandonment_rate >= self.ABANDONMENT_THRESHOLD:
                # High abandonment rate - trigger investigation
                action = FeedbackAction(
                    feedback_ids=[feedback.feedback_id],
                    action_type="alert",
                    description=f"High abandonment rate ({agg.abandonment_rate:.1%}) for {feedback.target_type} {feedback.target_id}",
                    target_component=feedback.target_type,
                    changes_made={
                        "abandonment_rate": agg.abandonment_rate,
                        "sample_size": agg.implicit_feedback_count
                    }
                )
                
                self.action_producer.send(json.dumps(action.dict()).encode('utf-8'))
                
    async def _process_retry(self, feedback: ImplicitFeedback):
        """Process retry feedback indicating potential issues"""
        retry_count = feedback.action_data.get("retry_count", 1)
        
        if retry_count >= 3:
            # Multiple retries indicate frustration
            action = FeedbackAction(
                feedback_ids=[feedback.feedback_id],
                action_type="ui_change",
                description=f"User retried {retry_count} times on {feedback.target_type} {feedback.target_id}",
                target_component=feedback.target_type,
                changes_made={
                    "suggested_change": "Improve clarity or error messages",
                    "retry_count": retry_count
                }
            )
            
            self.action_producer.send(json.dumps(action.dict()).encode('utf-8'))
            
    async def _process_positive_action(self, feedback: ImplicitFeedback):
        """Process positive implicit actions"""
        # Update user preferences based on positive actions
        pref_key = f"{feedback.user_id}:positive_interaction:{feedback.target_type}"
        
        pref_dict = self.preference_cache.get(pref_key)
        if pref_dict:
            pref = UserPreference(**pref_dict)
            pref.confidence = min(1.0, pref.confidence + 0.05)
            pref.source_feedback_count += 1
        else:
            pref = UserPreference(
                user_id=feedback.user_id,
                preference_type="interaction_style",
                preference_key=feedback.action,
                preference_value=feedback.target_id,
                confidence=0.5,
                source_feedback_count=1,
                applicable_contexts=[feedback.context.value]
            )
            
        self.preference_cache.put(pref_key, pref.dict())
        
    async def _learn_from_behavior(self, feedback: ImplicitFeedback):
        """Learn patterns from user behavior"""
        # Analyze dwell time patterns
        if feedback.dwell_time:
            # Short dwell time might indicate confusion or disinterest
            if feedback.dwell_time < 5:  # Less than 5 seconds
                await self._check_usability_issue(feedback)
            # Long dwell time might indicate engagement
            elif feedback.dwell_time > 60:  # More than 1 minute
                await self._mark_engaging_content(feedback)
                
    async def _check_usability_issue(self, feedback: ImplicitFeedback):
        """Check for potential usability issues"""
        # Get recent similar feedback
        recent_key = f"usability:{feedback.target_type}:{feedback.target_id}"
        recent_issues = self.pattern_cache.get(recent_key) or []
        
        recent_issues.append({
            "feedback_id": feedback.feedback_id,
            "dwell_time": feedback.dwell_time,
            "timestamp": feedback.timestamp
        })
        
        # Keep only recent issues (last hour)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        recent_issues = [i for i in recent_issues if i["timestamp"] > cutoff]
        
        self.pattern_cache.put(recent_key, recent_issues)
        
        # If pattern emerges, create action
        if len(recent_issues) >= 5:
            avg_dwell = np.mean([i["dwell_time"] for i in recent_issues])
            
            pattern = FeedbackPattern(
                pattern_type="usability_issue",
                description=f"Low engagement detected - average dwell time {avg_dwell:.1f}s",
                affected_targets=[feedback.target_id],
                time_range={
                    "start": min(i["timestamp"] for i in recent_issues),
                    "end": max(i["timestamp"] for i in recent_issues)
                },
                frequency=len(recent_issues),
                confidence=0.7,
                impact_score=0.5,
                example_feedback_ids=[i["feedback_id"] for i in recent_issues],
                metrics={"avg_dwell_time": avg_dwell}
            )
            
            self.pattern_cache.put(f"pattern:{pattern.pattern_id}", pattern.dict())
            
    async def _mark_engaging_content(self, feedback: ImplicitFeedback):
        """Mark content as engaging based on behavior"""
        # Store positive signal
        engagement_key = f"engagement:{feedback.target_type}:{feedback.target_id}"
        engagement_score = self.pattern_cache.get(engagement_key) or 0
        
        # Increment engagement score
        engagement_score += 1
        self.pattern_cache.put(engagement_key, engagement_score)
        
        # Send to knowledge graph if highly engaging
        if engagement_score >= 10:
            await self._publish_to_knowledge_graph({
                "type": "high_engagement_content",
                "target": feedback.target_id,
                "target_type": feedback.target_type,
                "engagement_score": engagement_score
            })
            
    async def _publish_to_knowledge_graph(self, data: Dict[str, Any]):
        """Publish insights to knowledge graph"""
        try:
            operations = [{
                "operation": "upsert_vertex",
                "label": "FeedbackInsight",
                "id_key": "insight_id",
                "properties": {
                    "insight_id": f"insight_{data['type']}_{data['target']}",
                    "type": data["type"],
                    "target": data["target"],
                    "target_type": data["target_type"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **data
                }
            }]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.knowledge_graph_url}/api/v1/ingest",
                    json={"operations": operations}
                )
                response.raise_for_status()
                
        except Exception as e:
            logger.error(f"Failed to publish to knowledge graph: {e}")
            
    async def process_feedback_stream(self):
        """Main loop to process feedback stream"""
        while True:
            try:
                msg = self.feedback_consumer.receive(timeout_millis=1000)
                feedback_data = json.loads(msg.data().decode('utf-8'))
                
                # Determine feedback type and process
                if feedback_data.get("type") in ["explicit_rating", "explicit_comment", "correction", "preference"]:
                    feedback = ExplicitFeedback(**feedback_data)
                    await self.process_explicit_feedback(feedback)
                else:
                    feedback = ImplicitFeedback(**feedback_data)
                    await self.process_implicit_feedback(feedback)
                    
                self.feedback_consumer.acknowledge(msg)
                
            except Exception as e:
                if "timeout" not in str(e).lower():
                    logger.error(f"Error processing feedback stream: {e}")
                    
    def close(self):
        """Clean up connections"""
        if self.pulsar_client:
            self.pulsar_client.close()
        if self.ignite_client:
            self.ignite_client.close() 