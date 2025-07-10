"""
Real-Time Collaboration Service

This service provides real-time collaboration capabilities for human-AI interaction:
- WebSocket-based real-time communication
- Shared workspaces with collaborative editing
- Real-time synchronization of changes
- Presence awareness and user tracking
- Conflict resolution for concurrent edits
- Session management and persistence
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import asdict
import uuid
from enum import Enum

# Q Platform imports
from shared.q_collaboration_schemas.models import (
    CollaborationSession, RealTimeUpdate, CollaborationContext,
    CollaborationType, CollaborationStatus
)
from app.services.pulsar_service import PulsarService
from app.services.ignite_service import IgniteService
from app.services.knowledge_graph_service import KnowledgeGraphService

logger = logging.getLogger(__name__)

class UpdateType(Enum):
    """Types of real-time updates"""
    CURSOR_MOVE = "cursor_move"
    TEXT_EDIT = "text_edit"
    SELECTION_CHANGE = "selection_change"
    DOCUMENT_CHANGE = "document_change"
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    STATUS_CHANGE = "status_change"
    DECISION_POINT = "decision_point"
    COMMENT = "comment"
    ANNOTATION = "annotation"
    VOICE_NOTE = "voice_note"
    SCREEN_SHARE = "screen_share"

class PresenceStatus(Enum):
    """User presence status"""
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"

class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    LAST_WRITER_WINS = "last_writer_wins"
    OPERATIONAL_TRANSFORM = "operational_transform"
    MERGE = "merge"
    MANUAL_RESOLUTION = "manual_resolution"

class RealTimeCollaborationService:
    """
    Service for managing real-time collaboration sessions
    """
    
    def __init__(self):
        self.pulsar_service = PulsarService()
        self.ignite_service = IgniteService()
        self.knowledge_graph = KnowledgeGraphService()
        
        # Active sessions and connections
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.websocket_connections: Dict[str, Dict[str, Any]] = {}
        self.user_presence: Dict[str, Dict[str, Any]] = {}
        self.shared_workspaces: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_concurrent_sessions = 1000
        self.session_timeout = 3600  # 1 hour
        self.presence_timeout = 300  # 5 minutes
        self.sync_interval = 1.0  # seconds
        
        # Conflict resolution
        self.conflict_resolution_strategy = ConflictResolution.OPERATIONAL_TRANSFORM
        
    async def initialize(self):
        """Initialize the real-time collaboration service"""
        logger.info("Initializing Real-Time Collaboration Service")
        
        # Setup Pulsar topics for real-time updates
        await self._setup_pulsar_topics()
        
        # Initialize shared workspace storage
        await self._initialize_workspace_storage()
        
        # Start background tasks
        asyncio.create_task(self._presence_cleanup_task())
        asyncio.create_task(self._session_cleanup_task())
        asyncio.create_task(self._sync_task())
        
        logger.info("Real-Time Collaboration Service initialized successfully")
    
    # ===== SESSION MANAGEMENT =====
    
    async def create_collaboration_session(
        self,
        session_id: str,
        collaboration_type: CollaborationType,
        agent_id: str,
        user_id: str,
        context: CollaborationContext,
        **kwargs
    ) -> CollaborationSession:
        """
        Create a new collaboration session
        
        Args:
            session_id: Unique session identifier
            collaboration_type: Type of collaboration
            agent_id: Agent ID
            user_id: User ID
            context: Collaboration context
            **kwargs: Additional session parameters
            
        Returns:
            Created collaboration session
        """
        logger.info(f"Creating collaboration session: {session_id}")
        
        # Create session
        session = CollaborationSession(
            session_id=session_id,
            collaboration_type=collaboration_type,
            agent_id=agent_id,
            user_id=user_id,
            workflow_id=kwargs.get("workflow_id"),
            task_id=kwargs.get("task_id"),
            title=kwargs.get("title", "Collaboration Session"),
            description=kwargs.get("description", ""),
            priority=kwargs.get("priority", 3),
            status=CollaborationStatus.PENDING,
            primary_human=user_id,
            assigned_experts=[],
            participating_agents=[agent_id],
            context_data=asdict(context),
            shared_workspace={},
            decision_points=[],
            started_at=datetime.utcnow(),
            estimated_duration=kwargs.get("estimated_duration", 60),
            actual_duration=None,
            deadline=kwargs.get("deadline"),
            resolution=None,
            decisions_made=[],
            training_data_generated=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=kwargs.get("metadata", {})
        )
        
        # Store session
        self.active_sessions[session_id] = session
        await self._persist_session(session)
        
        # Initialize shared workspace
        await self._initialize_shared_workspace(session_id)
        
        # Publish session creation event
        await self.pulsar_service.publish(
            "q.collaboration.session.created",
            {
                "session_id": session_id,
                "collaboration_type": collaboration_type.value,
                "participants": [agent_id, user_id],
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Collaboration session created: {session_id}")
        return session
    
    async def join_collaboration_session(
        self,
        session_id: str,
        user_id: str,
        websocket_connection: Any = None
    ) -> bool:
        """
        Join an existing collaboration session
        
        Args:
            session_id: Session to join
            user_id: User joining the session
            websocket_connection: WebSocket connection for real-time updates
            
        Returns:
            True if successfully joined, False otherwise
        """
        logger.info(f"User {user_id} joining session: {session_id}")
        
        # Check if session exists
        session = self.active_sessions.get(session_id)
        if not session:
            session = await self._load_session(session_id)
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return False
        
        # Add user to session
        if user_id not in session.participating_agents:
            session.participating_agents.append(user_id)
            session.updated_at = datetime.utcnow()
            await self._persist_session(session)
        
        # Register WebSocket connection
        if websocket_connection:
            await self._register_websocket(session_id, user_id, websocket_connection)
        
        # Update presence
        await self._update_user_presence(user_id, session_id, PresenceStatus.ONLINE)
        
        # Broadcast user join event
        await self._broadcast_update(
            session_id,
            RealTimeUpdate(
                update_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.utcnow(),
                update_type=UpdateType.USER_JOIN.value,
                content={"user_id": user_id},
                sender_id=user_id,
                visibility=[user_id]  # All participants can see
            )
        )
        
        logger.info(f"User {user_id} joined session: {session_id}")
        return True
    
    async def leave_collaboration_session(
        self,
        session_id: str,
        user_id: str
    ) -> bool:
        """
        Leave a collaboration session
        
        Args:
            session_id: Session to leave
            user_id: User leaving the session
            
        Returns:
            True if successfully left, False otherwise
        """
        logger.info(f"User {user_id} leaving session: {session_id}")
        
        # Remove user from session
        session = self.active_sessions.get(session_id)
        if session and user_id in session.participating_agents:
            session.participating_agents.remove(user_id)
            session.updated_at = datetime.utcnow()
            await self._persist_session(session)
        
        # Unregister WebSocket connection
        await self._unregister_websocket(session_id, user_id)
        
        # Update presence
        await self._update_user_presence(user_id, session_id, PresenceStatus.OFFLINE)
        
        # Broadcast user leave event
        await self._broadcast_update(
            session_id,
            RealTimeUpdate(
                update_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.utcnow(),
                update_type=UpdateType.USER_LEAVE.value,
                content={"user_id": user_id},
                sender_id=user_id,
                visibility=[]  # All participants can see
            )
        )
        
        logger.info(f"User {user_id} left session: {session_id}")
        return True
    
    # ===== REAL-TIME UPDATES =====
    
    async def send_real_time_update(
        self,
        session_id: str,
        update: RealTimeUpdate
    ) -> bool:
        """
        Send a real-time update to session participants
        
        Args:
            session_id: Target session
            update: Update to send
            
        Returns:
            True if update sent successfully, False otherwise
        """
        logger.debug(f"Sending real-time update to session: {session_id}")
        
        # Validate session
        if session_id not in self.active_sessions:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        # Store update for persistence
        await self._store_update(update)
        
        # Broadcast to all participants
        await self._broadcast_update(session_id, update)
        
        # Handle specific update types
        await self._handle_update_type(session_id, update)
        
        return True
    
    async def _handle_update_type(
        self,
        session_id: str,
        update: RealTimeUpdate
    ):
        """
        Handle specific update types with custom logic
        
        Args:
            session_id: Session ID
            update: Update to handle
        """
        update_type = UpdateType(update.update_type)
        
        if update_type == UpdateType.DOCUMENT_CHANGE:
            # Handle document changes with conflict resolution
            await self._handle_document_change(session_id, update)
        
        elif update_type == UpdateType.DECISION_POINT:
            # Handle decision points
            await self._handle_decision_point(session_id, update)
        
        elif update_type == UpdateType.STATUS_CHANGE:
            # Handle status changes
            await self._handle_status_change(session_id, update)
        
        elif update_type == UpdateType.COMMENT:
            # Handle comments and annotations
            await self._handle_comment(session_id, update)
    
    async def _handle_document_change(
        self,
        session_id: str,
        update: RealTimeUpdate
    ):
        """
        Handle document changes with conflict resolution
        
        Args:
            session_id: Session ID
            update: Document change update
        """
        logger.debug(f"Handling document change for session: {session_id}")
        
        # Get current document state
        workspace = self.shared_workspaces.get(session_id, {})
        current_doc = workspace.get("document", {})
        
        # Apply change with conflict resolution
        if self.conflict_resolution_strategy == ConflictResolution.OPERATIONAL_TRANSFORM:
            # Apply operational transformation
            resolved_change = await self._apply_operational_transform(
                current_doc,
                update.content
            )
        else:
            # Simple last-writer-wins
            resolved_change = update.content
        
        # Update workspace
        workspace["document"] = resolved_change
        workspace["last_updated"] = datetime.utcnow().isoformat()
        workspace["last_editor"] = update.sender_id
        
        self.shared_workspaces[session_id] = workspace
        
        # Persist workspace
        await self._persist_workspace(session_id, workspace)
        
        # Broadcast resolved change to other participants
        resolved_update = RealTimeUpdate(
            update_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.utcnow(),
            update_type=UpdateType.DOCUMENT_CHANGE.value,
            content=resolved_change,
            sender_id="system",
            visibility=[]  # All participants
        )
        
        await self._broadcast_update(session_id, resolved_update, exclude_user=update.sender_id)
    
    async def _handle_decision_point(
        self,
        session_id: str,
        update: RealTimeUpdate
    ):
        """
        Handle decision points in collaboration
        
        Args:
            session_id: Session ID
            update: Decision point update
        """
        logger.info(f"Handling decision point for session: {session_id}")
        
        # Add decision point to session
        session = self.active_sessions.get(session_id)
        if session:
            decision_point = {
                "id": str(uuid.uuid4()),
                "title": update.content.get("title", "Decision Required"),
                "description": update.content.get("description", ""),
                "options": update.content.get("options", []),
                "required_approvers": update.content.get("required_approvers", []),
                "deadline": update.content.get("deadline"),
                "created_by": update.sender_id,
                "created_at": datetime.utcnow().isoformat(),
                "status": "pending",
                "votes": {}
            }
            
            session.decision_points.append(decision_point)
            session.updated_at = datetime.utcnow()
            await self._persist_session(session)
            
            # Notify decision approvers
            await self._notify_decision_approvers(session_id, decision_point)
    
    # ===== SHARED WORKSPACE =====
    
    async def get_shared_workspace(
        self,
        session_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get shared workspace for a session
        
        Args:
            session_id: Session ID
            user_id: Requesting user ID
            
        Returns:
            Shared workspace data
        """
        logger.debug(f"Getting shared workspace for session: {session_id}")
        
        # Check if user has access to session
        session = self.active_sessions.get(session_id)
        if not session or user_id not in session.participating_agents:
            logger.warning(f"User {user_id} not authorized for session: {session_id}")
            return {}
        
        # Get workspace from cache
        workspace = self.shared_workspaces.get(session_id)
        if not workspace:
            # Load from persistent storage
            workspace = await self._load_workspace(session_id)
            if workspace:
                self.shared_workspaces[session_id] = workspace
        
        return workspace or {}
    
    async def update_shared_workspace(
        self,
        session_id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update shared workspace
        
        Args:
            session_id: Session ID
            user_id: User making the update
            updates: Updates to apply
            
        Returns:
            True if update successful, False otherwise
        """
        logger.debug(f"Updating shared workspace for session: {session_id}")
        
        # Get current workspace
        workspace = await self.get_shared_workspace(session_id, user_id)
        if not workspace:
            return False
        
        # Apply updates
        for key, value in updates.items():
            workspace[key] = value
        
        workspace["last_updated"] = datetime.utcnow().isoformat()
        workspace["last_editor"] = user_id
        
        # Update cache
        self.shared_workspaces[session_id] = workspace
        
        # Persist workspace
        await self._persist_workspace(session_id, workspace)
        
        # Broadcast update to participants
        update = RealTimeUpdate(
            update_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.utcnow(),
            update_type=UpdateType.DOCUMENT_CHANGE.value,
            content={"updates": updates},
            sender_id=user_id,
            visibility=[]  # All participants
        )
        
        await self._broadcast_update(session_id, update)
        
        return True
    
    # ===== PRESENCE MANAGEMENT =====
    
    async def get_session_presence(
        self,
        session_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get presence information for session participants
        
        Args:
            session_id: Session ID
            
        Returns:
            Presence information for all participants
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return {}
        
        presence_info = {}
        for participant in session.participating_agents:
            user_presence = self.user_presence.get(participant, {})
            session_presence = user_presence.get(session_id, {})
            
            presence_info[participant] = {
                "status": session_presence.get("status", PresenceStatus.OFFLINE.value),
                "last_seen": session_presence.get("last_seen"),
                "cursor_position": session_presence.get("cursor_position"),
                "current_selection": session_presence.get("current_selection")
            }
        
        return presence_info
    
    async def _update_user_presence(
        self,
        user_id: str,
        session_id: str,
        status: PresenceStatus,
        additional_data: Dict[str, Any] = None
    ):
        """
        Update user presence information
        
        Args:
            user_id: User ID
            session_id: Session ID
            status: Presence status
            additional_data: Additional presence data
        """
        if user_id not in self.user_presence:
            self.user_presence[user_id] = {}
        
        if session_id not in self.user_presence[user_id]:
            self.user_presence[user_id][session_id] = {}
        
        presence_data = {
            "status": status.value,
            "last_seen": datetime.utcnow().isoformat(),
            "session_id": session_id
        }
        
        if additional_data:
            presence_data.update(additional_data)
        
        self.user_presence[user_id][session_id] = presence_data
        
        # Broadcast presence update
        await self._broadcast_presence_update(session_id, user_id, presence_data)
    
    # ===== WEBSOCKET MANAGEMENT =====
    
    async def _register_websocket(
        self,
        session_id: str,
        user_id: str,
        websocket_connection: Any
    ):
        """
        Register WebSocket connection for real-time updates
        
        Args:
            session_id: Session ID
            user_id: User ID
            websocket_connection: WebSocket connection
        """
        connection_id = f"{session_id}:{user_id}"
        
        self.websocket_connections[connection_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "connection": websocket_connection,
            "connected_at": datetime.utcnow()
        }
        
        logger.info(f"WebSocket registered for user {user_id} in session {session_id}")
    
    async def _unregister_websocket(
        self,
        session_id: str,
        user_id: str
    ):
        """
        Unregister WebSocket connection
        
        Args:
            session_id: Session ID
            user_id: User ID
        """
        connection_id = f"{session_id}:{user_id}"
        
        if connection_id in self.websocket_connections:
            del self.websocket_connections[connection_id]
            logger.info(f"WebSocket unregistered for user {user_id} in session {session_id}")
    
    async def _broadcast_update(
        self,
        session_id: str,
        update: RealTimeUpdate,
        exclude_user: str = None
    ):
        """
        Broadcast update to all session participants
        
        Args:
            session_id: Session ID
            update: Update to broadcast
            exclude_user: User to exclude from broadcast
        """
        update_data = {
            "type": "real_time_update",
            "data": asdict(update)
        }
        
        # Send via WebSocket to connected users
        for connection_id, connection_info in self.websocket_connections.items():
            if connection_info["session_id"] == session_id:
                user_id = connection_info["user_id"]
                
                # Skip excluded user
                if exclude_user and user_id == exclude_user:
                    continue
                
                # Check visibility
                if update.visibility and user_id not in update.visibility:
                    continue
                
                try:
                    # Send update via WebSocket
                    await self._send_websocket_message(connection_info["connection"], update_data)
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")
                    # Remove broken connection
                    del self.websocket_connections[connection_id]
        
        # Also publish via Pulsar for other services
        await self.pulsar_service.publish(
            f"q.collaboration.updates.{session_id}",
            update_data
        )
    
    async def _send_websocket_message(
        self,
        websocket_connection: Any,
        message: Dict[str, Any]
    ):
        """
        Send message via WebSocket
        
        Args:
            websocket_connection: WebSocket connection
            message: Message to send
        """
        # This would be implemented based on the WebSocket library used
        # For now, it's a placeholder
        pass
    
    # ===== PERSISTENCE =====
    
    async def _persist_session(self, session: CollaborationSession):
        """Persist session to storage"""
        await self.ignite_service.put(
            f"collaboration_session:{session.session_id}",
            asdict(session)
        )
    
    async def _load_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Load session from storage"""
        session_data = await self.ignite_service.get(f"collaboration_session:{session_id}")
        if session_data:
            return CollaborationSession(**session_data)
        return None
    
    async def _persist_workspace(self, session_id: str, workspace: Dict[str, Any]):
        """Persist workspace to storage"""
        await self.ignite_service.put(
            f"shared_workspace:{session_id}",
            workspace
        )
    
    async def _load_workspace(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load workspace from storage"""
        return await self.ignite_service.get(f"shared_workspace:{session_id}")
    
    async def _store_update(self, update: RealTimeUpdate):
        """Store update for audit trail"""
        await self.ignite_service.put(
            f"rt_update:{update.session_id}:{update.update_id}",
            asdict(update)
        )
    
    # ===== BACKGROUND TASKS =====
    
    async def _presence_cleanup_task(self):
        """Background task to clean up stale presence data"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                cutoff_time = datetime.utcnow() - timedelta(seconds=self.presence_timeout)
                
                # Clean up stale presence data
                for user_id, user_sessions in list(self.user_presence.items()):
                    for session_id, presence_data in list(user_sessions.items()):
                        last_seen = datetime.fromisoformat(presence_data["last_seen"])
                        if last_seen < cutoff_time:
                            del user_sessions[session_id]
                    
                    # Remove user if no active sessions
                    if not user_sessions:
                        del self.user_presence[user_id]
                
            except Exception as e:
                logger.error(f"Error in presence cleanup task: {e}")
    
    async def _session_cleanup_task(self):
        """Background task to clean up inactive sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                cutoff_time = datetime.utcnow() - timedelta(seconds=self.session_timeout)
                
                # Clean up inactive sessions
                for session_id, session in list(self.active_sessions.items()):
                    if session.updated_at < cutoff_time:
                        # Check if any users are still active
                        has_active_users = any(
                            user_id in self.user_presence and 
                            session_id in self.user_presence[user_id]
                            for user_id in session.participating_agents
                        )
                        
                        if not has_active_users:
                            # Mark session as inactive
                            session.status = CollaborationStatus.CANCELLED
                            await self._persist_session(session)
                            del self.active_sessions[session_id]
                            
                            # Clean up workspace
                            if session_id in self.shared_workspaces:
                                del self.shared_workspaces[session_id]
                
            except Exception as e:
                logger.error(f"Error in session cleanup task: {e}")
    
    async def _sync_task(self):
        """Background task for periodic synchronization"""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                
                # Sync workspace changes
                for session_id, workspace in self.shared_workspaces.items():
                    if workspace.get("needs_sync", False):
                        await self._persist_workspace(session_id, workspace)
                        workspace["needs_sync"] = False
                
            except Exception as e:
                logger.error(f"Error in sync task: {e}")
    
    # ===== HELPER METHODS =====
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for real-time collaboration"""
        topics = [
            "q.collaboration.session.created",
            "q.collaboration.session.updated",
            "q.collaboration.session.completed",
            "q.collaboration.updates",
            "q.collaboration.presence"
        ]
        
        for topic in topics:
            await self.pulsar_service.ensure_topic(topic)
    
    async def _initialize_workspace_storage(self):
        """Initialize workspace storage"""
        # This would setup the storage backend for shared workspaces
        pass
    
    async def _initialize_shared_workspace(self, session_id: str):
        """Initialize shared workspace for a session"""
        workspace = {
            "session_id": session_id,
            "document": {},
            "annotations": [],
            "comments": [],
            "decisions": [],
            "files": [],
            "whiteboard": {},
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "version": 1
        }
        
        self.shared_workspaces[session_id] = workspace
        await self._persist_workspace(session_id, workspace)
    
    async def _apply_operational_transform(
        self,
        current_doc: Dict[str, Any],
        change: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply operational transformation for conflict resolution
        
        Args:
            current_doc: Current document state
            change: Proposed change
            
        Returns:
            Resolved document state
        """
        # This is a simplified operational transform
        # In a real implementation, this would be much more sophisticated
        
        # For now, apply simple merge strategy
        resolved_doc = current_doc.copy()
        
        # Apply changes
        for key, value in change.items():
            if key == "text" and key in resolved_doc:
                # Simple text merge (in practice, would use proper OT algorithms)
                resolved_doc[key] = value
            else:
                resolved_doc[key] = value
        
        return resolved_doc
    
    async def _broadcast_presence_update(
        self,
        session_id: str,
        user_id: str,
        presence_data: Dict[str, Any]
    ):
        """
        Broadcast presence update to session participants
        
        Args:
            session_id: Session ID
            user_id: User ID
            presence_data: Presence data
        """
        update = RealTimeUpdate(
            update_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.utcnow(),
            update_type="presence_update",
            content=presence_data,
            sender_id=user_id,
            visibility=[]  # All participants
        )
        
        await self._broadcast_update(session_id, update)
    
    async def _notify_decision_approvers(
        self,
        session_id: str,
        decision_point: Dict[str, Any]
    ):
        """
        Notify decision approvers
        
        Args:
            session_id: Session ID
            decision_point: Decision point data
        """
        # This would send notifications to required approvers
        # For now, just publish an event
        await self.pulsar_service.publish(
            "q.collaboration.decision.approval_required",
            {
                "session_id": session_id,
                "decision_point": decision_point,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# Global service instance
realtime_collaboration_service = RealTimeCollaborationService() 