import logging
import time
import asyncio
import threading
import uuid
import random
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from .agent_registry import AgentRegistry, Agent
from .agent_communication import AgentCommunicationHub, Message, MessageType, MessagePriority
from shared.pulsar_client import SharedPulsarClient

logger = logging.getLogger(__name__)

class NodeState(str, Enum):
    """States for consensus nodes"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    OBSERVER = "observer"

class LockState(str, Enum):
    """States for distributed locks"""
    AVAILABLE = "available"
    ACQUIRED = "acquired"
    EXPIRED = "expired"
    CONTENDED = "contended"

class ConsensusMessageType(str, Enum):
    """Types of consensus messages"""
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    HEARTBEAT = "heartbeat"
    APPEND_ENTRIES = "append_entries"
    APPEND_RESPONSE = "append_response"
    LEADER_ANNOUNCEMENT = "leader_announcement"

@dataclass
class LogEntry:
    """Represents an entry in the distributed log"""
    term: int
    index: int
    command: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    committed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'term': self.term,
            'index': self.index,
            'command': self.command,
            'data': self.data,
            'timestamp': self.timestamp,
            'committed': self.committed
        }

@dataclass
class VoteRequest:
    """Vote request for leader election"""
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int

@dataclass
class VoteResponse:
    """Vote response for leader election"""
    term: int
    vote_granted: bool
    voter_id: str

@dataclass
class DistributedLock:
    """Represents a distributed lock"""
    lock_id: str
    resource: str
    owner: Optional[str] = None
    acquired_at: Optional[float] = None
    expires_at: Optional[float] = None
    state: LockState = LockState.AVAILABLE
    waiters: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConsensusNode:
    """Implementation of a consensus node (simplified Raft-like)"""
    
    def __init__(self, node_id: str, cluster_members: Set[str], 
                 communication_hub: AgentCommunicationHub):
        self.node_id = node_id
        self.cluster_members = cluster_members
        self.communication_hub = communication_hub
        
        # Consensus state
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.leader_id: Optional[str] = None
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Timing
        self.last_heartbeat = time.time()
        self.election_timeout = random.uniform(3, 6)  # 3-6 seconds
        self.heartbeat_interval = 1.0  # 1 second
        
        # Callbacks
        self.on_leader_change: Optional[Callable] = None
        self.on_log_committed: Optional[Callable] = None
        
        # Tasks
        self._election_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the consensus node"""
        if self._running:
            return
        
        self._running = True
        
        # Register message handlers
        self.communication_hub.register_message_handler(
            MessageType.COORDINATION, self._handle_consensus_message
        )
        
        # Start election timeout
        self._election_task = asyncio.create_task(self._election_timeout_loop())
        
        logger.info(f"Consensus node {self.node_id} started as {self.state.value}")
    
    async def stop(self):
        """Stop the consensus node"""
        self._running = False
        
        if self._election_task:
            self._election_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        self.state = NodeState.FOLLOWER
        logger.info(f"Consensus node {self.node_id} stopped")
    
    async def _election_timeout_loop(self):
        """Handle election timeouts"""
        while self._running:
            try:
                if self.state == NodeState.FOLLOWER:
                    # Check if we need to start an election
                    if time.time() - self.last_heartbeat > self.election_timeout:
                        await self._start_election()
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                logger.error(f"Error in election timeout loop: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _start_election(self):
        """Start a new election"""
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_heartbeat = time.time()
        
        logger.info(f"Node {self.node_id} starting election for term {self.current_term}")
        
        # Vote for ourselves
        votes_received = 1
        
        # Send vote requests to all other nodes
        last_log_index = len(self.log) - 1 if self.log else 0
        last_log_term = self.log[-1].term if self.log else 0
        
        vote_request = VoteRequest(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=last_log_index,
            last_log_term=last_log_term
        )
        
        for member_id in self.cluster_members:
            if member_id != self.node_id:
                await self._send_vote_request(member_id, vote_request)
        
        # Wait for votes (simplified - in practice would be more complex)
        await asyncio.sleep(2)
        
        # Check if we won the election
        if votes_received > len(self.cluster_members) // 2:
            await self._become_leader()
        else:
            self.state = NodeState.FOLLOWER
            self.voted_for = None
    
    async def _send_vote_request(self, target_id: str, vote_request: VoteRequest):
        """Send a vote request to another node"""
        try:
            await self.communication_hub.send_direct_message(
                sender_id=self.node_id,
                recipient_id=target_id,
                subject="vote_request",
                content={
                    'type': ConsensusMessageType.VOTE_REQUEST.value,
                    'term': vote_request.term,
                    'candidate_id': vote_request.candidate_id,
                    'last_log_index': vote_request.last_log_index,
                    'last_log_term': vote_request.last_log_term
                },
                priority=MessagePriority.HIGH
            )
        except Exception as e:
            logger.error(f"Failed to send vote request to {target_id}: {e}")
    
    async def _become_leader(self):
        """Become the leader of the cluster"""
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        
        # Initialize leader state
        for member_id in self.cluster_members:
            if member_id != self.node_id:
                self.next_index[member_id] = len(self.log)
                self.match_index[member_id] = 0
        
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        # Start sending heartbeats
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Notify about leader change
        if self.on_leader_change:
            await self.on_leader_change(self.node_id, self.current_term)
        
        # Announce leadership
        await self._announce_leadership()
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats as leader"""
        while self._running and self.state == NodeState.LEADER:
            try:
                await self._send_heartbeats()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _send_heartbeats(self):
        """Send heartbeats to all followers"""
        for member_id in self.cluster_members:
            if member_id != self.node_id:
                try:
                    await self.communication_hub.send_direct_message(
                        sender_id=self.node_id,
                        recipient_id=member_id,
                        subject="heartbeat",
                        content={
                            'type': ConsensusMessageType.HEARTBEAT.value,
                            'term': self.current_term,
                            'leader_id': self.node_id,
                            'commit_index': self.commit_index
                        },
                        priority=MessagePriority.HIGH
                    )
                except Exception as e:
                    logger.error(f"Failed to send heartbeat to {member_id}: {e}")
    
    async def _announce_leadership(self):
        """Announce leadership to all cluster members"""
        try:
            await self.communication_hub.broadcast_message(
                sender_id=self.node_id,
                subject="leader_announcement",
                content={
                    'type': ConsensusMessageType.LEADER_ANNOUNCEMENT.value,
                    'term': self.current_term,
                    'leader_id': self.node_id
                }
            )
        except Exception as e:
            logger.error(f"Failed to announce leadership: {e}")
    
    async def _handle_consensus_message(self, message: Message):
        """Handle consensus-related messages"""
        try:
            msg_type = message.content.get('type')
            
            if msg_type == ConsensusMessageType.VOTE_REQUEST.value:
                await self._handle_vote_request(message)
            elif msg_type == ConsensusMessageType.VOTE_RESPONSE.value:
                await self._handle_vote_response(message)
            elif msg_type == ConsensusMessageType.HEARTBEAT.value:
                await self._handle_heartbeat(message)
            elif msg_type == ConsensusMessageType.LEADER_ANNOUNCEMENT.value:
                await self._handle_leader_announcement(message)
            elif msg_type == ConsensusMessageType.APPEND_ENTRIES.value:
                await self._handle_append_entries(message)
            elif msg_type == ConsensusMessageType.APPEND_RESPONSE.value:
                await self._handle_append_response(message)
                
        except Exception as e:
            logger.error(f"Error handling consensus message: {e}", exc_info=True)
    
    async def _handle_vote_request(self, message: Message):
        """Handle vote request from candidate"""
        term = message.content.get('term')
        candidate_id = message.content.get('candidate_id')
        
        vote_granted = False
        
        # Update term if newer
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
            self.state = NodeState.FOLLOWER
        
        # Grant vote if conditions are met
        if (term == self.current_term and
            (self.voted_for is None or self.voted_for == candidate_id)):
            vote_granted = True
            self.voted_for = candidate_id
            self.last_heartbeat = time.time()
        
        # Send vote response
        await self._send_vote_response(candidate_id, vote_granted)
    
    async def _send_vote_response(self, candidate_id: str, vote_granted: bool):
        """Send vote response to candidate"""
        try:
            await self.communication_hub.send_direct_message(
                sender_id=self.node_id,
                recipient_id=candidate_id,
                subject="vote_response",
                content={
                    'type': ConsensusMessageType.VOTE_RESPONSE.value,
                    'term': self.current_term,
                    'vote_granted': vote_granted,
                    'voter_id': self.node_id
                }
            )
        except Exception as e:
            logger.error(f"Failed to send vote response to {candidate_id}: {e}")
    
    async def _handle_vote_response(self, message: Message):
        """Handle vote response from follower"""
        if self.state != NodeState.CANDIDATE:
            return
        
        term = message.content.get('term')
        vote_granted = message.content.get('vote_granted')
        
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            return
        
        if vote_granted and term == self.current_term:
            # Count votes (simplified)
            logger.info(f"Received vote from {message.sender_id}")
    
    async def _handle_heartbeat(self, message: Message):
        """Handle heartbeat from leader"""
        term = message.content.get('term')
        leader_id = message.content.get('leader_id')
        
        if term >= self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.leader_id = leader_id
            self.last_heartbeat = time.time()
            
            # Reset election timeout
            self.election_timeout = random.uniform(3, 6)
    
    async def _handle_leader_announcement(self, message: Message):
        """Handle leader announcement"""
        term = message.content.get('term')
        leader_id = message.content.get('leader_id')
        
        if term >= self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.leader_id = leader_id
            self.last_heartbeat = time.time()
            
            logger.info(f"Acknowledged {leader_id} as leader for term {term}")
    
    async def _handle_append_entries(self, message: Message):
        """Handle append entries request"""
        # Simplified implementation
        pass
    
    async def _handle_append_response(self, message: Message):
        """Handle append entries response"""
        # Simplified implementation
        pass
    
    async def append_log_entry(self, command: str, data: Dict[str, Any]) -> bool:
        """Append a new log entry (leader only)"""
        if self.state != NodeState.LEADER:
            return False
        
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log),
            command=command,
            data=data
        )
        
        self.log.append(entry)
        logger.info(f"Leader {self.node_id} appended log entry: {command}")
        
        # In a full implementation, would replicate to followers
        return True
    
    def is_leader(self) -> bool:
        """Check if this node is the leader"""
        return self.state == NodeState.LEADER
    
    def get_leader_id(self) -> Optional[str]:
        """Get the current leader ID"""
        return self.leader_id

class DistributedLockManager:
    """Manages distributed locks for resource coordination"""
    
    def __init__(self, node_id: str, communication_hub: AgentCommunicationHub):
        self.node_id = node_id
        self.communication_hub = communication_hub
        self.locks: Dict[str, DistributedLock] = {}
        self.lock_timeout = 300  # 5 minutes default timeout
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the lock manager"""
        if self._running:
            return
        
        self._running = True
        
        # Register message handlers
        self.communication_hub.register_message_handler(
            MessageType.COORDINATION, self._handle_lock_message
        )
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_locks())
        
        logger.info(f"Distributed lock manager started for node {self.node_id}")
    
    async def stop(self):
        """Stop the lock manager"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info(f"Distributed lock manager stopped for node {self.node_id}")
    
    async def acquire_lock(self, resource: str, timeout: Optional[float] = None) -> bool:
        """Acquire a distributed lock on a resource"""
        lock_id = f"lock_{resource}_{uuid.uuid4()}"
        timeout = timeout or self.lock_timeout
        
        # Check if lock already exists
        if resource in self.locks:
            existing_lock = self.locks[resource]
            if existing_lock.state == LockState.ACQUIRED:
                # Add to waiters
                existing_lock.waiters.append(self.node_id)
                logger.info(f"Node {self.node_id} waiting for lock on {resource}")
                return False
        
        # Create new lock
        lock = DistributedLock(
            lock_id=lock_id,
            resource=resource,
            owner=self.node_id,
            acquired_at=time.time(),
            expires_at=time.time() + timeout,
            state=LockState.ACQUIRED
        )
        
        self.locks[resource] = lock
        
        # Announce lock acquisition
        await self._announce_lock_acquisition(lock)
        
        logger.info(f"Node {self.node_id} acquired lock on {resource}")
        return True
    
    async def release_lock(self, resource: str) -> bool:
        """Release a distributed lock"""
        if resource not in self.locks:
            return False
        
        lock = self.locks[resource]
        if lock.owner != self.node_id:
            return False
        
        # Announce lock release
        await self._announce_lock_release(lock)
        
        # Handle waiters
        if lock.waiters:
            next_owner = lock.waiters.pop(0)
            lock.owner = next_owner
            lock.acquired_at = time.time()
            lock.expires_at = time.time() + self.lock_timeout
            
            # Notify next owner
            await self._notify_lock_granted(lock, next_owner)
        else:
            # No waiters, remove lock
            del self.locks[resource]
        
        logger.info(f"Node {self.node_id} released lock on {resource}")
        return True
    
    async def _announce_lock_acquisition(self, lock: DistributedLock):
        """Announce lock acquisition to other nodes"""
        try:
            await self.communication_hub.broadcast_message(
                sender_id=self.node_id,
                subject="lock_acquired",
                content={
                    'type': 'lock_acquired',
                    'lock_id': lock.lock_id,
                    'resource': lock.resource,
                    'owner': lock.owner,
                    'acquired_at': lock.acquired_at,
                    'expires_at': lock.expires_at
                }
            )
        except Exception as e:
            logger.error(f"Failed to announce lock acquisition: {e}")
    
    async def _announce_lock_release(self, lock: DistributedLock):
        """Announce lock release to other nodes"""
        try:
            await self.communication_hub.broadcast_message(
                sender_id=self.node_id,
                subject="lock_released",
                content={
                    'type': 'lock_released',
                    'lock_id': lock.lock_id,
                    'resource': lock.resource,
                    'owner': lock.owner
                }
            )
        except Exception as e:
            logger.error(f"Failed to announce lock release: {e}")
    
    async def _notify_lock_granted(self, lock: DistributedLock, new_owner: str):
        """Notify a node that they've been granted a lock"""
        try:
            await self.communication_hub.send_direct_message(
                sender_id=self.node_id,
                recipient_id=new_owner,
                subject="lock_granted",
                content={
                    'type': 'lock_granted',
                    'lock_id': lock.lock_id,
                    'resource': lock.resource,
                    'acquired_at': lock.acquired_at,
                    'expires_at': lock.expires_at
                }
            )
        except Exception as e:
            logger.error(f"Failed to notify lock granted to {new_owner}: {e}")
    
    async def _handle_lock_message(self, message: Message):
        """Handle lock-related messages"""
        try:
            msg_type = message.content.get('type')
            
            if msg_type == 'lock_acquired':
                await self._handle_lock_acquired(message)
            elif msg_type == 'lock_released':
                await self._handle_lock_released(message)
            elif msg_type == 'lock_granted':
                await self._handle_lock_granted(message)
                
        except Exception as e:
            logger.error(f"Error handling lock message: {e}", exc_info=True)
    
    async def _handle_lock_acquired(self, message: Message):
        """Handle lock acquisition announcement"""
        resource = message.content.get('resource')
        owner = message.content.get('owner')
        
        if resource in self.locks:
            lock = self.locks[resource]
            if lock.owner != owner:
                # Conflict detected - resolve by node ID ordering
                logger.warning(f"Lock conflict detected for {resource}")
                # Simplified conflict resolution
                if owner < self.node_id:
                    # Other node wins
                    del self.locks[resource]
    
    async def _handle_lock_released(self, message: Message):
        """Handle lock release announcement"""
        resource = message.content.get('resource')
        
        if resource in self.locks:
            lock = self.locks[resource]
            if lock.owner == message.sender_id:
                # Valid release
                if lock.waiters:
                    # Handle waiters
                    next_owner = lock.waiters.pop(0)
                    lock.owner = next_owner
                    lock.acquired_at = time.time()
                    lock.expires_at = time.time() + self.lock_timeout
                else:
                    del self.locks[resource]
    
    async def _handle_lock_granted(self, message: Message):
        """Handle lock granted notification"""
        resource = message.content.get('resource')
        logger.info(f"Granted lock on {resource}")
    
    async def _cleanup_expired_locks(self):
        """Clean up expired locks"""
        while self._running:
            try:
                current_time = time.time()
                expired_resources = []
                
                for resource, lock in self.locks.items():
                    if lock.expires_at and current_time > lock.expires_at:
                        expired_resources.append(resource)
                
                for resource in expired_resources:
                    logger.info(f"Cleaning up expired lock on {resource}")
                    await self.release_lock(resource)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in lock cleanup: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    def get_lock_status(self, resource: str) -> Optional[Dict[str, Any]]:
        """Get status of a lock"""
        if resource not in self.locks:
            return None
        
        lock = self.locks[resource]
        return {
            'lock_id': lock.lock_id,
            'resource': lock.resource,
            'owner': lock.owner,
            'state': lock.state.value,
            'acquired_at': lock.acquired_at,
            'expires_at': lock.expires_at,
            'waiters': lock.waiters.copy()
        }

class CoordinationProtocolManager:
    """Main coordinator for all coordination protocols"""
    
    def __init__(self, node_id: str, agent_registry: AgentRegistry, 
                 communication_hub: AgentCommunicationHub):
        self.node_id = node_id
        self.agent_registry = agent_registry
        self.communication_hub = communication_hub
        
        # Get cluster members from agent registry
        self.cluster_members = {agent.agent_id for agent in agent_registry.get_all_agents()}
        
        # Initialize components
        self.consensus_node = ConsensusNode(node_id, self.cluster_members, communication_hub)
        self.lock_manager = DistributedLockManager(node_id, communication_hub)
        
        # Coordination state
        self.coordination_groups: Dict[str, Set[str]] = {}
        self.active_coordinations: Dict[str, Dict[str, Any]] = {}
        
        self._running = False
    
    async def start(self):
        """Start all coordination protocols"""
        if self._running:
            return
        
        self._running = True
        
        # Start consensus node
        await self.consensus_node.start()
        
        # Start lock manager
        await self.lock_manager.start()
        
        logger.info(f"Coordination protocol manager started for node {self.node_id}")
    
    async def stop(self):
        """Stop all coordination protocols"""
        self._running = False
        
        # Stop components
        await self.consensus_node.stop()
        await self.lock_manager.stop()
        
        logger.info(f"Coordination protocol manager stopped for node {self.node_id}")
    
    async def coordinate_task(self, task_id: str, participating_agents: List[str], 
                            coordination_type: str = "consensus") -> bool:
        """Coordinate a task across multiple agents"""
        if coordination_type == "consensus":
            return await self._coordinate_with_consensus(task_id, participating_agents)
        elif coordination_type == "lock":
            return await self._coordinate_with_locks(task_id, participating_agents)
        else:
            logger.error(f"Unknown coordination type: {coordination_type}")
            return False
    
    async def _coordinate_with_consensus(self, task_id: str, participating_agents: List[str]) -> bool:
        """Coordinate using consensus protocol"""
        if not self.consensus_node.is_leader():
            logger.warning(f"Node {self.node_id} is not leader, cannot coordinate task {task_id}")
            return False
        
        # Append coordination request to log
        coordination_data = {
            'task_id': task_id,
            'participating_agents': participating_agents,
            'coordinator': self.node_id,
            'timestamp': time.time()
        }
        
        success = await self.consensus_node.append_log_entry(
            command="coordinate_task",
            data=coordination_data
        )
        
        if success:
            self.active_coordinations[task_id] = coordination_data
            logger.info(f"Started consensus coordination for task {task_id}")
        
        return success
    
    async def _coordinate_with_locks(self, task_id: str, participating_agents: List[str]) -> bool:
        """Coordinate using distributed locks"""
        # Acquire locks on all participating agents
        lock_resource = f"task_coordination_{task_id}"
        
        success = await self.lock_manager.acquire_lock(lock_resource)
        if success:
            self.active_coordinations[task_id] = {
                'type': 'lock',
                'resource': lock_resource,
                'participating_agents': participating_agents,
                'coordinator': self.node_id
            }
            logger.info(f"Started lock-based coordination for task {task_id}")
        
        return success
    
    async def finish_coordination(self, task_id: str) -> bool:
        """Finish coordination for a task"""
        if task_id not in self.active_coordinations:
            return False
        
        coordination = self.active_coordinations[task_id]
        
        if coordination.get('type') == 'lock':
            # Release locks
            resource = coordination.get('resource')
            if resource:
                await self.lock_manager.release_lock(resource)
        
        del self.active_coordinations[task_id]
        logger.info(f"Finished coordination for task {task_id}")
        return True
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get status of all coordination protocols"""
        return {
            'node_id': self.node_id,
            'consensus_state': self.consensus_node.state.value,
            'is_leader': self.consensus_node.is_leader(),
            'leader_id': self.consensus_node.get_leader_id(),
            'current_term': self.consensus_node.current_term,
            'active_coordinations': len(self.active_coordinations),
            'active_locks': len(self.lock_manager.locks),
            'cluster_members': list(self.cluster_members)
        }

# Singleton instance
coordination_manager: Optional[CoordinationProtocolManager] = None 