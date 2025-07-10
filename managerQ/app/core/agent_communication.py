import logging
import time
import asyncio
import threading
import uuid
import json
from typing import Dict, List, Optional, Set, Callable, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pulsar

from .agent_registry import AgentRegistry, Agent
from shared.pulsar_client import SharedPulsarClient

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    """Types of messages between agents"""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    GROUP = "group"
    COORDINATION = "coordination"
    HEARTBEAT = "heartbeat"
    SERVICE_DISCOVERY = "service_discovery"
    WORKFLOW_SYNC = "workflow_sync"
    RESOURCE_REQUEST = "resource_request"
    STATUS_UPDATE = "status_update"

class MessagePriority(str, Enum):
    """Message priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

class DeliveryMode(str, Enum):
    """Message delivery modes"""
    FIRE_AND_FORGET = "fire_and_forget"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"
    RELIABLE = "reliable"

@dataclass
class Message:
    """Represents a message between agents"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str] = None  # None for broadcast messages
    group_id: Optional[str] = None
    subject: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    timestamp: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    correlation_id: Optional[str] = None  # For request-response patterns
    reply_to: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'group_id': self.group_id,
            'subject': self.subject,
            'content': self.content,
            'priority': self.priority.value,
            'delivery_mode': self.delivery_mode.value,
            'timestamp': self.timestamp,
            'expires_at': self.expires_at,
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'headers': self.headers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            message_id=data['message_id'],
            message_type=MessageType(data['message_type']),
            sender_id=data['sender_id'],
            recipient_id=data.get('recipient_id'),
            group_id=data.get('group_id'),
            subject=data.get('subject', ''),
            content=data.get('content', {}),
            priority=MessagePriority(data.get('priority', 'normal')),
            delivery_mode=DeliveryMode(data.get('delivery_mode', 'at_least_once')),
            timestamp=data.get('timestamp', time.time()),
            expires_at=data.get('expires_at'),
            correlation_id=data.get('correlation_id'),
            reply_to=data.get('reply_to'),
            headers=data.get('headers', {})
        )

@dataclass
class AgentGroup:
    """Represents a group of agents for coordination"""
    group_id: str
    name: str
    description: str
    members: Set[str] = field(default_factory=set)
    coordinators: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_retention_hours: int = 24
    max_members: int = 100

class MessageRouter:
    """Routes messages between agents using different strategies"""
    
    def __init__(self, agent_registry: AgentRegistry, pulsar_client: SharedPulsarClient):
        self.agent_registry = agent_registry
        self.pulsar_client = pulsar_client
        self.routing_table: Dict[str, str] = {}  # agent_id -> topic_name
        self.group_routes: Dict[str, Set[str]] = defaultdict(set)  # group_id -> agent_ids
        
    def update_routing_table(self):
        """Update routing table from agent registry"""
        for agent in self.agent_registry.get_all_agents():
            self.routing_table[agent.agent_id] = agent.topic_name
    
    def get_agent_topic(self, agent_id: str) -> Optional[str]:
        """Get the communication topic for an agent"""
        if agent_id not in self.routing_table:
            self.update_routing_table()
        return self.routing_table.get(agent_id)
    
    def add_agent_to_group(self, agent_id: str, group_id: str):
        """Add agent to a communication group"""
        self.group_routes[group_id].add(agent_id)
    
    def remove_agent_from_group(self, agent_id: str, group_id: str):
        """Remove agent from a communication group"""
        self.group_routes[group_id].discard(agent_id)
    
    def get_group_members(self, group_id: str) -> Set[str]:
        """Get all members of a group"""
        return self.group_routes[group_id].copy()

class MessageDeliveryService:
    """Handles reliable message delivery with different delivery modes"""
    
    def __init__(self, pulsar_client: SharedPulsarClient):
        self.pulsar_client = pulsar_client
        self.pending_messages: Dict[str, Message] = {}
        self.delivery_confirmations: Dict[str, Dict] = defaultdict(dict)
        self.retry_attempts: Dict[str, int] = defaultdict(int)
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
    async def send_message(self, message: Message, topic: str) -> bool:
        """Send a message with specified delivery mode"""
        try:
            if message.delivery_mode == DeliveryMode.FIRE_AND_FORGET:
                return await self._send_fire_and_forget(message, topic)
            elif message.delivery_mode == DeliveryMode.AT_LEAST_ONCE:
                return await self._send_at_least_once(message, topic)
            elif message.delivery_mode == DeliveryMode.EXACTLY_ONCE:
                return await self._send_exactly_once(message, topic)
            elif message.delivery_mode == DeliveryMode.RELIABLE:
                return await self._send_reliable(message, topic)
            else:
                logger.error(f"Unknown delivery mode: {message.delivery_mode}")
                return False
        except Exception as e:
            logger.error(f"Failed to send message {message.message_id}: {e}", exc_info=True)
            return False
    
    async def _send_fire_and_forget(self, message: Message, topic: str) -> bool:
        """Send message without delivery confirmation"""
        try:
            self.pulsar_client.publish_message(topic, message.to_dict())
            return True
        except Exception as e:
            logger.error(f"Fire-and-forget send failed: {e}")
            return False
    
    async def _send_at_least_once(self, message: Message, topic: str) -> bool:
        """Send message with at-least-once delivery guarantee"""
        try:
            # Store for potential retry
            self.pending_messages[message.message_id] = message
            
            self.pulsar_client.publish_message(topic, message.to_dict())
            
            # Start confirmation timeout
            asyncio.create_task(self._wait_for_confirmation(message.message_id, topic))
            return True
        except Exception as e:
            logger.error(f"At-least-once send failed: {e}")
            return False
    
    async def _send_exactly_once(self, message: Message, topic: str) -> bool:
        """Send message with exactly-once delivery guarantee"""
        # This would require additional deduplication logic
        # For now, implement as reliable delivery
        return await self._send_reliable(message, topic)
    
    async def _send_reliable(self, message: Message, topic: str) -> bool:
        """Send message with reliable delivery (with retries)"""
        attempt = 0
        while attempt < self.max_retries:
            try:
                self.pulsar_client.publish_message(topic, message.to_dict())
                return True
            except Exception as e:
                attempt += 1
                if attempt < self.max_retries:
                    logger.warning(f"Reliable send attempt {attempt} failed, retrying: {e}")
                    await asyncio.sleep(self.retry_delay * attempt)
                else:
                    logger.error(f"Reliable send failed after {attempt} attempts: {e}")
        return False
    
    async def _wait_for_confirmation(self, message_id: str, topic: str):
        """Wait for delivery confirmation and retry if needed"""
        await asyncio.sleep(30)  # Wait 30 seconds for confirmation
        
        if message_id in self.pending_messages and message_id not in self.delivery_confirmations:
            # No confirmation received, retry
            self.retry_attempts[message_id] += 1
            if self.retry_attempts[message_id] < self.max_retries:
                message = self.pending_messages[message_id]
                logger.info(f"Retrying message {message_id} (attempt {self.retry_attempts[message_id]})")
                await self._send_at_least_once(message, topic)
            else:
                logger.error(f"Message {message_id} failed after {self.max_retries} attempts")
                self.pending_messages.pop(message_id, None)
    
    def confirm_delivery(self, message_id: str, recipient_id: str):
        """Confirm message delivery"""
        self.delivery_confirmations[message_id][recipient_id] = time.time()
        
        # Clean up if all confirmations received
        if message_id in self.pending_messages:
            self.pending_messages.pop(message_id, None)
            self.retry_attempts.pop(message_id, None)

class GroupManager:
    """Manages agent groups for coordination and communication"""
    
    def __init__(self):
        self.groups: Dict[str, AgentGroup] = {}
        self.agent_memberships: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> group_ids
        
    def create_group(self, name: str, description: str = "", coordinator_id: Optional[str] = None,
                    max_members: int = 100, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new agent group"""
        group_id = f"group_{uuid.uuid4()}"
        
        group = AgentGroup(
            group_id=group_id,
            name=name,
            description=description,
            max_members=max_members,
            metadata=metadata or {}
        )
        
        if coordinator_id:
            group.coordinators.add(coordinator_id)
            group.members.add(coordinator_id)
            self.agent_memberships[coordinator_id].add(group_id)
        
        self.groups[group_id] = group
        logger.info(f"Created group '{name}' with ID {group_id}")
        return group_id
    
    def add_agent_to_group(self, agent_id: str, group_id: str) -> bool:
        """Add an agent to a group"""
        if group_id not in self.groups:
            return False
        
        group = self.groups[group_id]
        if len(group.members) >= group.max_members:
            logger.warning(f"Group {group_id} is at maximum capacity")
            return False
        
        group.members.add(agent_id)
        self.agent_memberships[agent_id].add(group_id)
        logger.info(f"Added agent {agent_id} to group {group_id}")
        return True
    
    def remove_agent_from_group(self, agent_id: str, group_id: str) -> bool:
        """Remove an agent from a group"""
        if group_id not in self.groups:
            return False
        
        group = self.groups[group_id]
        group.members.discard(agent_id)
        group.coordinators.discard(agent_id)
        self.agent_memberships[agent_id].discard(group_id)
        
        logger.info(f"Removed agent {agent_id} from group {group_id}")
        return True
    
    def get_group(self, group_id: str) -> Optional[AgentGroup]:
        """Get group by ID"""
        return self.groups.get(group_id)
    
    def get_agent_groups(self, agent_id: str) -> List[AgentGroup]:
        """Get all groups an agent belongs to"""
        group_ids = self.agent_memberships[agent_id]
        return [self.groups[gid] for gid in group_ids if gid in self.groups]
    
    def promote_to_coordinator(self, agent_id: str, group_id: str) -> bool:
        """Promote an agent to group coordinator"""
        if group_id not in self.groups or agent_id not in self.groups[group_id].members:
            return False
        
        self.groups[group_id].coordinators.add(agent_id)
        logger.info(f"Promoted agent {agent_id} to coordinator of group {group_id}")
        return True

class AgentCommunicationHub:
    """Central hub for agent-to-agent communication"""
    
    def __init__(self, agent_registry: AgentRegistry, pulsar_client: SharedPulsarClient):
        self.agent_registry = agent_registry
        self.pulsar_client = pulsar_client
        
        # Initialize components
        self.message_router = MessageRouter(agent_registry, pulsar_client)
        self.delivery_service = MessageDeliveryService(pulsar_client)
        self.group_manager = GroupManager()
        
        # Message handling
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.message_history: deque = deque(maxlen=10000)
        self.active_conversations: Dict[str, List[Message]] = defaultdict(list)
        
        # Service discovery
        self.service_registry: Dict[str, Dict[str, Any]] = {}
        self.service_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # Communication topics
        self.broadcast_topic = "persistent://public/default/q.agent.broadcast"
        self.discovery_topic = "persistent://public/default/q.agent.discovery"
        self.coordination_topic = "persistent://public/default/q.agent.coordination"
        
        # Consumer management
        self._consumers: Dict[str, Any] = {}
        self._running = False
    
    async def start(self):
        """Start the communication hub"""
        if self._running:
            return
        
        self._running = True
        
        # Start consumers for different message types
        await self._start_consumers()
        
        # Start service discovery
        await self._start_service_discovery()
        
        logger.info("Agent communication hub started")
    
    async def stop(self):
        """Stop the communication hub"""
        self._running = False
        
        # Close all consumers
        for consumer in self._consumers.values():
            try:
                consumer.close()
            except Exception as e:
                logger.error(f"Error closing consumer: {e}")
        
        self._consumers.clear()
        logger.info("Agent communication hub stopped")
    
    async def _start_consumers(self):
        """Start Pulsar consumers for different communication channels"""
        try:
            self.pulsar_client._connect()
            if not self.pulsar_client._client:
                logger.error("Pulsar client not available for communication hub")
                return
            
            # Broadcast consumer
            broadcast_consumer = self.pulsar_client._client.subscribe(
                self.broadcast_topic,
                subscription_name="agent-communication-broadcast-sub"
            )
            self._consumers['broadcast'] = broadcast_consumer
            asyncio.create_task(self._consume_messages(broadcast_consumer, MessageType.BROADCAST))
            
            # Discovery consumer
            discovery_consumer = self.pulsar_client._client.subscribe(
                self.discovery_topic,
                subscription_name="agent-communication-discovery-sub"
            )
            self._consumers['discovery'] = discovery_consumer
            asyncio.create_task(self._consume_messages(discovery_consumer, MessageType.SERVICE_DISCOVERY))
            
            # Coordination consumer
            coordination_consumer = self.pulsar_client._client.subscribe(
                self.coordination_topic,
                subscription_name="agent-communication-coordination-sub"
            )
            self._consumers['coordination'] = coordination_consumer
            asyncio.create_task(self._consume_messages(coordination_consumer, MessageType.COORDINATION))
            
        except Exception as e:
            logger.error(f"Failed to start communication consumers: {e}", exc_info=True)
    
    async def _consume_messages(self, consumer, message_type: MessageType):
        """Consume messages from a specific topic"""
        while self._running:
            try:
                msg = consumer.receive(timeout_millis=1000)
                if msg:
                    message_data = json.loads(msg.data().decode('utf-8'))
                    message = Message.from_dict(message_data)
                    
                    await self._handle_message(message)
                    consumer.acknowledge(msg)
                    
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error(f"Error consuming {message_type.value} messages: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: Message):
        """Handle received message"""
        # Record message in history
        self.message_history.append(message)
        
        # Track conversations
        if message.correlation_id:
            self.active_conversations[message.correlation_id].append(message)
        
        # Route to registered handlers
        handlers = self.message_handlers.get(message.message_type, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error in message handler: {e}", exc_info=True)
    
    async def send_direct_message(self, sender_id: str, recipient_id: str, subject: str, 
                                 content: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL,
                                 delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE) -> str:
        """Send a direct message to another agent"""
        
        # Get recipient's topic
        recipient_topic = self.message_router.get_agent_topic(recipient_id)
        if not recipient_topic:
            raise ValueError(f"Agent {recipient_id} not found or not reachable")
        
        # Create message
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.DIRECT,
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject=subject,
            content=content,
            priority=priority,
            delivery_mode=delivery_mode
        )
        
        # Send message
        success = await self.delivery_service.send_message(message, recipient_topic)
        if success:
            logger.info(f"Direct message sent from {sender_id} to {recipient_id}: {subject}")
        
        return message.message_id
    
    async def broadcast_message(self, sender_id: str, subject: str, content: Dict[str, Any],
                               group_id: Optional[str] = None, 
                               personality_filter: Optional[str] = None) -> str:
        """Broadcast a message to multiple agents"""
        
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.BROADCAST if not group_id else MessageType.GROUP,
            sender_id=sender_id,
            group_id=group_id,
            subject=subject,
            content=content,
            priority=MessagePriority.NORMAL
        )
        
        if group_id:
            # Send to group members
            group = self.group_manager.get_group(group_id)
            if group:
                for member_id in group.members:
                    if member_id != sender_id:  # Don't send to self
                        recipient_topic = self.message_router.get_agent_topic(member_id)
                        if recipient_topic:
                            await self.delivery_service.send_message(message, recipient_topic)
        else:
            # Broadcast to all agents or filtered subset
            agents = self.agent_registry.get_all_agents()
            if personality_filter:
                agents = [a for a in agents if personality_filter in a.capabilities.personalities]
            
            for agent in agents:
                if agent.agent_id != sender_id:  # Don't send to self
                    await self.delivery_service.send_message(message, agent.topic_name)
        
        logger.info(f"Broadcast message sent from {sender_id}: {subject}")
        return message.message_id
    
    async def send_coordination_request(self, sender_id: str, request_type: str, 
                                      content: Dict[str, Any], target_agents: Optional[List[str]] = None) -> str:
        """Send a coordination request to specific agents or all agents"""
        
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.COORDINATION,
            sender_id=sender_id,
            subject=f"coordination_{request_type}",
            content={'request_type': request_type, **content},
            priority=MessagePriority.HIGH,
            delivery_mode=DeliveryMode.RELIABLE
        )
        
        if target_agents:
            # Send to specific agents
            for agent_id in target_agents:
                recipient_topic = self.message_router.get_agent_topic(agent_id)
                if recipient_topic:
                    await self.delivery_service.send_message(message, recipient_topic)
        else:
            # Send to coordination topic for all interested agents
            await self.delivery_service.send_message(message, self.coordination_topic)
        
        logger.info(f"Coordination request sent from {sender_id}: {request_type}")
        return message.message_id
    
    async def reply_to_message(self, original_message: Message, sender_id: str, 
                              content: Dict[str, Any]) -> str:
        """Reply to a received message"""
        
        if not original_message.reply_to and not original_message.sender_id:
            raise ValueError("Cannot reply to message without reply_to or sender_id")
        
        reply_message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.DIRECT,
            sender_id=sender_id,
            recipient_id=original_message.sender_id,
            subject=f"Re: {original_message.subject}",
            content=content,
            correlation_id=original_message.correlation_id or original_message.message_id,
            priority=original_message.priority
        )
        
        # Send reply
        recipient_topic = self.message_router.get_agent_topic(original_message.sender_id)
        if recipient_topic:
            await self.delivery_service.send_message(reply_message, recipient_topic)
        
        return reply_message.message_id
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a handler for specific message types"""
        self.message_handlers[message_type].append(handler)
        logger.info(f"Registered handler for {message_type.value} messages")
    
    async def _start_service_discovery(self):
        """Start service discovery mechanism"""
        # This would periodically announce available services
        asyncio.create_task(self._service_discovery_loop())
    
    async def _service_discovery_loop(self):
        """Periodic service discovery announcements"""
        while self._running:
            try:
                # Announce available services from registered agents
                for agent in self.agent_registry.get_all_agents():
                    service_info = {
                        'agent_id': agent.agent_id,
                        'personality': agent.personality,
                        'capabilities': list(agent.capabilities.personalities),
                        'tools': list(agent.capabilities.supported_tools),
                        'status': agent.status.value,
                        'load': agent.metrics.current_load
                    }
                    
                    discovery_message = Message(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.SERVICE_DISCOVERY,
                        sender_id=agent.agent_id,
                        subject="service_announcement",
                        content=service_info
                    )
                    
                    await self.delivery_service.send_message(discovery_message, self.discovery_topic)
                
                await asyncio.sleep(60)  # Announce every minute
                
            except Exception as e:
                logger.error(f"Error in service discovery loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            'total_messages': len(self.message_history),
            'active_conversations': len(self.active_conversations),
            'registered_groups': len(self.group_manager.groups),
            'pending_deliveries': len(self.delivery_service.pending_messages),
            'message_types_distribution': {
                msg_type.value: sum(1 for msg in self.message_history if msg.message_type == msg_type)
                for msg_type in MessageType
            },
            'agents_in_groups': sum(len(group.members) for group in self.group_manager.groups.values())
        }

# Singleton instance
agent_communication_hub: Optional[AgentCommunicationHub] = None 