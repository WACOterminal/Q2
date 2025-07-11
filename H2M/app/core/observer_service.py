import asyncio
import logging
import json
import pulsar
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect
from collections import deque

logger = logging.getLogger(__name__)

class ObserverService:
    def __init__(self, pulsar_client: pulsar.Client):
        self.client = pulsar_client
        self.user_sessions: Dict[str, WebSocket] = {}
        self.user_action_history: Dict[str, deque] = {}
        self.producer = self.client.create_producer('persistent://public/default/q.h2m.user_actions')
        logger.info("ObserverService initialized and Pulsar producer created.")

    async def handle_connection(self, websocket: WebSocket, user_id: str):
        """Manages a user's WebSocket connection."""
        await websocket.accept()
        self.user_sessions[user_id] = websocket
        self.user_action_history[user_id] = deque(maxlen=100) # Store last 100 actions
        logger.info(f"Observer session started for user: {user_id}")
        
        try:
            while True:
                data = await websocket.receive_text()
                action_data = json.loads(data)
                self._process_user_action(user_id, action_data)
        except WebSocketDisconnect:
            self._disconnect(user_id)

    def _process_user_action(self, user_id: str, action: Dict):
        """Processes a user action and publishes it to Pulsar."""
        action['user_id'] = user_id
        action['timestamp'] = asyncio.get_event_loop().time()
        
        self.user_action_history[user_id].append(action)
        
        # Publish to Pulsar for the Chief of Staff agent to consume
        self.producer.send_async(json.dumps(action).encode('utf-8'), callback=self._pulsar_callback)
        logger.debug(f"Published action for user {user_id}", action=action)

    def _pulsar_callback(self, res, msg_id):
        logger.debug(f"Pulsar message published: {msg_id}")

    def _disconnect(self, user_id: str):
        """Cleans up a disconnected user session."""
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        if user_id in self.user_action_history:
            del self.user_action_history[user_id]
        logger.info(f"Observer session ended for user: {user_id}")

# This will be instantiated in main.py
observer_service = None 