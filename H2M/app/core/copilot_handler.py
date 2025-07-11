import asyncio
import logging
import json
import pulsar
from typing import Dict, List, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class CoPilotHandler:
    def __init__(self, pulsar_client: pulsar.Client):
        self.client = pulsar_client
        self.active_sessions: Dict[str, WebSocket] = {} # conversation_id -> websocket
        self._running = False
        self._consumer_thread = None

    def start(self):
        """Starts the handler in a background thread."""
        self._running = True
        # This should be a daemon thread so it exits when the main app does
        self._consumer_thread = asyncio.create_task(self._consumer_loop())
        logger.info("CoPilotHandler started.")

    def stop(self):
        """Stops the handler."""
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.cancel()
        logger.info("CoPilotHandler stopped.")

    async def register_session(self, conversation_id: str, websocket: WebSocket):
        """Registers a new co-piloting session for a user."""
        await websocket.accept()
        self.active_sessions[conversation_id] = websocket
        logger.info(f"New co-pilot session registered for conversation: {conversation_id}")

    def unregister_session(self, conversation_id: str):
        """Unregisters a co-piloting session."""
        if conversation_id in self.active_sessions:
            del self.active_sessions[conversation_id]
            logger.info(f"Co-pilot session unregistered for conversation: {conversation_id}")

    async def handle_human_response(self, response_data: Dict[str, Any]):
        """Handles a response from the human co-pilot via WebSocket."""
        reply_topic = response_data.get("reply_topic")
        if not reply_topic:
            logger.error("No reply_topic in human response.")
            return
            
        producer = self.client.create_producer(reply_topic)
        producer.send(json.dumps(response_data).encode('utf-8'))
        producer.close()
        logger.info(f"Relayed human response to topic: {reply_topic}")

    async def _consumer_loop(self):
        """The main loop for consuming agent co-pilot requests."""
        consumer = self.client.subscribe(
            'persistent://public/default/q.h2m.copilot_requests',
            subscription_name='h2m-copilot-handler-sub'
        )
        while self._running:
            try:
                msg = consumer.receive(timeout_millis=1000)
                await self._process_agent_request(msg)
                consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error(f"Error in CoPilotHandler consumer loop: {e}", exc_info=True)
                if 'msg' in locals():
                    consumer.negative_acknowledge(msg)
                await asyncio.sleep(5)
    
    async def _process_agent_request(self, msg: pulsar.Message):
        """Processes a request from an agent and forwards it to the UI."""
        try:
            payload = json.loads(msg.data().decode('utf-8'))
            conversation_id = payload.get("conversation_id")
            
            if conversation_id in self.active_sessions:
                websocket = self.active_sessions[conversation_id]
                # Forward the agent's request directly to the UI
                await websocket.send_text(json.dumps(payload))
                logger.info(f"Forwarded agent request to UI for conversation: {conversation_id}")
            else:
                logger.warning(f"Received co-pilot request for inactive session: {conversation_id}")
        except Exception as e:
            logger.error("Failed to process agent request", exc_info=True)

# This will be instantiated in main.py
copilot_handler = CoPilotHandler(pulsar_client=None) # Placeholder 