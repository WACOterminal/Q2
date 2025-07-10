
import logging
import threading
import pulsar
import json
from collections import defaultdict
from typing import List, Dict
from starlette.websockets import WebSocket
import asyncio

logger = logging.getLogger(__name__)

class LogStreamer:
    """
    Listens to a Pulsar topic for agent logs and streams them to subscribed WebSocket clients.
    """

    def __init__(self, service_url: str, log_topic: str):
        self.service_url = service_url
        self.log_topic = log_topic
        self.client: pulsar.Client = None
        self.consumer: pulsar.Consumer = None
        self._running = False
        self._thread: threading.Thread = None
        # A dictionary to hold WebSocket clients, keyed by task_id
        self.subscribers: Dict[str, List[WebSocket]] = defaultdict(list)
        self.lock = threading.Lock()

    def start(self):
        """Starts the Pulsar consumer in a background thread."""
        logger.info("Starting LogStreamer...", topic=self.log_topic)
        self.client = pulsar.Client(self.service_url)
        self.consumer = self.client.subscribe(
            self.log_topic,
            subscription_name="managerq-log-streamer-sub",
            consumer_type=pulsar.ConsumerType.Shared
        )
        self._running = True
        self._thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self._thread.start()
        logger.info("LogStreamer started.")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join()
        if self.consumer:
            self.consumer.close()
        if self.client:
            self.client.close()
        logger.info("LogStreamer stopped.")

    async def add_subscriber(self, task_id: str, websocket: WebSocket):
        """Adds a new WebSocket client to the subscriber list for a given task_id."""
        with self.lock:
            logger.info(f"New log subscriber for task_id: {task_id}")
            self.subscribers[task_id].append(websocket)

    async def remove_subscriber(self, task_id: str, websocket: WebSocket):
        """Removes a WebSocket client from the subscriber list."""
        with self.lock:
            logger.info(f"Removing log subscriber for task_id: {task_id}")
            if task_id in self.subscribers:
                self.subscribers[task_id].remove(websocket)
                if not self.subscribers[task_id]:
                    del self.subscribers[task_id]

    def _consumer_loop(self):
        """The main loop for consuming log messages."""
        while self._running:
            try:
                msg = self.consumer.receive(timeout_millis=1000)
                self.handle_message(msg)
                self.consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error(f"Error in LogStreamer loop: {e}", exc_info=True)

    def handle_message(self, msg: pulsar.Message):
        """Processes a log message and forwards it to relevant subscribers."""
        try:
            log_data = json.loads(msg.data().decode('utf-8'))
            task_id = log_data.get("task_id")

            if task_id and task_id in self.subscribers:
                # This needs to run in the main asyncio loop where websockets live
                asyncio.run(self.broadcast_to_subscribers(task_id, log_data))
        except Exception as e:
            logger.error(f"Failed to handle log message: {e}", exc_info=True)

    async def broadcast_to_subscribers(self, task_id: str, log_data: dict):
        """Broadcasts a log message to all subscribers for a given task_id."""
        with self.lock:
            subscribers = self.subscribers.get(task_id, [])
            
        message_str = json.dumps(log_data)
        
        for ws in subscribers:
            try:
                await ws.send_text(message_str)
            except Exception:
                # Handle potential disconnects
                await self.remove_subscriber(task_id, ws)

# Singleton instance - will be initialized in main.py
log_streamer: LogStreamer = None 