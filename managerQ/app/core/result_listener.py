import logging
import pulsar
import fastavro
import io
import threading
import time
import asyncio
import json
from typing import Dict, Any, Optional

from managerQ.app.core.workflow_manager import workflow_manager
from managerQ.app.core.task_dispatcher import task_dispatcher
from managerQ.app.models import TaskStatus
from managerQ.app.models import WorkflowEvent
from managerQ.app.api.dashboard_ws import broadcast_workflow_event
from managerQ.app.core.observability_manager import observability_manager
from shared.pulsar_client import SharedPulsarClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Avro schema for the result messages, must match agentQ's schema
RESULT_SCHEMA = fastavro.parse_schema({
    "namespace": "q.agentq", "type": "record", "name": "ResultMessage",
    "fields": [
        {"name": "id", "type": "string"},
        {"name": "result", "type": "string"},
        {"name": "llm_model", "type": "string"},
        {"name": "prompt", "type": "string"},
        {"name": "timestamp", "type": "long"},
        # New fields for workflow context
        {"name": "workflow_id", "type": ["null", "string"], "default": None},
        {"name": "task_id", "type": ["null", "string"], "default": None},
        {"name": "agent_personality", "type": ["null", "string"], "default": None},
    ]
})

class ResultListener(threading.Thread):
    def __init__(self, pulsar_client: SharedPulsarClient, results_topic: str = "persistent://public/default/q.agentq.results"):
        super().__init__(daemon=True)
        self.pulsar_client = pulsar_client
        self.results_topic = results_topic
        self.results: Dict[str, Any] = {}
        self.events: Dict[str, asyncio.Event] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def run(self):
        self.pulsar_client._connect()
        if not self.pulsar_client._client:
            logger.error("Pulsar client not available in ResultListener. Thread will exit.")
            return

        consumer = self.pulsar_client._client.subscribe(self.results_topic, "managerq-results-sub")
        logger.info(f"ResultListener started. Listening on {self.results_topic}")
        while not self._stop_event.is_set():
            try:
                msg = consumer.receive(timeout_millis=1000)
                if msg:
                    # Assuming the message is a simple string: "task_id,result_data"
                    data = msg.data().decode('utf-8')
                    task_id, result_data = data.split(',', 1)
                    
                    with self._lock:
                        self.results[task_id] = result_data
                        if task_id in self.events:
                            self.events[task_id].set()
                            
                    consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error(f"Error in ResultListener consumer loop: {e}", exc_info=True)

    def stop(self):
        self._stop_event.set()

    async def wait_for_result(self, task_id: str, timeout: int = 60) -> Any:
        with self._lock:
            if task_id in self.results:
                return self.results.pop(task_id)
            
            event = asyncio.Event()
            self.events[task_id] = event
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timed out waiting for result for task {task_id}")
        finally:
            with self._lock:
                self.events.pop(task_id, None)

        with self._lock:
            return self.results.pop(task_id, None)

# Singleton instance
result_listener: Optional[ResultListener] = None 