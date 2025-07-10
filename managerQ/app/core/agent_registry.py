import threading
import pulsar
import logging
from typing import Dict, List, Optional
import time
import random
from shared.pulsar_client import SharedPulsarClient

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, agent_id: str, personality: str, topic_name: str):
        self.agent_id = agent_id
        self.personality = personality
        self.topic_name = topic_name
        self.last_seen = time.time()

class AgentRegistry(threading.Thread):
    def __init__(self, pulsar_client: SharedPulsarClient, registration_topic: str = "persistent://public/default/q.agentq.registrations"):
        super().__init__(daemon=True)
        self.pulsar_client = pulsar_client
        self.registration_topic = registration_topic
        self.agents: Dict[str, Agent] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def run(self):
        self.pulsar_client._connect() # Ensure client is connected
        if not self.pulsar_client._client:
            logger.error("Pulsar client not available in AgentRegistry. Thread will exit.")
            return

        consumer = self.pulsar_client._client.subscribe(self.registration_topic, "managerq-registry-sub")
        logger.info(f"AgentRegistry started. Listening on {self.registration_topic}")
        while not self._stop_event.is_set():
            try:
                msg = consumer.receive(timeout_millis=1000)
                if msg:
                    data = msg.data().decode('utf-8')
                    # Assuming data is a simple comma-separated string: "agent_id,personality,topic_name"
                    agent_id, personality, topic_name = data.split(',')
                    with self._lock:
                        self.agents[agent_id] = Agent(agent_id, personality, topic_name)
                    logger.info(f"Registered/updated agent: {agent_id} ({personality})")
                    consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error(f"Error in AgentRegistry consumer loop: {e}", exc_info=True)

    def stop(self):
        self._stop_event.set()

    def get_agent(self, personality: str) -> Optional[Agent]:
        with self._lock:
            available_agents = [agent for agent in self.agents.values() if agent.personality == personality]
            if not available_agents:
                return None
            return random.choice(available_agents)

# Singleton instance
# pulsar_client must be initialized and passed in when the app starts
agent_registry: Optional[AgentRegistry] = None 