import logging
import uuid
from typing import Dict, Any, Optional

from .agent_registry import AgentRegistry
from shared.pulsar_client import SharedPulsarClient

logger = logging.getLogger(__name__)

class TaskDispatcher:
    def __init__(self, pulsar_client: SharedPulsarClient, agent_registry: AgentRegistry):
        self.pulsar_client = pulsar_client
        self.agent_registry = agent_registry

    def dispatch_task(self, personality: str, prompt: str, workflow_id: Optional[str] = None) -> str:
        """
        Dispatches a task to an available agent with the specified personality.
        """
        agent = self.agent_registry.get_agent(personality)
        if not agent:
            raise RuntimeError(f"No available agents with personality '{personality}'")

        task_id = str(uuid.uuid4())
        
        task_data = {
            "id": task_id,
            "prompt": prompt,
            "workflow_id": workflow_id,
            "agent_personality": personality
        }
        
        self.pulsar_client.publish_message(agent.topic_name, task_data)
        
        logger.info(f"Dispatched task {task_id} to agent {agent.agent_id} on topic {agent.topic_name}")
        return task_id

# Singleton instance
task_dispatcher: Optional[TaskDispatcher] = None 