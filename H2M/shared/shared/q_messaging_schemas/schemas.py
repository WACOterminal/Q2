# shared/q_messaging_schemas/schemas.py
from pulsar.schema import Record, String, Long, Map, List, Float
from pulsar.schema.definition import Nullable

class PromptMessage(Record):
    id = String()
    prompt = String()
    model = String()
    timestamp = Long()
    workflow_id = Nullable(String())
    task_id = Nullable(String())
    agent_personality = Nullable(String())

class ResultMessage(Record):
    id = String()
    result = String()
    llm_model = String()
    prompt = String()
    timestamp = Long()
    workflow_id = Nullable(String())
    task_id = Nullable(String())
    agent_personality = Nullable(String())

class AgentRegistration(Record):
    agent_id = String()
    task_topic = String()

class ThoughtMessage(Record):
    conversation_id = String()
    thought = String()
    timestamp = Long()

class LogMessage(Record):
    workflow_id = Nullable(String())
    task_id = Nullable(String())
    agent_id = Nullable(String())
    level = String()
    message = String()
    timestamp = Long()
    details = Map(String(), String())

class TaskAnnouncement(Record):
    """
    Schema for broadcasting a task to potential agents for bidding.
    Sent by the TaskDispatcher.
    """
    task_id = String(required=True)
    task_prompt = String(required=True)
    task_personality = String(required=True)
    broadcast_time = Long(required=True)
    bid_window_seconds = Float(required=True)
    
    # Optional context for better bidding decisions
    workflow_id = Nullable(String())
    user_id = Nullable(String())
    tools_required = Nullable(List(String()))
    resource_requirements = Nullable(Map(String(), String())) # e.g. {"cpu": "0.5", "memory_gb": "1"}

class AgentBid(Record):
    """
    Schema for an agent's bid in response to a TaskAnnouncement.
    Sent by an agentQ instance.
    """
    task_id = String(required=True)
    agent_id = String(required=True)
    
    # The agent's bid. Could be cost, time, or a composite score. Lower is better.
    bid_value = Float(required=True) 
    
    # Agent's self-assessed capabilities for the dispatcher's reference
    can_meet_requirements = String(required=True) # Using string "true" or "false"
    confidence_score = Float(required=True) # e.g., 0.95
    
    # Agent's current load
    current_load_factor = Float(required=True) # e.g., 0.3
    
    timestamp = Long(required=True) 