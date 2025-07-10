# shared/q_messaging_schemas/schemas.py
from pulsar.schema import Record, String, Long, Map
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