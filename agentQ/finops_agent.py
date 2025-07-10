# agentQ/finops_agent.py
import logging
import threading
import pulsar
import io
import time
import json
import fastavro

from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.finops_tools import get_cloud_cost_report_tool, get_llm_usage_stats_tool, get_k8s_resource_utilization_tool
from agentQ.app.core.prompts import FINOPS_PROMPT_TEMPLATE
from pulsar.schema import AvroSchema
from shared.q_messaging_schemas.schemas import PromptMessage, ResultMessage
from agentQ.app.main import register_with_manager, react_loop

logger = logging.getLogger(__name__)

AGENT_ID = "finops_agent"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"
REGISTRATION_TOPIC = "persistent://public/default/q.managerq.agent.registrations"

def run_finops_agent(pulsar_client, qpulse_client, llm_config, context_manager):
    """
    The main function for the FinOps agent.
    """
    logger.info("Starting FinOps Agent...")
    
    finops_toolbox = Toolbox()
    finops_toolbox.register_tool(get_cloud_cost_report_tool)
    finops_toolbox.register_tool(get_llm_usage_stats_tool)
    finops_toolbox.register_tool(get_k8s_resource_utilization_tool)

    registration_producer = pulsar_client.create_producer(REGISTRATION_TOPIC)
    result_producer = pulsar_client.create_producer(
        llm_config['result_topic'],
        schema=AvroSchema(ResultMessage)
    )
    
    register_with_manager(registration_producer, AGENT_ID, TASK_TOPIC)
    
    consumer = pulsar_client.subscribe(
        TASK_TOPIC, f"agentq-sub-{AGENT_ID}",
        schema=AvroSchema(PromptMessage)
    )

    def consumer_loop():
        while True:
            try:
                msg = consumer.receive(timeout_millis=1000)
                prompt_data = msg.value()
                
                final_result = react_loop(
                    prompt_data, 
                    context_manager, 
                    finops_toolbox, 
                    qpulse_client, 
                    llm_config, 
                    None,
                    None, # No thoughts producer
                    FINOPS_PROMPT_TEMPLATE
                )
                
                result_message = ResultMessage(
                    id=prompt_data.id,
                    result=final_result,
                    llm_model=llm_config.get('model'),
                    prompt=prompt_data.prompt,
                    timestamp=int(time.time() * 1000),
                    workflow_id=prompt_data.workflow_id,
                    task_id=prompt_data.task_id,
                    agent_personality=AGENT_ID
                )
                result_producer.send(result_message)
                
                consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error(f"Error in FinOps agent loop: {e}", exc_info=True)

    threading.Thread(target=consumer_loop, daemon=True).start()
    logger.info("FinOps Agent consumer thread started.") 