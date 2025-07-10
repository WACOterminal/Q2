# agentQ/planner_agent.py
import logging
import threading
import pulsar
import io
import time
import json
import fastavro

from agentQ.app.core.prompts import PLANNER_SYSTEM_PROMPT, ANALYSIS_SYSTEM_PROMPT
from pulsar.schema import AvroSchema
from shared.q_messaging_schemas.schemas import PromptMessage, ResultMessage
from agentQ.app.main import register_with_manager, setup_default_agent # Import setup_default_agent
from shared.q_pulse_client.models import QPChatRequest, QPChatMessage
from shared.vault_client import VaultClient # Import VaultClient
from agentQ.app.core.toolbox import Tool
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient

logger = logging.getLogger(__name__)

AGENT_ID = "planner_agent"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"
REGISTRATION_TOPIC = "persistent://public/default/q.managerq.agent.registrations"

def search_lessons(query: str, config: dict = {}) -> str:
    """
    Searches the knowledge graph for lessons related to the user's query.
    """
    logger.info("Searching for relevant lessons in the knowledge graph.")
    knowledgegraph_url = config.get('services', {}).get('knowledgegraphq_url', 'http://localhost:8003')
    service_token = config.get('service_token')

    if not service_token:
        return "Error: Service token not available. Cannot search for lessons."

    try:
        kg_client = KnowledgeGraphClient(base_url=knowledgegraph_url, token=service_token)
        
        # Simple keyword extraction. In a real system, this could be more advanced.
        keywords = set(query.lower().split())
        
        # Gremlin query to find lessons related to the keywords
        groovy_script = f"""
            g.V().hasLabel('tool', 'concept').where(values('name').is(within({str(list(keywords))})))
             .in('RELATES_TO').hasLabel('lesson').valueMap('lesson_text', 'final_status').limit(5).toList()
        """
        
        lessons = kg_client.run_gremlin_script(groovy_script)
        if not lessons:
            return "No relevant lessons found."
        
        # Format the lessons for injection into the prompt
        formatted_lessons = []
        for lesson in lessons:
            text = lesson.get('lesson_text', [''])[0]
            status = lesson.get('final_status', [''])[0]
            formatted_lessons.append(f"- (From a {status} workflow): {text}")
            
        return "\\n".join(formatted_lessons)

    except Exception as e:
        logger.error(f"Failed to search for lessons in KnowledgeGraph: {e}", exc_info=True)
        return "Error: An unexpected error occurred while searching for lessons."

search_lessons_tool = Tool(
    name="search_lessons",
    description="Searches the knowledge graph for lessons learned from past workflows that are relevant to the current query.",
    func=search_lessons
)

def run_planner_agent(pulsar_client, qpulse_client, llm_config):
    """
    The main function for the Planner agent.
    """
    logger.info("Starting Planner Agent...")
    
    # --- Setup Planner's Toolbox ---
    # The planner needs a vault client to be configured like other agents
    vault_client = VaultClient()
    # We can reuse the default agent setup to get a toolbox with the vectorstore tool
    # This is a simplification; a more robust setup might have a dedicated config.
    planner_toolbox, _ = setup_default_agent(llm_config, vault_client)
    planner_toolbox.register_tool(search_lessons_tool)
    # -----------------------------

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
                
                logger.info(f"Planner agent received goal: {prompt_data.prompt}")
                
                user_prompt = prompt_data.prompt

                # --- Insight Retrieval Step ---
                retrieved_insights = ""
                retrieved_lessons = ""
                try:
                    logger.info("Planner searching for relevant insights and lessons...")
                    # Search the 'insights' collection specifically
                    insights_result = planner_toolbox.execute_tool("search_knowledge_base", query=user_prompt, collection="insights")
                    if "No relevant information found" not in insights_result:
                        retrieved_insights = insights_result
                        logger.info("Found relevant insights for planning.")
                    
                    # New: Search for structured lessons
                    lessons_result = planner_toolbox.execute_tool("search_lessons", query=user_prompt)
                    if "No relevant lessons found" not in lessons_result:
                        retrieved_lessons = lessons_result
                        logger.info("Found relevant lessons for planning.")

                except Exception as e:
                    logger.error(f"Planner failed to retrieve insights or lessons: {e}", exc_info=True)
                
                # --- New: Two-Phase Planning ---
                
                # 1. First-pass analysis
                analysis_prompt = f"{ANALYSIS_SYSTEM_PROMPT.format(insights=retrieved_insights, lessons=retrieved_lessons)}\n\n**User Request:**\n{user_prompt}"
                analysis_messages = [QPChatMessage(role="system", content=analysis_prompt)]
                analysis_request = QPChatRequest(model=llm_config['model'], messages=analysis_messages, temperature=0.0, max_tokens=512)
                analysis_response = qpulse_client.get_chat_completion(analysis_request)
                analysis_json = json.loads(analysis_response.choices[0].message.content)

                if analysis_json.get("is_ambiguous"):
                    # Goal is unclear, return an error structure that managerQ can interpret
                    error_result = {
                        "error": "AMBIGUOUS_GOAL",
                        "clarifying_question": analysis_json.get("clarifying_question", "The goal is unclear, please provide more detail.")
                    }
                    plan_json_str = json.dumps(error_result)
                else:
                    # 2. Goal is clear, proceed to generate the full plan
                    high_level_steps = "\n".join(f"- {step}" for step in analysis_json.get("high_level_steps", []))
                    summary = analysis_json.get("summary", "")
                    
                    planner_prompt = (
                        f"{PLANNER_SYSTEM_PROMPT.format(insights=retrieved_insights, lessons=retrieved_lessons)}\n\n"
                        f"**Goal Summary:**\n{summary}\n\n"
                        f"**High-Level Steps:**\n{high_level_steps}\n\n"
                        f"**Original User Request:**\n{user_prompt}"
                    )
                    
                    messages = [QPChatMessage(role="system", content=planner_prompt)]
                    request = QPChatRequest(model=llm_config['model'], messages=messages, temperature=0.0)
                    response = qpulse_client.get_chat_completion(request)
                    plan_json_str = response.choices[0].message.content

                # The result is the raw JSON string of the plan or the error
                result_message = ResultMessage(
                    id=prompt_data.id,
                    result=plan_json_str,
                    llm_model=response.model,
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
                logger.error(f"Error in Planner agent loop: {e}", exc_info=True)

    threading.Thread(target=consumer_loop, daemon=True).start()
    logger.info("Planner Agent consumer thread started.") 