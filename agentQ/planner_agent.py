# agentQ/planner_agent.py
import logging
import threading
import pulsar
import io
import time
import json
import fastavro
import structlog
from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.context import ContextManager
from agentQ.app.core.workflow_generator_tool import workflow_generator_tool
from agentQ.app.core.collaboration_tool import collaboration_tools

from agentQ.app.core.prompts import PLANNER_SYSTEM_PROMPT, ANALYSIS_SYSTEM_PROMPT
from pulsar.schema import AvroSchema
from shared.q_messaging_schemas.schemas import PromptMessage, ResultMessage
from agentQ.app.main import register_with_manager, setup_default_agent # Import setup_default_agent
from shared.q_pulse_client.models import QPChatRequest, QPChatMessage
from shared.vault_client import VaultClient # Import VaultClient
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient

# New imports for enhanced planning
from agentQ.app.services.knowledge_graph_planner import knowledge_graph_planner, PlanningComplexity, PlanningStrategy
from agentQ.app.tools.advanced_gremlin_tools import advanced_gremlin_tools
from agentQ.app.core.toolbox import Tool

logger = structlog.get_logger("planner_agent")

# --- Agent Definition ---
AGENT_ID = "planner-agent-01"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"
REGISTRATION_TOPIC = "persistent://public/default/q.managerq.agent.registrations"

PLANNER_AGENT_SYSTEM_PROMPT = """
You are an Enhanced Master Planner AI with advanced knowledge graph reasoning capabilities. You are a strategic orchestrator with access to sophisticated planning tools and historical insights.

**Your Enhanced Capabilities:**
1. **Knowledge Graph-Driven Planning:** You can query the knowledge graph to understand dependencies, patterns, and historical contexts
2. **Advanced Pattern Recognition:** Use tools like `advanced_pattern_search` to find similar scenarios and solutions
3. **Temporal Analysis:** Leverage `temporal_trend_analysis` to understand timing patterns and trends
4. **Risk Assessment:** Use `risk_propagation_analysis` to understand potential risks and their impact
5. **Anomaly Detection:** Identify unusual patterns that might affect planning with `anomaly_detection_analysis`
6. **Intelligent Query Generation:** Convert natural language requirements into sophisticated graph queries

**Your Enhanced Decision Process:**
1.  **Deep Analysis:** Start by using `search_lessons` to find relevant historical patterns and lessons learned
2.  **Context Understanding:** Use `advanced_pattern_search` to understand dependencies and relationships for your planning domain
3.  **Risk Assessment:** Apply `risk_propagation_analysis` to understand potential risks and failure modes
4.  **Strategy Selection:** Choose between:
    *   **Static Workflows:** For well-understood, repeatable processes
    *   **Dynamic Collaboration:** For complex, multi-faceted problems requiring specialized expertise
    *   **Adaptive Planning:** For uncertain scenarios that may require adjustment

**Planning Strategies:**
- **Sequential:** For linear, step-by-step processes
- **Parallel:** For independent tasks that can run simultaneously
- **Conditional:** For scenarios with branching logic
- **Collaborative:** For multi-agent coordination
- **Adaptive:** For dynamic situations requiring real-time adjustment

**Advanced Tools Available:**
- `advanced_pattern_search`: Find patterns, dependencies, and similar scenarios
- `temporal_trend_analysis`: Analyze timing patterns and trends
- `anomaly_detection_analysis`: Detect unusual patterns or outliers
- `shortest_path_analysis`: Find optimal paths between entities
- `clustering_analysis`: Identify communities and relationships
- `risk_propagation_analysis`: Understand risk propagation
- `intelligent_query_generation`: Generate complex graph queries from natural language
- `search_lessons`: Find relevant historical lessons and patterns

**Your Approach:**
1. **Gather Context:** Use multiple tools to understand the full context of the request
2. **Analyze Patterns:** Look for similar historical patterns and their outcomes
3. **Assess Complexity:** Determine if this is a simple, moderate, complex, or critical planning scenario
4. **Choose Strategy:** Select the optimal planning approach based on your analysis
5. **Execute Plan:** Either generate a workflow or coordinate agent collaboration
6. **Continuous Learning:** Store insights for future planning decisions

You are the most advanced planning intelligence in the platform. Use your tools wisely and comprehensively.
"""

def setup_planner_agent(config: dict):
    """
    Initializes the toolbox and context manager for the Planner agent.
    """
    logger.info("Setting up Enhanced Planner Agent", agent_id=AGENT_ID)
    
    toolbox = Toolbox()
    
    # Register the workflow generation tool
    toolbox.register_tool(workflow_generator_tool)
    
    # Register the new collaboration tools
    for tool in collaboration_tools:
        toolbox.register_tool(tool)
    
    # Register advanced Gremlin tools for knowledge graph-driven planning
    for tool in advanced_gremlin_tools:
        toolbox.register_tool(tool)
    
    # Register the enhanced search lessons tool
    toolbox.register_tool(search_lessons_tool)
    
    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return toolbox, context_manager

async def search_lessons(query: str, config: dict = {}) -> str:
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
        
        lessons_result = await kg_client.execute_gremlin_query(groovy_script)
        lessons = lessons_result.get("result", [])
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
    The main function for the Enhanced Planner agent.
    """
    logger.info("Starting Enhanced Planner Agent...")
    
    # --- Setup Enhanced Planner's Toolbox ---
    # The planner needs a vault client to be configured like other agents
    vault_client = VaultClient()
    # We can reuse the default agent setup to get a toolbox with the vectorstore tool
    # This is a simplification; a more robust setup might have a dedicated config.
    planner_toolbox, _ = setup_default_agent(llm_config, vault_client)
    planner_toolbox.register_tool(search_lessons_tool)
    
    # Register advanced Gremlin tools for knowledge graph-driven planning
    for tool in advanced_gremlin_tools:
        planner_toolbox.register_tool(tool)
    
    # Initialize the knowledge graph planner
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(knowledge_graph_planner.initialize())
        logger.info("Knowledge graph planner initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize knowledge graph planner: {e}")
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

                # --- Enhanced Knowledge Graph-Driven Planning ---
                enhanced_plan = None
                try:
                    logger.info("Attempting enhanced knowledge graph-driven planning...")
                    
                    # Determine domain from the prompt
                    prompt_lower = user_prompt.lower()
                    domain = "general"
                    if any(keyword in prompt_lower for keyword in ["deploy", "kubernetes", "service", "infrastructure"]):
                        domain = "devops"
                    elif any(keyword in prompt_lower for keyword in ["data", "analysis", "query", "database"]):
                        domain = "data"
                    elif any(keyword in prompt_lower for keyword in ["security", "vulnerability", "patch", "auth"]):
                        domain = "security"
                    elif any(keyword in prompt_lower for keyword in ["machine learning", "model", "training", "ml"]):
                        domain = "ml"
                    
                    # Use the enhanced planner for complex scenarios
                    if any(keyword in prompt_lower for keyword in ["complex", "multi", "coordinate", "analyze", "optimize"]):
                        logger.info(f"Using enhanced planner for domain: {domain}")
                        import asyncio
                        loop = asyncio.get_event_loop()
                        enhanced_plan = loop.run_until_complete(knowledge_graph_planner.create_enhanced_plan(
                            prompt=user_prompt,
                            user_id="planner_agent",
                            domain=domain
                        ))
                        
                        # Convert enhanced plan to the expected format
                        if enhanced_plan:
                            plan_json = {
                                "plan_type": "enhanced",
                                "strategy": enhanced_plan.strategy.value,
                                "complexity": enhanced_plan.context.complexity.value,
                                "workflow": enhanced_plan.workflow.dict() if hasattr(enhanced_plan.workflow, 'dict') else str(enhanced_plan.workflow),
                                "insights": [{"type": i.insight_type, "confidence": i.confidence, "reasoning": i.reasoning} for i in enhanced_plan.insights],
                                "risk_assessment": enhanced_plan.risk_assessment,
                                "success_probability": enhanced_plan.recommendations[0].success_probability if enhanced_plan.recommendations else 0.8
                            }
                            plan_json_str = json.dumps(plan_json)
                            logger.info("Enhanced plan generated successfully")
                        else:
                            logger.info("Enhanced planner returned no plan, falling back to traditional approach")
                            enhanced_plan = None
                    
                except Exception as e:
                    logger.warning(f"Enhanced planning failed, falling back to traditional approach: {e}")
                    enhanced_plan = None

                # --- Fallback: Traditional Insight Retrieval ---
                if not enhanced_plan:
                    logger.info("Using traditional planning approach...")
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
                
                # --- Traditional Planning (when enhanced planning not used) ---
                if not enhanced_plan:
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
                        
                        # Use enhanced system prompt for traditional planning
                        planner_prompt = (
                            f"{PLANNER_AGENT_SYSTEM_PROMPT}\n\n"
                            f"**Available Insights:**\n{retrieved_insights}\n\n"
                            f"**Relevant Lessons:**\n{retrieved_lessons}\n\n"
                            f"**Goal Summary:**\n{summary}\n\n"
                            f"**High-Level Steps:**\n{high_level_steps}\n\n"
                            f"**Original User Request:**\n{user_prompt}"
                        )
                        
                        messages = [QPChatMessage(role="system", content=planner_prompt)]
                        request = QPChatRequest(model=llm_config['model'], messages=messages, temperature=0.0)
                        response = qpulse_client.get_chat_completion(request)
                        plan_json_str = response.choices[0].message.content

                # --- Result Handling ---
                if enhanced_plan:
                    logger.info(f"Generated enhanced plan with strategy: {enhanced_plan.strategy.value}")
                    logger.info(f"Plan complexity: {enhanced_plan.context.complexity.value}")
                    logger.info(f"Number of insights: {len(enhanced_plan.insights)}")
                    llm_model = "knowledge_graph_planner"
                else:
                    logger.info(f"Generated traditional plan: {plan_json_str[:200]}...")
                    llm_model = response.model if 'response' in locals() else llm_config['model']
                
                # The result is the raw JSON string of the plan or the error
                result_message = ResultMessage(
                    id=prompt_data.id,
                    result=plan_json_str,
                    llm_model=llm_model,
                    prompt=prompt_data.prompt,
                    timestamp=int(time.time() * 1000),
                    workflow_id=prompt_data.workflow_id,
                    task_id=prompt_data.task_id,
                    agent_personality=AGENT_ID
                )
                result_producer.send(result_message)
                
                if enhanced_plan:
                    logger.info(f"Enhanced plan sent for task: {prompt_data.task_id}")
                else:
                    logger.info(f"Traditional plan sent for task: {prompt_data.task_id}")
                
                consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error(f"Error in Enhanced Planner agent loop: {e}", exc_info=True)
                
                # Send error result if possible
                try:
                    error_result = {
                        "error": "PLANNING_ERROR",
                        "message": f"An error occurred during planning: {str(e)}",
                        "fallback_available": True
                    }
                    error_message = ResultMessage(
                        id=getattr(prompt_data, 'id', 'unknown'),
                        result=json.dumps(error_result),
                        llm_model="error_handler",
                        prompt=getattr(prompt_data, 'prompt', 'unknown'),
                        timestamp=int(time.time() * 1000),
                        workflow_id=getattr(prompt_data, 'workflow_id', 'unknown'),
                        task_id=getattr(prompt_data, 'task_id', 'unknown'),
                        agent_personality=AGENT_ID
                    )
                    result_producer.send(error_message)
                except Exception as send_error:
                    logger.error(f"Failed to send error result: {send_error}", exc_info=True)

    threading.Thread(target=consumer_loop, daemon=True).start()
    logger.info("Enhanced Planner Agent consumer thread started with knowledge graph capabilities.") 