
import logging
from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.database_tool import query_database_tool
from agentQ.app.core.create_goal_tool import create_goal_tool

logger = logging.getLogger(__name__)

AGENT_ID = "meta_analyzer_agent"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

META_ANALYZER_SYSTEM_PROMPT = """
You are a Meta-Analyzer Agent. Your purpose is to analyze the performance of the entire AI platform and identify opportunities for improvement.

Your primary process is:
1.  **Analyze Performance:** Use the `query_workflow_database` tool to find workflows that have failed frequently or have a high number of steps.
2.  **Identify Inefficiencies:** Based on the data, identify a specific workflow template that appears to be inefficient or error-prone.
3.  **Propose an Improvement:** Formulate a clear, high-level goal for a `SoftwareEngineer` agent to improve the identified workflow template.
4.  **Delegate the Task:** Use the `create_goal` tool to create this new sub-goal. Your job is then complete.

You have these tools available:
{tools}

Begin!
"""

def setup_meta_analyzer_agent(config: dict, vault_client):
    """Sets up the toolbox for the Meta-Analyzer agent."""
    logger.info("Setting up Meta-Analyzer agent...")
    toolbox = Toolbox()
    
    # This agent needs to query the DB and create new goals
    toolbox.register_tool(query_database_tool)
    toolbox.register_tool(create_goal_tool)
    
    # We don't need a full ContextManager for this specialized agent for now
    return toolbox, None 